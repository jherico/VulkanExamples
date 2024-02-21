/*
 * Vulkan Example - Multi threaded command buffer generation and rendering
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <common/random.hpp>
#include <common/threadpool.hpp>
#include <examples/example.hpp>
#include <rendering/frustum.hpp>

#include <shaders/multithreading/phong.frag.inl>
#include <shaders/multithreading/phong.vert.inl>
#include <shaders/multithreading/starsphere.frag.inl>
#include <shaders/multithreading/starsphere.vert.inl>

// Vertex layout used in this example
// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

constexpr glm::vec3 SPHERE_SCALE{ 35.0f, 0.0f, 35.0f };
constexpr float M_TAU = static_cast<float>(2.0f * M_PI);

class VulkanExample : public vkx::ExampleBase {
public:
    bool displayStarSphere = true;
    struct {
        vks::model::Model ufo;
        vks::model::Model skysphere;
    } meshes;

    // Shared matrices used for thread push constant blocks
    struct {
        glm::mat4 projection;
        glm::mat4 view;
    } matrices;

    struct {
        vk::Pipeline phong;
        vk::Pipeline starsphere;
    } pipelines;

    vk::PipelineLayout pipelineLayout;

    std::vector<vk::CommandBuffer> skyboxCommandBuffers;

    // Number of animated objects to be renderer
    // by using threads and secondary command buffers
    uint32_t numObjectsPerThread;

    // Multi threaded stuff
    // Max. number of concurrent threads
    static uint32_t numThreads;

    struct ObjectData {
        glm::vec3 pos;
        float rotationAxis;
        float rotationDir;
        float rotationSpeed;
        float scale;
        float deltaT;
        bool visible = true;

        // Use push constants to update shader
        // parameters on a per-thread base
        struct PushConstant {
            glm::mat4 mvp;
            glm::vec3 color;
        } pushConstant;

        void init(vkx::Random& random) {
            pos = random.sphere(SPHERE_SCALE);
            rotationAxis = random.radian();
            deltaT = random.real();
            rotationDir = random.boolean() ? 1.0f : -1.0f;
            // Rotate at between 2 and 6 degrees per second
            rotationSpeed = glm::radians(random.real(2.0f, 6.0f)) * rotationDir;
            scale = random.real(0.75f, 1.25f);
            pushConstant.color = random.color();
        }

        void update(float frameTimer) {
            rotationAxis += 2.5f * rotationSpeed * frameTimer;
            rotationAxis = fmod(rotationAxis, M_TAU);
            deltaT += 0.15f * frameTimer;
            deltaT = fmod(deltaT, 1.0f);
            pos.y = sin(deltaT * M_TAU) * 2.5f;
        }

        glm::mat4 getModel() {
            auto model = glm::translate(glm::mat4(1.0f), pos);
            model = glm::rotate(model, -sinf(deltaT * M_TAU) * 0.25f, glm::vec3(rotationDir, 0.0f, 0.0f));
            model = glm::rotate(model, rotationAxis, glm::vec3(0.0f, rotationDir, 0.0f));
            model = glm::rotate(model, deltaT * M_TAU, glm::vec3(0.0f, rotationDir, 0.0f));
            model = glm::scale(model, glm::vec3(scale));
            return model;
        }
    };

    struct ThreadData {
        vk::CommandPool commandPool;
        // One command buffer per render object
        std::vector<vk::CommandBuffer> commandBuffer;
        // Per object information (position, rotation, etc.)
        std::vector<ObjectData> objectsData;

        void destroy() {
            const auto& device = vks::Context::get().device;
            if (!commandBuffer.empty()) {
                device.free(commandPool, commandBuffer);
                commandBuffer.clear();
            }
            if (commandPool) {
                device.destroy(commandPool);
                commandPool = nullptr;
            }
        }

        void prepare(uint32_t numObjectsPerThread, uint32_t swapchainImageCount, vkx::Random& random) {
            auto& context = vks::Context::get();
            auto& device = context.device;
            // Create one command pool for each thread
            commandPool = device.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, context.queuesInfo.graphics.index });
            // One secondary command buffer per object that is updated by this thread
            commandBuffer = device.allocateCommandBuffers({ commandPool, vk::CommandBufferLevel::eSecondary, numObjectsPerThread * swapchainImageCount });
            // Generate secondary command buffers for each thread
            objectsData.resize(numObjectsPerThread);
            for (uint32_t j = 0; j < numObjectsPerThread; j++) {
                objectsData[j].init(random);
            }
        }
    };
    std::vector<ThreadData> threadData;

    vkx::ThreadPool threadPool;

    // Max. dimension of the ufo mesh for use as the sphere
    // radius for frustum culling
    float objectSphereDim;

    // View frustum for culling invisible objects
    vks::Frustum frustum;

    vkx::Random random;

    VulkanExample() {
        // zoom = -32.5f;
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ 0.0f, 37.5f, 0.0f });
        // enableTextOverlay = true;
        title = "Vulkan Example - Multi threaded rendering";
        // Get number of max. concurrrent threads
        numThreads = std::thread::hardware_concurrency();
        assert(numThreads > 0);
        std::cout << "numThreads = " << numThreads << std::endl;
        srand(time(NULL));

        threadPool.setThreadCount(numThreads);

        numObjectsPerThread = 256 / numThreads;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.phong);
        device.destroy(pipelines.starsphere);

        device.destroy(pipelineLayout);

        graphicsQueue.freeCommandBuffers(skyboxCommandBuffers);
        skyboxCommandBuffers.clear();

        meshes.ufo.destroy();
        meshes.skysphere.destroy();

        for (auto& thread : threadData) {
            thread.destroy();
        }
    }

    // Create all threads and initialize shader push constants
    void prepareMultiThreadedRenderer() {
        // Create a secondary command buffer for rendering the star sphere
        skyboxCommandBuffers = graphicsQueue.allocateCommandBuffers(swapChain.imageCount, vk::CommandBufferLevel::eSecondary);

        threadData.resize(numThreads);

        for (auto& thread : threadData) {
            thread.prepare(numObjectsPerThread, swapChain.imageCount, random);
        }
    }

    const vk::CommandBuffer& getCurrentThreadCommandBuffer(uint32_t threadIndex, uint32_t cmdBufferIndex) {
        return threadData[threadIndex].commandBuffer[(cmdBufferIndex * swapChain.imageCount) + currentIndex];
    }

    // Builds the secondary command buffer for each thread
    void threadRenderCode(uint32_t threadIndex, uint32_t cmdBufferIndex, const vk::CommandBufferInheritanceInfo& inheritanceInfo) {
        ThreadData* thread = &threadData[threadIndex];
        auto& objectData = thread->objectsData[cmdBufferIndex];
        auto& cmdBuffer = getCurrentThreadCommandBuffer(threadIndex, cmdBufferIndex);

        // Check visibility against view frustum
        objectData.visible = frustum.checkSphere(objectData.pos, objectSphereDim * 0.5f);

        if (!objectData.visible) {
            return;
        }

        cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eRenderPassContinue, &inheritanceInfo });
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phong);

        // Update
        if (!paused) {
            objectData.update(frameTimer);
        }

        objectData.pushConstant.mvp = matrices.projection * matrices.view * objectData.getModel();

        // Update shader push constant block
        // Contains model view matrix
        cmdBuffer.pushConstants<ObjectData::PushConstant>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, objectData.pushConstant);

        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, meshes.ufo.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.ufo.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.ufo.indexCount, 1, 0, 0, 0);

        cmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer&) override {}

    void updateSkyboxCommandBuffer(const vk::CommandBufferInheritanceInfo& inheritanceInfo) {
        // Secondary command buffer for the sky sphere
        vk::CommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue;
        commandBufferBeginInfo.pInheritanceInfo = &inheritanceInfo;
        const auto& skyboxCommandBuffer = skyboxCommandBuffers[currentIndex];
        skyboxCommandBuffer.begin(commandBufferBeginInfo);
        skyboxCommandBuffer.setViewport(0, vks::util::viewport(size));
        skyboxCommandBuffer.setScissor(0, vks::util::rect2D(size));
        skyboxCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.starsphere);

        glm::mat4 mvp = matrices.projection * matrices.view;
        mvp[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        mvp = glm::scale(mvp, glm::vec3(2.0f));
        skyboxCommandBuffer.pushConstants<glm::mat4>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, mvp);

        vk::DeviceSize offsets = 0;
        skyboxCommandBuffer.bindVertexBuffers(0, meshes.skysphere.vertices.buffer, offsets);
        skyboxCommandBuffer.bindIndexBuffer(meshes.skysphere.indices.buffer, 0, vk::IndexType::eUint32);
        skyboxCommandBuffer.drawIndexed(meshes.skysphere.indexCount, 1, 0, 0, 0);

        skyboxCommandBuffer.end();
    }

    void buildCommandBuffers() override {
        perImageData.resize(swapChain.imageCount);
        auto commandBuffers = graphicsQueue.allocateCommandBuffers(swapChain.imageCount);
        // Destroy and recreate command buffers if already present
        vk::CommandBufferBeginInfo cmdBufInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse };
        for (currentIndex = 0; currentIndex < swapChain.imageCount; ++currentIndex) {
            perImageData[currentIndex].commandBuffer = commandBuffers[currentIndex];
        }
    }

    // Updates the secondary command buffers using a thread pool
    // and puts them into the primary command buffer that's
    // lat submitted to the graphicsQueue for rendering
    void drawCurrentCommandBuffer() override {
        auto& swapChainImage = swapChain.images[currentIndex];

        vk::RenderingAttachmentInfo colorAttachmentInfo;
        colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
        colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachmentInfo.clearValue = vks::util::clearColor({ 0.0f, 0.0f, 0.2f, 0.0f });
        colorAttachmentInfo.imageView = swapChainImage.view;

        vk::RenderingAttachmentInfo depthAttachmentInfo;
        depthAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
        depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachmentInfo.clearValue = vk::ClearDepthStencilValue{ 1.0, 0 };
        depthAttachmentInfo.imageView = depthStencilView;

        vk::RenderingInfo renderingInfo;
        renderingInfo.flags = vk::RenderingFlagBits::eContentsSecondaryCommandBuffers;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.layerCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;
        renderingInfo.pDepthAttachment = &depthAttachmentInfo;
        renderingInfo.pStencilAttachment = &depthAttachmentInfo;
        renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, size };

        // Inheritance info for the secondary command buffers
        vk::CommandBufferInheritanceRenderingInfo inheritanceRenderingInfo;
        inheritanceRenderingInfo.colorAttachmentCount = 1;
        inheritanceRenderingInfo.pColorAttachmentFormats = &defaultColorFormat;
        inheritanceRenderingInfo.depthAttachmentFormat = defaultDepthStencilFormat;
        inheritanceRenderingInfo.stencilAttachmentFormat = defaultDepthStencilFormat;
        vk::CommandBufferInheritanceInfo inheritanceInfo;
        inheritanceInfo.pNext = &inheritanceRenderingInfo;

        // Contains the list of secondary command buffers to be executed
        std::vector<vk::CommandBuffer> secondaryCommandBuffers;
        secondaryCommandBuffers.reserve((numThreads * numObjectsPerThread) + 1);
        if (displayStarSphere) {
            // Secondary command buffer with star background sphere
            updateSkyboxCommandBuffer(inheritanceInfo);
            secondaryCommandBuffers.push_back(skyboxCommandBuffers[currentIndex]);
        }

        // Add a job to the thread's graphicsQueue for each object to be rendered
        for (uint32_t t = 0; t < numThreads; t++) {
            for (uint32_t i = 0; i < numObjectsPerThread; i++) {
                threadPool.threads[t]->addJob([=] { threadRenderCode(t, i, inheritanceInfo); });
            }
        }
        threadPool.wait();
        // Only submit if object is within the current view frustum
        for (uint32_t t = 0; t < numThreads; t++) {
            for (uint32_t i = 0; i < numObjectsPerThread; i++) {
                if (threadData[t].objectsData[i].visible) {
                    secondaryCommandBuffers.push_back(getCurrentThreadCommandBuffer(t, i));
                }
            }
        }

        // Build the top level command buffer
        using namespace vks::util;
        // Set target frame buffer
        auto& primaryCommandBuffer = perImageData[currentIndex].commandBuffer;
        primaryCommandBuffer.begin(vk::CommandBufferBeginInfo{});
        // We're not using renderpasses so we need to transition the swapchain image manually
        swapChainImage.setLayout(primaryCommandBuffer, ImageTransitionState::COLOR_ATTACHMENT);
        // The primary command buffer does not contain any rendering commands
        // These are stored (and retrieved) from the secondary command buffers
        primaryCommandBuffer.beginRendering(renderingInfo);
        // Execute render commands from the secondary command buffer
        primaryCommandBuffer.executeCommands(secondaryCommandBuffers);
        primaryCommandBuffer.endRendering();
        // We're not using renderpasses so we need to transition the swapchain image manually
        swapChainImage.setLayout(primaryCommandBuffer, ImageTransitionState::PRESENT);
        primaryCommandBuffer.end();

        queueCommandBuffer(primaryCommandBuffer, vkx::RenderStates::RENDER_SCENE, vk::PipelineStageFlagBits2::eColorAttachmentOutput, true);
    }

    void loadMeshes() {
        meshes.ufo.loadFromFile(getAssetPath() + "models/retroufo_red_lowpoly.dae", vertexLayout, 0.12f);
        meshes.skysphere.loadFromFile(getAssetPath() + "models/sphere.obj", vertexLayout, 1.0f);
        objectSphereDim = std::max(std::max(meshes.ufo.dim.max.x, meshes.ufo.dim.max.y), meshes.ufo.dim.max.z);
    }

    void setupPipelineLayout() {
        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo;
        // Push constants for model matrices
        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(ObjectData::PushConstant) };
        // Push constant ranges are part of the pipeline layout
        pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
        pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::multithreading::phong::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::multithreading::phong::frag, vk::ShaderStageFlagBits::eFragment);

        // Solid rendering pipeline
        pipelines.phong = pipelineBuilder.create(context.pipelineCache);

        // Star sphere rendering pipeline
        pipelineBuilder.destroyShaderModules();
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        pipelineBuilder.depthStencilState = false;
        pipelineBuilder.loadShader(vkx::shaders::multithreading::starsphere::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::multithreading::starsphere::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.starsphere = pipelineBuilder.create(context.pipelineCache);
    }

    void updateMatrices() {
        matrices.projection = camera.matrices.perspective;
        matrices.view = camera.matrices.view;
        frustum.update(matrices.projection * matrices.view);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadMeshes();
        setupPipelineLayout();
        preparePipelines();
        prepareMultiThreadedRenderer();
        updateMatrices();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateMatrices(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Statistics")) {
            ui.text("Active threads: %d", numThreads);
        }
        if (ui.header("Settings")) {
            ui.checkBox("Stars", &displayStarSphere);
        }
    }
};

uint32_t VulkanExample::numThreads = std::thread::hardware_concurrency();

VULKAN_EXAMPLE_MAIN()