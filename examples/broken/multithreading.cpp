/*
* Vulkan Example - Multi threaded command buffer generation and rendering
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"

#include "threadPool.hpp"
#include "frustum.hpp"


// Vertex layout used in this example
// Vertex layout for this example
std::vector<vkx::VertexLayout> vertexLayout =
{
    vkx::VertexLayout::VERTEX_LAYOUT_POSITION,
    vkx::VertexLayout::VERTEX_LAYOUT_NORMAL,
    vkx::VertexLayout::VERTEX_LAYOUT_COLOR,
};

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vk::PipelineVertexInputStateCreateInfo inputState;
        std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    struct {
        vkx::MeshBuffer ufo;
        vkx::MeshBuffer skysphere;
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

    vk::CommandBuffer primaryCommandBuffer;
    vk::CommandBuffer secondaryCommandBuffer;

    // Number of animated objects to be renderer
    // by using threads and secondary command buffers
    uint32_t numObjectsPerThread;

    // Multi threaded stuff
    // Max. number of concurrent threads
    uint32_t numThreads;

    // Use push constants to update shader
    // parameters on a per-thread base
    struct ThreadPushConstantBlock {
        glm::mat4 mvp;
        glm::vec3 color;
    };

    struct ObjectData {
        glm::mat4 model;
        glm::vec3 pos;
        glm::vec3 rotation;
        float rotationDir;
        float rotationSpeed;
        float scale;
        float deltaT;
        float stateT = 0;
        bool visible = true;
    };

    struct ThreadData {
        vkx::MeshBuffer mesh;
        vk::CommandPool commandPool;
        // One command buffer per render object
        std::vector<vk::CommandBuffer> commandBuffer;
        // One push constant block per render object
        std::vector<ThreadPushConstantBlock> pushConstBlock;
        // Per object information (position, rotation, etc.)
        std::vector<ObjectData> objectData;
    };
    std::vector<ThreadData> threadData;

    vkx::ThreadPool threadPool;

    // vk::Fence to wait for all command buffers to finish before
    // presenting to the swap chain
    vk::Fence renderFence;

    // Max. dimension of the ufo mesh for use as the sphere
    // radius for frustum culling
    float objectSphereDim;

    // View frustum for culling invisible objects
    vkTools::Frustum frustum;

    VulkanExample() : vkx::ExampleBase(ENABLE_VALIDATION) {
        camera.setZoom(-32.5f);
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ 0.0f, 37.5f, 0.0f });
        enableTextOverlay = true;
        title = "Vulkan Example - Multi threaded rendering";
        // Get number of max. concurrrent threads
        numThreads = std::thread::hardware_concurrency();
        assert(numThreads > 0);
#if defined(__ANDROID__)
        LOGD("numThreads = %d", numThreads);
#else
        std::cout << "numThreads = " << numThreads << std::endl;
#endif
        srand(time(NULL));

        threadPool.setThreadCount(numThreads);

        numObjectsPerThread = 256 / numThreads;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources 
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.phong);
        device.destroyPipeline(pipelines.starsphere);

        device.destroyPipelineLayout(pipelineLayout);

        device.freeCommandBuffers(cmdPool, primaryCommandBuffer);
        device.freeCommandBuffers(cmdPool, secondaryCommandBuffer);

        meshes.ufo.destroy();
        meshes.skysphere.destroy();

        for (auto& thread : threadData) {
            device.freeCommandBuffers(thread.commandPool, thread.commandBuffer.size(), thread.commandBuffer.data());
            device.destroyCommandPool(thread.commandPool);
        }

        device.destroyFence(renderFence);
    }

    float rnd(float range) {
        return range * (rand() / double(RAND_MAX));
    }

    // Create all threads and initialize shader push constants
    void prepareMultiThreadedRenderer() {
        // Since this demo updates the command buffers on each frame
        // we don't use the per-framebuffer command buffers from the
        // base class, and create a single primary command buffer instead
        vk::CommandBufferAllocateInfo cmdBufAllocateInfo =
            vkx::commandBufferAllocateInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
        primaryCommandBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

        // Create a secondary command buffer for rendering the star sphere
        cmdBufAllocateInfo.level = vk::CommandBufferLevel::eSecondary;
        secondaryCommandBuffer = device.allocateCommandBuffers(cmdBufAllocateInfo)[0];

        threadData.resize(numThreads);

        withPrimaryCommandBuffer([&](const vk::CommandBuffer& setupCmdBuffer) {
        });

        float maxX = std::floor(std::sqrt(numThreads * numObjectsPerThread));
        uint32_t posX = 0;
        uint32_t posZ = 0;

        for (uint32_t i = 0; i < numThreads; i++) {
            ThreadData *thread = &threadData[i];

            // Create one command pool for each thread
            vk::CommandPoolCreateInfo cmdPoolInfo;
            cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
            cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
            thread->commandPool = device.createCommandPool(cmdPoolInfo);

            // One secondary command buffer per object that is updated by this thread
            thread->commandBuffer.resize(numObjectsPerThread);
            // Generate secondary command buffers for each thread
            vk::CommandBufferAllocateInfo secondaryCmdBufAllocateInfo =
                vkx::commandBufferAllocateInfo(
                    thread->commandPool,
                    vk::CommandBufferLevel::eSecondary,
                    thread->commandBuffer.size());
            thread->commandBuffer = device.allocateCommandBuffers(secondaryCmdBufAllocateInfo);
            thread->mesh = meshes.ufo;
            thread->pushConstBlock.resize(numObjectsPerThread);
            thread->objectData.resize(numObjectsPerThread);

            float step = 360.0f / (float)(numThreads * numObjectsPerThread);
            for (uint32_t j = 0; j < numObjectsPerThread; j++) {
                float radius = 8.0f + rnd(8.0f) - rnd(4.0f);

                thread->objectData[j].pos.x = (posX - maxX / 2.0f) * 3.0f + rnd(1.5f) - rnd(1.5f);
                thread->objectData[j].pos.z = (posZ - maxX / 2.0f) * 3.0f + rnd(1.5f) - rnd(1.5f);

                posX += 1.0f;
                if (posX >= maxX) {
                    posX = 0.0f;
                    posZ += 1.0f;
                }

                thread->objectData[j].rotation = glm::vec3(0.0f, rnd(360.0f), 0.0f);
                thread->objectData[j].deltaT = rnd(1.0f);
                thread->objectData[j].rotationDir = (rnd(100.0f) < 50.0f) ? 1.0f : -1.0f;
                thread->objectData[j].rotationSpeed = (2.0f + rnd(4.0f)) * thread->objectData[j].rotationDir;
                thread->objectData[j].scale = 0.75f + rnd(0.5f);

                thread->pushConstBlock[j].color = glm::vec3(rnd(1.0f), rnd(1.0f), rnd(1.0f));
            }
        }
    }

    // Builds the secondary command buffer for each thread
    void threadRenderCode(uint32_t threadIndex, uint32_t cmdBufferIndex, vk::CommandBufferInheritanceInfo inheritanceInfo) {
        ThreadData *thread = &threadData[threadIndex];
        ObjectData *objectData = &thread->objectData[cmdBufferIndex];

        // Check visibility against view frustum
        objectData->visible = frustum.checkSphere(objectData->pos, objectSphereDim * 0.5f);

        if (!objectData->visible) {
            return;
        }

        vk::CommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue;
        commandBufferBeginInfo.pInheritanceInfo = &inheritanceInfo;

        vk::CommandBuffer cmdBuffer = thread->commandBuffer[cmdBufferIndex];
        cmdBuffer.begin(commandBufferBeginInfo);

        cmdBuffer.setViewport(0, vkx::viewport(size));
        cmdBuffer.setScissor(0, vkx::rect2D(size));

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phong);

        // Update
        objectData->rotation.y += 2.5f * objectData->rotationSpeed * frameTimer;
        if (objectData->rotation.y > 360.0f) {
            objectData->rotation.y -= 360.0f;
        }
        objectData->deltaT += 0.15f * frameTimer;
        if (objectData->deltaT > 1.0f)
            objectData->deltaT -= 1.0f;
        objectData->pos.y = sin(glm::radians(objectData->deltaT * 360.0f)) * 2.5f;

        objectData->model = glm::translate(glm::mat4(), objectData->pos);
        objectData->model = glm::rotate(objectData->model, -sinf(glm::radians(objectData->deltaT * 360.0f)) * 0.25f, glm::vec3(objectData->rotationDir, 0.0f, 0.0f));
        objectData->model = glm::rotate(objectData->model, glm::radians(objectData->rotation.y), glm::vec3(0.0f, objectData->rotationDir, 0.0f));
        objectData->model = glm::rotate(objectData->model, glm::radians(objectData->deltaT * 360.0f), glm::vec3(0.0f, objectData->rotationDir, 0.0f));
        objectData->model = glm::scale(objectData->model, glm::vec3(objectData->scale));

        thread->pushConstBlock[cmdBufferIndex].mvp = matrices.projection * matrices.view * objectData->model;

        // Update shader push constant block
        // Contains model view matrix
        cmdBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(ThreadPushConstantBlock), &thread->pushConstBlock[cmdBufferIndex]);

        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, thread->mesh.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(thread->mesh.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(thread->mesh.indexCount, 1, 0, 0, 0);

        cmdBuffer.end();
    }

    void updateSecondaryCommandBuffer(vk::CommandBufferInheritanceInfo inheritanceInfo) {
        // Secondary command buffer for the sky sphere
        vk::CommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue;
        commandBufferBeginInfo.pInheritanceInfo = &inheritanceInfo;
        secondaryCommandBuffer.begin(commandBufferBeginInfo);
        secondaryCommandBuffer.setViewport(0, vkx::viewport(size));
        secondaryCommandBuffer.setScissor(0, vkx::rect2D(size));
        secondaryCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.starsphere);

        glm::mat4 mvp = matrices.projection * glm::mat4_cast(camera.orientation);

        secondaryCommandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(mvp), &mvp);

        vk::DeviceSize offsets = 0;
        secondaryCommandBuffer.bindVertexBuffers(0, meshes.skysphere.vertices.buffer, offsets);
        secondaryCommandBuffer.bindIndexBuffer(meshes.skysphere.indices.buffer, 0, vk::IndexType::eUint32);
        secondaryCommandBuffer.drawIndexed(meshes.skysphere.indexCount, 1, 0, 0, 0);

        secondaryCommandBuffer.end();
    }

    // Updates the secondary command buffers using a thread pool 
    // and puts them into the primary command buffer that's 
    // lat submitted to the queue for rendering
    void updateCommandBuffers(vk::Framebuffer framebuffer) {
        vk::CommandBufferBeginInfo cmdBufInfo;

        vk::ClearValue clearValues[2];
        clearValues[0].color = vkx::clearColor({ 0.0f, 0.0f, 0.2f, 0.0f });
        clearValues[1].depthStencil = vk::ClearDepthStencilValue( 1.0f, 0 );

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent = size;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;
        renderPassBeginInfo.framebuffer = framebuffer;

        // Set target frame buffer
        primaryCommandBuffer.begin(cmdBufInfo);

        // The primary command buffer does not contain any rendering commands
        // These are stored (and retrieved) from the secondary command buffers
        primaryCommandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eSecondaryCommandBuffers);

        // Inheritance info for the secondary command buffers
        vk::CommandBufferInheritanceInfo inheritanceInfo;
        inheritanceInfo.renderPass = renderPass;
        // Secondary command buffer also use the currently active framebuffer
        inheritanceInfo.framebuffer = framebuffer;

        // Contains the list of secondary command buffers to be executed
        std::vector<vk::CommandBuffer> commandBuffers;

        // Secondary command buffer with star background sphere
        updateSecondaryCommandBuffer(inheritanceInfo);
        commandBuffers.push_back(secondaryCommandBuffer);

        // Add a job to the thread's queue for each object to be rendered
        for (uint32_t t = 0; t < numThreads; t++) {
            for (uint32_t i = 0; i < numObjectsPerThread; i++) {
                threadPool.threads[t]->addJob([=] { threadRenderCode(t, i, inheritanceInfo); });
            }
        }

        threadPool.wait();

        // Only submit if object is within the current view frustum
        for (uint32_t t = 0; t < numThreads; t++) {
            for (uint32_t i = 0; i < numObjectsPerThread; i++) {
                if (threadData[t].objectData[i].visible) {
                    commandBuffers.push_back(threadData[t].commandBuffer[i]);
                }
            }
        }

        // Execute render commands from the secondary command buffer
        primaryCommandBuffer.executeCommands(commandBuffers);
        primaryCommandBuffer.endRenderPass();
        primaryCommandBuffer.end();
    }

    void draw() override {
        prepareFrame();
        updateCommandBuffers(framebuffers[currentBuffer]);

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &primaryCommandBuffer;
        queue.submit(submitInfo, renderFence);

        // Wait for fence to signal that all command buffers are ready
        vk::Result fenceRes;
        do {
            fenceRes = device.waitForFences(renderFence, VK_TRUE, 100000000);
        }
        while (fenceRes == vk::Result::eTimeout);
        device.resetFences(renderFence);

        submitFrame();
    }

    void loadMeshes() {
        meshes.ufo = loadMesh(getAssetPath() + "models/retroufo_red_lowpoly.dae", vertexLayout, 0.12f);
        meshes.skysphere = loadMesh(getAssetPath() + "models/sphere.obj", vertexLayout, 1.0f);
        objectSphereDim = std::max(std::max(meshes.ufo.dim.x, meshes.ufo.dim.y), meshes.ufo.dim.z);
    }

    void setupVertexDescriptions() {
        // Binding description
        vertices.bindingDescriptions.resize(1);
        vertices.bindingDescriptions[0] =
            vkx::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vkx::vertexSize(vertexLayout), vk::VertexInputRate::eVertex);

        // Attribute descriptions
        // Describes memory layout and shader positions
        vertices.attributeDescriptions.resize(3);
        // Location 0 : Position
        vertices.attributeDescriptions[0] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0,  vk::Format::eR32G32B32Sfloat, 0);
        // Location 1 : Normal
        vertices.attributeDescriptions[1] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1,  vk::Format::eR32G32B32Sfloat, sizeof(float) * 3);
        // Location 3 : Color
        vertices.attributeDescriptions[2] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2,  vk::Format::eR32G32B32Sfloat, sizeof(float) * 6);

        vertices.inputState = vk::PipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
        vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
        vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
        vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
    }

    void setupPipelineLayout() {
        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo;
        // Push constants for model matrices
        vk::PushConstantRange pushConstantRange =
            vkx::pushConstantRange(vk::ShaderStageFlagBits::eVertex, sizeof(ThreadPushConstantBlock), 0);

        // Push constant ranges are part of the pipeline layout
        pPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pPipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

        pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    }

    void preparePipelines() {
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vkx::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

        vk::PipelineRasterizationStateCreateInfo rasterizationState =
            vkx::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise);

        vk::PipelineColorBlendAttachmentState blendAttachmentState =
            vkx::pipelineColorBlendAttachmentState();

        vk::PipelineColorBlendStateCreateInfo colorBlendState =
            vkx::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        vk::PipelineDepthStencilStateCreateInfo depthStencilState =
            vkx::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

        vk::PipelineViewportStateCreateInfo viewportState =
            vkx::pipelineViewportStateCreateInfo(1, 1);

        vk::PipelineMultisampleStateCreateInfo multisampleState =
            vkx::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

        std::vector<vk::DynamicState> dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState =
            vkx::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), dynamicStateEnables.size());

        // Solid rendering pipeline
        // Load shaders
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = loadShader(getAssetPath() + "shaders/multithreading/phong.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/multithreading/phong.frag.spv", vk::ShaderStageFlagBits::eFragment);

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
            vkx::pipelineCreateInfo(pipelineLayout, renderPass);

        pipelineCreateInfo.pVertexInputState = &vertices.inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = shaderStages.size();
        pipelineCreateInfo.pStages = shaderStages.data();

        pipelines.phong = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

        // Star sphere rendering pipeline
        rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        depthStencilState.depthWriteEnable = VK_FALSE;
        shaderStages[0] = loadShader(getAssetPath() + "shaders/multithreading/starsphere.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/multithreading/starsphere.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.starsphere = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];
    }

    void updateMatrices() {
        matrices.projection = camera.matrices.perspective;
        matrices.view = camera.matrices.view;
        frustum.update(matrices.projection * matrices.view);
    }

    void prepare() {
        ExampleBase::prepare();
        // Create a fence for synchronization
        renderFence = device.createFence(vk::FenceCreateInfo());
        loadMeshes();
        setupVertexDescriptions();
        setupPipelineLayout();
        preparePipelines();
        prepareMultiThreadedRenderer();
        updateMatrices();
        prepared = true;
    }

    virtual void render() {
        if (!prepared)
            return;
        draw();
    }

    virtual void viewChanged() {
        updateMatrices();
    }

    virtual void getOverlayText(vkx::TextOverlay *textOverlay) {
        textOverlay->addText("Using " + std::to_string(numThreads) + " threads", 5.0f, 85.0f, vkx::TextOverlay::alignLeft);
    }
};

RUN_EXAMPLE(VulkanExample)
