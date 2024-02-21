/*
 * Vulkan Example - Compute shader sloth simulation
 *
 * Updated compute shader by Lukas Bergdoll (https://github.com/Voultapher)
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/compute.hpp>
#include <examples/example.hpp>

#include <shaders/computecloth/cloth.frag.inl>
#include <shaders/computecloth/cloth.vert.inl>
#include <shaders/computecloth/cloth.comp.inl>
#include <shaders/computecloth/sphere.frag.inl>
#include <shaders/computecloth/sphere.vert.inl>

constexpr uint32_t sceneSetup = 0;

struct Cloth {
    glm::uvec2 gridsize = glm::uvec2(60, 60);
    glm::vec2 size = glm::vec2(2.5f, 2.5f);
} cloth;

// SSBO cloth grid particle declaration
struct Particle {
    glm::vec4 pos;
    glm::vec4 vel;
    glm::vec4 uv;
    glm::vec4 normal;
    float pinned{ 0.0 };
    glm::vec3 _pad0;
};

// Resources for the compute part of the example
struct Compute : public vkx::Compute {
    using Parent = vkx::Compute;

    uint32_t readSet = 0;
    struct StorageBuffers {
        vks::Buffer input;
        vks::Buffer output;
    } storageBuffers;

    vks::Buffer uniformBuffer;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    std::array<vk::DescriptorSet, 2> descriptorSets;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::BufferMemoryBarrier2 acquireBarrier, releaseBarrier;

    struct UBO {
        float deltaT = 0.0f;
        float particleMass = 0.1f;
        float springStiffness = 2000.0f;
        float damping = 0.25f;
        float restDistH{ 0 };
        float restDistV{ 0 };
        float restDistD{ 0 };
        float sphereRadius = 0.5f;
        glm::vec4 spherePos = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        glm::vec4 gravity = glm::vec4(0.0f, 9.8f, 0.0f, 0.0f);
        glm::ivec2 particleCount;
    } ubo;

    void prepare(uint32_t swapchainImageCount) override {
        Parent::prepare(swapchainImageCount);
        // Create a command buffer for compute operations
        prepareDescriptors();
        preparePipeline();
        prepareBuffers();
    }

    void destroy() override {
        storageBuffers.input.destroy();
        storageBuffers.output.destroy();
        uniformBuffer.destroy();
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(pipeline);
        device.destroy(descriptorPool);
        Parent::destroy();
    }

    void prepareDescriptors() {
        // Create compute pipeline
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };

        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 2 },
            { vk::DescriptorType::eStorageBuffer, 4 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });

        descriptorSetLayout =
            context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        // Create two descriptor sets with input and output buffers switched
        descriptorSets[0] = device.allocateDescriptorSets(allocInfo)[0];
        descriptorSets[1] = device.allocateDescriptorSets(allocInfo)[0];

        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            { descriptorSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &storageBuffers.input.descriptor },
            { descriptorSets[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &storageBuffers.output.descriptor },
            { descriptorSets[0], 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },

            { descriptorSets[1], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &storageBuffers.output.descriptor },
            { descriptorSets[1], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &storageBuffers.input.descriptor },
            { descriptorSets[1], 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },
        };

        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);
    }

    void preparePipeline() {
        // Push constants used to pass some parameters
        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) };
        pipelineLayout = context.device.createPipelineLayout({ {}, 1, &descriptorSetLayout, 1, &pushConstantRange });

        // Create pipeline
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = pipelineLayout;
        computePipelineCreateInfo.stage = vks::shaders::loadShader(context.device, vkx::shaders::computecloth::cloth::comp, vk::ShaderStageFlagBits::eCompute);
        pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroy(computePipelineCreateInfo.stage.module);
    }

    void prepareBuffers() {
        const auto& loader = vks::Loader::get();

        std::vector<Particle> particleBuffer(cloth.gridsize.x * cloth.gridsize.y);
        float dx = cloth.size.x / (cloth.gridsize.x - 1);
        float dy = cloth.size.y / (cloth.gridsize.y - 1);
        float du = 1.0f / (cloth.gridsize.x - 1);
        float dv = 1.0f / (cloth.gridsize.y - 1);

        switch (sceneSetup) {
            case 0: {
                // Horz. cloth falls onto sphere
                glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3(-cloth.size.x / 2.0f, -2.0f, -cloth.size.y / 2.0f));
                for (uint32_t i = 0; i < cloth.gridsize.y; i++) {
                    for (uint32_t j = 0; j < cloth.gridsize.x; j++) {
                        particleBuffer[i + j * cloth.gridsize.y].pos = transM * glm::vec4(dx * j, 0.0f, dy * i, 1.0f);
                        particleBuffer[i + j * cloth.gridsize.y].vel = glm::vec4(0.0f);
                        particleBuffer[i + j * cloth.gridsize.y].uv = glm::vec4(1.0f - du * i, dv * j, 0.0f, 0.0f);
                    }
                }
                break;
            }
            case 1: {
                // Vert. Pinned cloth
                glm::mat4 transM = glm::translate(glm::mat4(1.0f), glm::vec3(-cloth.size.x / 2.0f, -cloth.size.y / 2.0f, 0.0f));
                for (uint32_t i = 0; i < cloth.gridsize.y; i++) {
                    for (uint32_t j = 0; j < cloth.gridsize.x; j++) {
                        particleBuffer[i + j * cloth.gridsize.y].pos = transM * glm::vec4(dx * j, dy * i, 0.0f, 1.0f);
                        particleBuffer[i + j * cloth.gridsize.y].vel = glm::vec4(0.0f);
                        particleBuffer[i + j * cloth.gridsize.y].uv = glm::vec4(du * j, dv * i, 0.0f, 0.0f);
                        // Pin some particles
                        particleBuffer[i + j * cloth.gridsize.y].pinned =
                            (i == 0) &&
                            ((j == 0) || (j == cloth.gridsize.x / 3) || (j == cloth.gridsize.x - cloth.gridsize.x / 3) || (j == cloth.gridsize.x - 1));
                        // Remove sphere
                        ubo.spherePos.z = -10.0f;
                    }
                }
                break;
            }
        }

        vk::DeviceSize storageBufferSize = particleBuffer.size() * sizeof(Particle);
        const auto& bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;
        // SSBO won't be changed on the host after upload so copy to device local memory
        storageBuffers.input = loader.stageToDeviceBuffer(computeQueue, bufferUsage, particleBuffer);
        storageBuffers.output = loader.stageToDeviceBuffer(computeQueue, bufferUsage, particleBuffer);

        releaseBarrier = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Src and dst queues
            context.queuesInfo.compute.index,
            context.queuesInfo.graphics.index,
            // Buffer
            storageBuffers.output.buffer,
            0,
            VK_WHOLE_SIZE,
        };

        acquireBarrier = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderWrite,
            // Src and dst queues
            context.queuesInfo.graphics.index,
            context.queuesInfo.compute.index,
            // Buffer
            storageBuffers.output.buffer,
            0,
            VK_WHOLE_SIZE,
        };

        // Execute a release barrier to pair up with the initial acquire barrier on the graphics queue
        loader.withPrimaryCommandBuffer(computeQueue, [&](const vk::CommandBuffer& cmdBuffer) {
            cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, releaseBarrier });
        });
    }

    void computeToComputeBarrier(const vk::CommandBuffer& cmdBuf, bool reverse) {
        std::array<vk::BufferMemoryBarrier2, 2> barriers;
        barriers[0] = vk::BufferMemoryBarrier2{ //
                                                // src stage and access
                                                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                                // dst stage and access
                                                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                                                // src and dst stage
                                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                // buffer, offset and size
                                                reverse ? storageBuffers.input.buffer : storageBuffers.output.buffer, 0, VK_WHOLE_SIZE
        };
        barriers[1] = vk::BufferMemoryBarrier2{ //
                                                // src stage and access
                                                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                                                // dst stage and access
                                                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                                                // src and dst stage
                                                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                // buffer, offset and size
                                                reverse ? storageBuffers.output.buffer : storageBuffers.input.buffer, 0, VK_WHOLE_SIZE
        };
        cmdBuf.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, barriers, nullptr });
    }

    void buildCommandBuffers() override {
        for (const auto& cmdBuf : commandBuffers) {
            cmdBuf.begin(vk::CommandBufferBeginInfo{});
            cmdBuf.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, acquireBarrier });
            cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            uint32_t calculateNormals = 0;
            cmdBuf.pushConstants<uint32_t>(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, calculateNormals);
            // Dispatch the compute job
            const uint32_t iterations = 64;
            for (uint32_t j = 0; j < iterations; j++) {
                readSet = 1 - readSet;
                cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSets[readSet], nullptr);
                if (j == iterations - 1) {
                    calculateNormals = 1;
                    cmdBuf.pushConstants<uint32_t>(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, calculateNormals);
                }
                cmdBuf.dispatch(cloth.gridsize.x / 10, cloth.gridsize.y / 10, 1);
                computeToComputeBarrier(cmdBuf, j % 2 == 1);
            }
            cmdBuf.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, releaseBarrier });
            cmdBuf.end();
        }
    }
};

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    uint32_t sceneSetup{ 0 };
    uint32_t indexCount{ 0 };
    bool simulateWind = false;

    vks::texture::Texture2D textureCloth;
    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_NORMAL,
    } };
    vks::model::Model modelSphere;

    // Resources for the graphics part of the example
    struct {
        vk::DescriptorSetLayout descriptorSetLayout;
        vk::DescriptorSet descriptorSet;
        vk::PipelineLayout pipelineLayout;
        struct Pipelines {
            vk::Pipeline cloth;
            vk::Pipeline sphere;
        } pipelines;
        vks::Buffer indices;
        vks::Buffer uniformBuffer;
        struct graphicsUBO {
            glm::mat4 projection;
            glm::mat4 view;
            glm::vec4 lightPos = glm::vec4(-1.0f, 2.0f, -1.0f, 1.0f);
        } ubo;
    } graphics;

    Compute compute;
    vk::BufferMemoryBarrier2 acquireBarrier, releaseBarrier;

    VulkanExample() {
        title = "Compute shader cloth simulation";
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-30.0f, -45.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -3.5f));
        settings.overlay = true;
        srand((unsigned int)time(NULL));
    }

    ~VulkanExample() {
        // Graphics
        graphics.indices.destroy();
        graphics.uniformBuffer.destroy();
        device.destroy(graphics.pipelines.cloth, nullptr);
        device.destroy(graphics.pipelines.sphere, nullptr);
        device.destroy(graphics.pipelineLayout, nullptr);
        device.destroy(graphics.descriptorSetLayout, nullptr);
        textureCloth.destroy();
        modelSphere.destroy();

        // Compute
        compute.destroy();
    }

    void loadAssets() override {
        textureCloth.loadFromFile(getAssetPath() + "textures/vulkan_cloth_rgba.ktx", vF::eR8G8B8A8Unorm);
        modelSphere.loadFromFile(getAssetPath() + "models/geosphere.obj", vertexLayout, compute.ubo.sphereRadius * 0.05f);
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& commandBuffer) override {
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, acquireBarrier });
        Parent::updateCommandBufferPreDraw(commandBuffer);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) override {
        commandBuffer.setViewport(0, viewport());
        commandBuffer.setScissor(0, scissor());

        // Render sphere
        if (sceneSetup == 0) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipelines.sphere);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
            commandBuffer.bindIndexBuffer(modelSphere.indices.buffer, 0, vk::IndexType::eUint32);
            commandBuffer.bindVertexBuffers(0, modelSphere.vertices.buffer, { 0 });
            commandBuffer.drawIndexed(modelSphere.indexCount, 1, 0, 0, 0);
        }

        // Render cloth
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipelines.cloth);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
        commandBuffer.bindIndexBuffer(graphics.indices.buffer, 0, vk::IndexType::eUint32);
        commandBuffer.bindVertexBuffers(0, compute.storageBuffers.output.buffer, { 0 });
        commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
    }

    void updateCommandBufferPostDraw(const vk::CommandBuffer& commandBuffer) override {
        Parent::updateCommandBufferPostDraw(commandBuffer);
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, releaseBarrier });
    }

    // Setup and fill the compute shader storage buffers containing the particles
    void prepareStorageBuffers() {
        // Indices
        std::vector<uint32_t> indices;
        for (uint32_t y = 0; y < cloth.gridsize.y - 1; y++) {
            for (uint32_t x = 0; x < cloth.gridsize.x; x++) {
                indices.push_back((y + 1) * cloth.gridsize.x + x);
                indices.push_back((y)*cloth.gridsize.x + x);
            }
            // Primitive restart (signlaed by special value 0xFFFFFFFF)
            indices.push_back(0xFFFFFFFF);
        }
        uint32_t indexBufferSize = static_cast<uint32_t>(indices.size()) * sizeof(uint32_t);
        indexCount = static_cast<uint32_t>(indices.size());
        graphics.indices = loader.stageToDeviceBuffer(graphicsQueue, vBU::eIndexBuffer, indices);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 3 },
            { vk::DescriptorType::eCombinedImageSampler, 2 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupLayoutsAndDescriptors() {
        // Set layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        graphics.descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        graphics.pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &graphics.descriptorSetLayout });
        graphics.descriptorSet = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &graphics.descriptorSetLayout })[0];
        auto clothDescriptor = textureCloth.makeDescriptor(defaultSampler);
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { graphics.descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &graphics.uniformBuffer.descriptor },
            { graphics.descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &clothDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, graphics.pipelineLayout };
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleStrip;
        pipelineBuilder.inputAssemblyState.primitiveRestartEnable = VK_TRUE;
        // pipelineBuilder.inputAssemblyState.primitiveRestartEnable = VK_TRUE;
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        // Binding description
        pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, sizeof(Particle), vk::VertexInputRate::eVertex } };

        pipelineBuilder.vertexInputState.attributeDescriptions = {
            { 0, 0, vF::eR32G32B32Sfloat, offsetof(Particle, pos) },
            { 1, 0, vF::eR32G32Sfloat, offsetof(Particle, uv) },
            { 2, 0, vF::eR32G32B32Sfloat, offsetof(Particle, normal) },
        };

        pipelineBuilder.loadShader(vkx::shaders::computecloth::cloth::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::computecloth::cloth::frag, vk::ShaderStageFlagBits::eFragment);
        graphics.pipelines.cloth = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Sphere rendering pipeline
        pipelineBuilder.loadShader(vkx::shaders::computecloth::sphere::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::computecloth::sphere::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.vertexInputState = {};
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
        pipelineBuilder.inputAssemblyState.primitiveRestartEnable = VK_FALSE;
        graphics.pipelines.sphere = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Compute shader uniform buffer block
        compute.uniformBuffer = loader.createUniformBuffer(compute.ubo);

        // Initial values
        float dx = cloth.size.x / (cloth.gridsize.x - 1);
        float dy = cloth.size.y / (cloth.gridsize.y - 1);

        compute.ubo.restDistH = dx;
        compute.ubo.restDistV = dy;
        compute.ubo.restDistD = sqrtf(dx * dx + dy * dy);
        compute.ubo.particleCount = cloth.gridsize;

        updateComputeUBO();

        // Vertex shader uniform buffer block
        graphics.uniformBuffer = loader.createUniformBuffer(graphics.ubo);
        updateGraphicsUBO();
    }

    void updateComputeUBO() {
        if (!paused) {
            compute.ubo.deltaT = 0.000005f;
            // todo: base on frametime
            // compute.ubo.deltaT = frameTimer * 0.0075f;

            std::mt19937 rg((unsigned)time(nullptr));
            std::uniform_real_distribution<float> rd(1.0f, 6.0f);

            if (simulateWind) {
                compute.ubo.gravity.x = cos(glm::radians(-timer * 360.0f)) * (rd(rg) - rd(rg));
                compute.ubo.gravity.z = sin(glm::radians(timer * 360.0f)) * (rd(rg) - rd(rg));
            } else {
                compute.ubo.gravity.x = 0.0f;
                compute.ubo.gravity.z = 0.0f;
            }
        } else {
            compute.ubo.deltaT = 0.0f;
        }
        compute.uniformBuffer.copy(compute.ubo);
    }

    void updateGraphicsUBO() {
        graphics.ubo.projection = camera.matrices.perspective;
        graphics.ubo.view = camera.matrices.view;
        graphics.uniformBuffer.copy(graphics.ubo);
    }

    void postRender() override {
        queueCommandBuffer(compute.commandBuffers[currentIndex], vkx::RenderStates::COMPUTE_POST, vk::PipelineStageFlagBits2::eComputeShader);
    }

    void prepare() override {
        ExampleBase::prepare();
        compute.prepare(swapChain.imageCount);
        prepareStorageBuffers();

        prepareUniformBuffers();
        setupDescriptorPool();
        setupLayoutsAndDescriptors();
        preparePipelines();
        updateComputeUBO();
        buildCommandBuffers();
        compute.buildCommandBuffers();
        prepared = true;
    }

    void update(float deltaTime) override {
        Parent::update(deltaTime);
        updateComputeUBO();
    }

    void viewChanged() override { updateGraphicsUBO(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            ui.checkBox("Simulate wind", &simulateWind);
            // bool pinned = sceneSetup == 0 ? false : true;
            // ui.checkBox("Pinned", &pinned);
            // int oldSceneSetup = sceneSetup;
            // if (pinned) {
            //     sceneSetup = 1;
            // } else {
            //     sceneSetup = 0;
            // }
            // if (oldSceneSetup != sceneSetup) {
            //     waitIdle();
            //     prepareStorageBuffers();
            //     buildCommandBuffers();
            // }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
