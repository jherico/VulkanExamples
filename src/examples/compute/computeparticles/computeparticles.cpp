/*
 * Vulkan Example - Attraction based compute shader particle system
 *
 * Updated compute shader by Lukas Bergdoll (https://github.com/Voultapher)
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <common/random.hpp>
#include <examples/compute.hpp>
#include <examples/example.hpp>

#include <shaders/computeparticles/particle.frag.inl>
#include <shaders/computeparticles/particle.vert.inl>
#include <shaders/computeparticles/particle.comp.inl>

#if defined(__ANDROID__)
// Lower particle count on Android for performance reasons
#define PARTICLE_COUNT 64 * 1024
#else
#define PARTICLE_COUNT 256 * 1024
#endif

struct Particle {
    glm::vec2 pos;
    glm::vec2 vel;
    glm::vec4 gradientPos;
};

class ComputeParticles : public vkx::Compute {
    using Parent = vkx::Compute;

public:
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    vk::BufferMemoryBarrier2 releaseBarrier;
    vk::BufferMemoryBarrier2 acquireBarrier;

    struct {
        vks::Buffer storage;
        vks::Buffer uniform;
    } buffers;

    struct UBO {
        float deltaT{ 0 };
        float destX{ 0 };
        float destY{ 0 };
        int32_t particleCount = PARTICLE_COUNT;
    } ubo;

    void prepare(uint32_t swapchainImageCount) override {
        Parent::prepare(swapchainImageCount);
        prepareBuffers();
        prepareDescriptors();
        preparePipeline();
        buildCommandBuffers();
    }

    void destroy() override {
        buffers.storage.destroy();
        buffers.uniform.destroy();
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(pipeline);
        device.destroy(descriptorPool);
        Parent::destroy();
    }

    void prepareBuffers() {
        auto& loader = vks::Loader::get();
        // Prepare and initialize uniform buffer containing shader uniforms
        buffers.uniform = loader.createUniformBuffer(ubo);
        vkx::Random rand;

        // Setup and fill the compute shader storage buffers for vertex positions and velocities

        // Initial particle positions
        std::vector<Particle> particleBuffer(PARTICLE_COUNT);
        for (auto& particle : particleBuffer) {
            particle.pos = rand.v2(-1.0f, 1.0f);
            particle.gradientPos.x = particle.pos.x / 2.0f;
        }

        auto bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer;

        // Staging
        // SSBO is static, copy to device local memory
        // This results in better performance
        buffers.storage = loader.stageToDeviceBuffer(computeQueue, bufferUsage, particleBuffer);

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
            buffers.storage.buffer,
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
            buffers.storage.buffer,
            0,
            VK_WHOLE_SIZE,
        };

        // Execute a release barrier to pair up with the initial acquire barrier on the graphics queue
        loader.withPrimaryCommandBuffer(computeQueue, [&](const vk::CommandBuffer& cmdBuffer) {
            cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, releaseBarrier });
        });
    }

    void prepareDescriptors() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Particle position storage buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 1 : Uniform buffer
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            // Binding 0 : Particle position storage buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffers.storage.descriptor },
            // Binding 1 : Uniform buffer
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &buffers.uniform.descriptor },
        };

        device.updateDescriptorSets(computeWriteDescriptorSets, {});
    }

    void preparePipeline() {
        // Create pipeline
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = pipelineLayout;
        computePipelineCreateInfo.stage = vks::shaders::loadShader(device, vkx::shaders::computeparticles::particle::comp, vk::ShaderStageFlagBits::eCompute);
        pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroy(computePipelineCreateInfo.stage.module);
    }

    void buildCommandBuffers() override {
        for (int i = 0; i < commandBuffers.size(); ++i) {
            auto& commandBuffer = commandBuffers[i];
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, acquireBarrier });
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);
            commandBuffer.dispatch(PARTICLE_COUNT / 16, 1, 1);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, releaseBarrier });
            commandBuffer.end();
        }
    }
};

class VulkanExample : public vkx::ExampleBase {
    using Parent = ExampleBase;

public:
    float timer = 0.0f;
    float animStart = 20.0f;
    bool animate = true;

    ComputeParticles compute;
    struct {
        vk::Pipeline pipeline;
        vk::PipelineLayout pipelineLayout;
        vk::DescriptorSet descriptorSet;
        vk::DescriptorSetLayout descriptorSetLayout;
    } graphics;
    struct {
        vks::texture::Texture2D particle;
        vks::texture::Texture2D gradient;
    } textures;

    VulkanExample() { title = "Vulkan Example - Compute shader particle system"; }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        compute.destroy();
        device.destroy(graphics.pipeline);
        graphics.pipeline = nullptr;
        device.destroy(graphics.pipelineLayout);
        graphics.pipelineLayout = nullptr;
        device.destroy(graphics.descriptorSetLayout);
        graphics.descriptorSetLayout = nullptr;
        textures.particle.destroy();
        textures.gradient.destroy();
    }

    void loadAssets() override {
        textures.particle.loadFromFile(getAssetPath() + "textures/particle01_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
        textures.gradient.loadFromFile(getAssetPath() + "textures/particle_gradient_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
    }

    void prepareDescriptors() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 2 },
        };
        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Particle color map
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 1 : Particle gradient ramp
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        graphics.descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        graphics.descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &graphics.descriptorSetLayout })[0];
        // vk::Image descriptor for the color map texture

        std::vector<vk::DescriptorImageInfo> texDescriptors{
            textures.particle.makeDescriptor(defaultSampler),
            textures.gradient.makeDescriptor(defaultSampler),
        };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Particle color map
            { graphics.descriptorSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptors[0] },
            // Binding 1 : Particle gradient ramp
            { graphics.descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptors[1] },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        graphics.pipelineLayout = device.createPipelineLayout({ {}, 1, &graphics.descriptorSetLayout });
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, graphics.pipelineLayout };
        pipelineBuilder.dynamicRendering(swapChain.surfaceFormat.format, deviceInfo.supportedDepthFormat);
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::ePointList;
        pipelineBuilder.depthStencilState = { false };
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        // Additive blending
        blendAttachmentState.colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;

        // Binding description
        pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, sizeof(Particle), vk::VertexInputRate::eVertex } };

        // Attribute descriptions
        // Describes memory layout and shader positions
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Location 0 : Position
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, pos) },
            // Location 1 : Gradient position
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, gradientPos) },
        };

        // Rendering pipeline
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::computeparticles::particle::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::computeparticles::particle::frag, vk::ShaderStageFlagBits::eFragment);
        graphics.pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    vk::BufferMemoryBarrier2 releaseBarrier, acquireBarrier;

    void prepare() override {
        ExampleBase::prepare();
        compute.prepare(swapChain.imageCount);

        releaseBarrier = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eVertexAttributeInput,
            vk::AccessFlagBits2::eVertexAttributeRead,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Src and dst queues
            context.queuesInfo.graphics.index,
            context.queuesInfo.compute.index,
            // Buffer
            compute.buffers.storage.buffer,
            0,
            VK_WHOLE_SIZE,
        };

        acquireBarrier = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eVertexAttributeInput,
            vk::AccessFlagBits2::eVertexAttributeRead,
            // Src and dst queues
            context.queuesInfo.compute.index,
            context.queuesInfo.graphics.index,
            // Buffer
            compute.buffers.storage.buffer,
            0,
            VK_WHOLE_SIZE,
        };

        prepareDescriptors();
        preparePipelines();
        buildCommandBuffers();
        prepared = true;
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, acquireBarrier });
        Parent::updateCommandBufferPreDraw(cmdBuffer);
    }

    void updateCommandBufferPostDraw(const vk::CommandBuffer& cmdBuffer) override {
        Parent::updateCommandBufferPostDraw(cmdBuffer);
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, releaseBarrier });
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        // Draw the particle system using the update vertex buffer
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, compute.buffers.storage.buffer, { 0 });
        cmdBuffer.draw(PARTICLE_COUNT, 1, 0, 0);
    }

    void updateUniformBuffers() {
        compute.ubo.deltaT = frameTimer * 2.5f;
        if (animate) {
            compute.ubo.destX = sinf(glm::radians(timer * 360.0f)) * 0.75f;
            compute.ubo.destY = 0.f;
        } else {
            float normalizedMx = (mousePos.x - static_cast<float>(size.width / 2)) / static_cast<float>(size.width / 2);
            float normalizedMy = (mousePos.y - static_cast<float>(size.height / 2)) / static_cast<float>(size.height / 2);
            compute.ubo.destX = normalizedMx;
            compute.ubo.destY = normalizedMy;
        }
        compute.buffers.uniform.copy(compute.ubo);
    }

    void postRender() override {
        vks::frame::QueuedCommandBuilder builder{ compute.commandBuffers[currentIndex], vkx::RenderStates::COMPUTE_POST,
                                                  vk::PipelineStageFlagBits2::eComputeShader };
        builder.withQueueFamilyIndex(computeQueue.familyInfo.index);
        queueCommandBuffer(builder);
    }

    void update(float deltaTime) override {
        vkx::ExampleBase::update(deltaTime);
        if (animate) {
            if (animStart > 0.0f) {
                animStart -= frameTimer * 5.0f;
            } else if (animStart <= 0.0f) {
                timer += frameTimer * 0.04f;
                if (timer > 1.f)
                    timer = 0.f;
            }
        }

        updateUniformBuffers();
    }

    void toggleAnimation() { animate = !animate; }

    void keyPressed(uint32_t key) override {
        switch (key) {
            case KEY_A:
                toggleAnimation();
                break;
        }
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            ui.checkBox("Moving attractor", &animate);
        }
    }
};

VULKAN_EXAMPLE_MAIN()
