/*
* Vulkan Example - Attraction based compute shader particle system
*
* Updated compute shader by Lukas Bergdoll (https://github.com/Voultapher)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

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
    ComputeParticles(const vks::Context& context)
        : Parent(context) {}
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::CommandBuffer commandBuffer;

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

    void prepare() {
        Parent::prepare();
        prepareBuffers();
        prepareDescriptors();
        preparePipeline();

        // Create a command buffer for compute operations
        commandBuffer = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{ commandPool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        updateCommandBuffer(commandBuffer);
        // Create compute pipeline
        // Compute pipelines are created separate from graphics pipelines
        // even if they use the same queue
    }

    void destroy() {
        buffers.storage.destroy();
        buffers.uniform.destroy();
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(pipeline);
        device.destroy(descriptorPool);
        Parent::destroy();
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
        computePipelineCreateInfo.stage =
            vks::shaders::loadShader(device, vkx::getAssetPath() + "shaders/computeparticles/particle.comp.spv", vk::ShaderStageFlagBits::eCompute);

        pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroyShaderModule(computePipelineCreateInfo.stage.module);
    }

    void updateCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        // Compute particle movement
        // Add memory barrier to ensure that the (rendering) vertex shader operations have finished
        // Required as the compute shader will overwrite the vertex buffer data
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);
        // Dispatch the compute job
        cmdBuffer.dispatch(PARTICLE_COUNT / 16, 1, 1);
        cmdBuffer.end();
    }

    void prepareBuffers() {
        // Prepare and initialize uniform buffer containing shader uniforms
        buffers.uniform = context.createUniformBuffer(ubo);
        std::mt19937 rGenerator;
        std::uniform_real_distribution<float> rDistribution(-1.0f, 1.0f);

        // Setup and fill the compute shader storage buffers for vertex positions and velocities

        // Initial particle positions
        std::vector<Particle> particleBuffer(PARTICLE_COUNT);
        for (auto& particle : particleBuffer) {
            particle.pos = glm::vec2(rDistribution(rGenerator), rDistribution(rGenerator));
            particle.vel = glm::vec2(0.0f);
            particle.gradientPos.x = particle.pos.x / 2.0f;
        }

        uint32_t storageBufferSize = (uint32_t)(particleBuffer.size() * sizeof(Particle));

        // Staging
        // SSBO is static, copy to device local memory
        // This results in better performance
        buffers.storage = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, particleBuffer);
    }

    void submit() { Parent::submit(commandBuffer); }
};

class VulkanExample : public vkx::ExampleBase {
public:
    float timer = 0.0f;
    float animStart = 20.0f;
    bool animate = true;

    ComputeParticles compute{ context };
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

        device.destroyPipeline(graphics.pipeline);

        device.destroyPipelineLayout(graphics.pipelineLayout);
        device.destroyDescriptorSetLayout(graphics.descriptorSetLayout);

        textures.particle.destroy();
        textures.gradient.destroy();
    }

    void loadAssets() override {
        textures.particle.loadFromFile(context, getAssetPath() + "textures/particle01_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
        textures.gradient.loadFromFile(context, getAssetPath() + "textures/particle_gradient_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
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

        memcpy(compute.buffers.uniform.mapped, &compute.ubo, sizeof(compute.ubo));
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
            { textures.particle.sampler, textures.particle.view, vk::ImageLayout::eGeneral },
            { textures.gradient.sampler, textures.gradient.view, vk::ImageLayout::eGeneral },
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
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, graphics.pipelineLayout, renderPass };
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::ePointList;
        pipelineBuilder.depthStencilState = { false };
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        // Additive blending
        blendAttachmentState.colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
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
        pipelineBuilder.loadShader(getAssetPath() + "shaders/computeparticles/particle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/computeparticles/particle.frag.spv", vk::ShaderStageFlagBits::eFragment);
        graphics.pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    void prepare() override {
        ExampleBase::prepare();
        compute.prepare();
        prepareDescriptors();
        preparePipelines();
        buildCommandBuffers();
        renderSignalSemaphores.push_back(compute.semaphores.ready);
        prepared = true;
    }

    void draw() override {
        // Submit graphics commands
        ExampleBase::draw();

        static std::once_flag once;
        std::call_once(once, [&] { addRenderWaitSemaphore(compute.semaphores.complete, vk::PipelineStageFlagBits::eComputeShader); });

        compute.submit();
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
