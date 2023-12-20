/*
* Vulkan Example - Compute shader N-body simulation using two passes and shared compute shader memory
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

#if defined(__ANDROID__)
// Lower particle count on Android for performance reasons
#define PARTICLES_PER_ATTRACTOR 3 * 1024
#else
#define PARTICLES_PER_ATTRACTOR 4 * 1024
#endif

class ComputeNBody : public vkx::Compute {
    using Parent = vkx::Compute;

public:
    ComputeNBody(const vks::Context& context)
        : Parent(context) {}
    // SSBO particle declaration
    struct Particle {
        glm::vec4 pos;  // xyz = position, w = mass
        glm::vec4 vel;  // xyz = velocity, w = gradient texture position
    };
    uint32_t numParticles;
    vks::Buffer storageBuffer;        // (Shader) storage buffer object containing the particles
    vks::Buffer uniformBuffer;        // Uniform buffer object containing particle system parameters
    vk::CommandBuffer commandBuffer;  // Command buffer storing the dispatch commands and barriers
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;  // Compute shader binding layout
    vk::DescriptorSet descriptorSet;              // Compute shader bindings
    vk::PipelineLayout pipelineLayout;            // Layout of the compute pipeline
    vk::Pipeline pipelineCalculate;               // Compute pipeline for N-Body velocity calculation (1st pass)
    vk::Pipeline pipelineIntegrate;               // Compute pipeline for euler integration (2nd pass)
    vk::Pipeline blur;
    vk::PipelineLayout pipelineLayoutBlur;
    vk::DescriptorSetLayout descriptorSetLayoutBlur;
    vk::DescriptorSet descriptorSetBlur;
    struct computeUBO {     // Compute shader uniform block object
        float deltaT{ 0 };  //		Frame delta time
        float destX{ 0 };   //		x position of the attractor
        float destY{ 0 };   //		y position of the attractor
        int32_t particleCount;
    } ubo;

    void prepare() {
        Parent::prepare();

        // Create compute pipeline
        // Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)
        prepareStorageBuffers();
        prepareDescriptors();
        preparePipelines();
    }

    void destroy() {
        storageBuffer.destroy();
        uniformBuffer.destroy();
        device.destroy(pipelineCalculate);
        device.destroy(pipelineIntegrate);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(descriptorPool);
        Parent::destroy();
    }

    void prepareDescriptors() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 2 },
            { vk::DescriptorType::eStorageBuffer, 1 },
            { vk::DescriptorType::eCombinedImageSampler, 2 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Particle position storage buffer
            { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 1 : Uniform buffer
            { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };

        descriptorSetLayout =
            device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            // Binding 0 : Particle position storage buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &storageBuffer.descriptor },
            // Binding 1 : Uniform buffer
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },
        };

        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Create pipelines
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = pipelineLayout;
        // 1st pass
        computePipelineCreateInfo.stage =
            vks::shaders::loadShader(device, vkx::getAssetPath() + "shaders/computenbody/particle_calculate.comp.spv", vk::ShaderStageFlagBits::eCompute);
        // Set shader parameters via specialization constants
        struct SpecializationData {
            uint32_t sharedDataSize;
            float gravity;
            float power;
            float soften;
        } specializationData;

        std::vector<vk::SpecializationMapEntry> specializationMapEntries{
            { 0, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t) },
            { 1, offsetof(SpecializationData, gravity), sizeof(float) },
            { 2, offsetof(SpecializationData, power), sizeof(float) },
            { 3, offsetof(SpecializationData, soften), sizeof(float) },
        };

        specializationData.sharedDataSize =
            std::min((uint32_t)1024, (uint32_t)(context.deviceProperties.limits.maxComputeSharedMemorySize / sizeof(glm::vec4)));

        specializationData.gravity = 0.002f;
        specializationData.power = 0.75f;
        specializationData.soften = 0.05f;

        vk::SpecializationInfo specializationInfo{ static_cast<uint32_t>(specializationMapEntries.size()), specializationMapEntries.data(),
                                                   sizeof(specializationData), &specializationData };
        computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;
        pipelineCalculate = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroyShaderModule(computePipelineCreateInfo.stage.module);
        // 2nd pass
        computePipelineCreateInfo.stage =
            vks::shaders::loadShader(device, vkx::getAssetPath() + "shaders/computenbody/particle_integrate.comp.spv", vk::ShaderStageFlagBits::eCompute);
        pipelineIntegrate = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroyShaderModule(computePipelineCreateInfo.stage.module);

        // Create a command buffer for compute operations
        commandBuffer = device.allocateCommandBuffers({ commandPool, vk::CommandBufferLevel::ePrimary, 1 })[0];

        // Build a single command buffer containing the compute dispatch commands
        buildComputeCommandBuffer();
    }

    // Setup and fill the compute shader storage buffers containing the particles
    void prepareStorageBuffers() {
#if 0
        std::vector<glm::vec3> attractors = {
            glm::vec3(2.5f, 1.5f, 0.0f),
            glm::vec3(-2.5f, -1.5f, 0.0f),
        };
#else
        std::vector<glm::vec3> attractors = {
            glm::vec3(5.0f, 0.0f, 0.0f),  glm::vec3(-5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 5.0f),
            glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(0.0f, 4.0f, 0.0f),  glm::vec3(0.0f, -8.0f, 0.0f),
        };
#endif

        numParticles = static_cast<uint32_t>(attractors.size()) * PARTICLES_PER_ATTRACTOR;
        ubo.particleCount = numParticles;
        // Compute shader uniform buffer block
        uniformBuffer = context.createUniformBuffer(ubo);

        // Initial particle positions
        std::vector<Particle> particleBuffer(numParticles);

        std::mt19937 rndGen(static_cast<uint32_t>(time(0)));
        std::normal_distribution<float> rndDist(0.0f, 1.0f);

        for (uint32_t i = 0; i < static_cast<uint32_t>(attractors.size()); i++) {
            for (uint32_t j = 0; j < PARTICLES_PER_ATTRACTOR; j++) {
                Particle& particle = particleBuffer[i * PARTICLES_PER_ATTRACTOR + j];

                // First particle in group as heavy center of gravity
                if (j == 0) {
                    particle.pos = glm::vec4(attractors[i] * 1.5f, 90000.0f);
                    particle.vel = glm::vec4(glm::vec4(0.0f));
                } else {
                    // Position
                    glm::vec3 position(attractors[i] + glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen)) * 0.75f);
                    float len = glm::length(glm::normalize(position - attractors[i]));
                    position.y *= 2.0f - (len * len);

                    // Velocity
                    glm::vec3 angular = glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
                    glm::vec3 velocity =
                        glm::cross((position - attractors[i]), angular) + glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen) * 0.025f);

                    float mass = (rndDist(rndGen) * 0.5f + 0.5f) * 75.0f;
                    particle.pos = glm::vec4(position, mass);
                    particle.vel = glm::vec4(velocity, 0.0f);
                }

                // Color gradient offset
                particle.vel.w = (float)i * 1.0f / static_cast<uint32_t>(attractors.size());
            }
        }

        storageBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, particleBuffer);
    }

    void buildComputeCommandBuffer() {
        // Compute particle movement
        commandBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        // First pass: Calculate particle movement
        // -------------------------------------------------------------------------------------------------------
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineCalculate);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);
        commandBuffer.dispatch(numParticles / 256, 1, 1);

        // Add memory barrier to ensure that compute shader has finished writing to the buffer
        vk::BufferMemoryBarrier bufferBarrier{ vk::AccessFlagBits::eShaderWrite,
                                               vk::AccessFlagBits::eShaderRead,
                                               VK_QUEUE_FAMILY_IGNORED,
                                               VK_QUEUE_FAMILY_IGNORED,
                                               storageBuffer.buffer,
                                               0,
                                               VK_WHOLE_SIZE };

        // Second pass: Integrate particles
        // -------------------------------------------------------------------------------------------------------
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, bufferBarrier,
                                      nullptr);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineIntegrate);
        commandBuffer.dispatch(numParticles / 256, 1, 1);
        commandBuffer.end();
    }

    void submit() {
        // Submit compute commands
        Parent::submit(commandBuffer);
    }
};

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::texture::Texture2D particle;
        vks::texture::Texture2D gradient;
    } textures;

    // Resources for the graphics part of the example
    struct {
        vks::Buffer uniformBuffer;                    // Contains scene matrices
        vk::DescriptorSetLayout descriptorSetLayout;  // Particle system rendering shader binding layout
        vk::DescriptorSet descriptorSet;              // Particle system rendering shader bindings
        vk::PipelineLayout pipelineLayout;            // Layout of the graphics pipeline
        vk::Pipeline pipeline;                        // Particle rendering pipeline
        struct {
            glm::mat4 projection;
            glm::mat4 view;
            glm::vec2 screenDim;
        } ubo;
    } graphics;

    // Resources for the compute part of the example
    ComputeNBody compute{ context };

    VulkanExample() {
        title = "Compute shader N-body system";
        settings.overlay = true;
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-26.0f, 75.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -14.0f));
        camera.movementSpeed = 2.5f;
    }

    ~VulkanExample() {
        // Compute
        compute.destroy();

        // Graphics
        graphics.uniformBuffer.destroy();
        device.destroyPipeline(graphics.pipeline);
        device.destroyPipelineLayout(graphics.pipelineLayout);
        device.destroyDescriptorSetLayout(graphics.descriptorSetLayout);

        textures.particle.destroy();
        textures.gradient.destroy();
    }

    void loadAssets() override {
        textures.particle.loadFromFile(context, getAssetPath() + "textures/particle01_rgba.ktx", vF::eR8G8B8A8Unorm);
        textures.gradient.loadFromFile(context, getAssetPath() + "textures/particle_gradient_rgba.ktx", vF::eR8G8B8A8Unorm);
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {}

    void updateCommandBufferPostDraw(const vk::CommandBuffer& cmdBuffer) override {}

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, viewport());
        cmdBuffer.setScissor(0, scissor());
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, compute.storageBuffer.buffer, { 0 });
        cmdBuffer.draw(compute.numParticles, 1, 0, 0);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 2 },
            { vk::DescriptorType::eStorageBuffer, 1 },
            { vk::DescriptorType::eCombinedImageSampler, 2 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        graphics.descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        graphics.pipelineLayout = device.createPipelineLayout({ {}, 1, &graphics.descriptorSetLayout });
    }

    void setupDescriptorSet() {
        graphics.descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &graphics.descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { graphics.descriptorSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.particle.descriptor },
            { graphics.descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.gradient.descriptor },
            { graphics.descriptorSet, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &graphics.uniformBuffer.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, graphics.pipelineLayout, renderPass };
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::ePointList;
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        // Additive blending
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;
        pipelineBuilder.depthStencilState = { false };
        pipelineBuilder.vertexInputState.bindingDescriptions = {
            { 0, sizeof(ComputeNBody::Particle), vk::VertexInputRate::eVertex },
        };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Location 0 : Position
            { 0, 0, vF::eR32G32B32A32Sfloat, offsetof(ComputeNBody::Particle, pos) },
            // Location 1 : Velocity (used for gradient lookup)
            { 1, 0, vF::eR32G32B32A32Sfloat, offsetof(ComputeNBody::Particle, vel) },
        };
        // Load shaders
        pipelineBuilder.loadShader(getAssetPath() + "shaders/computenbody/particle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/computenbody/particle.frag.spv", vk::ShaderStageFlagBits::eFragment);
        graphics.pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        graphics.uniformBuffer = context.createUniformBuffer(graphics.ubo);

        updateGraphicsUniformBuffers();
    }

    void updateUniformBuffers() {
        compute.ubo.deltaT = paused ? 0.0f : frameTimer * 0.05f;
        compute.ubo.destX = sin(glm::radians(timer * 360.0f)) * 0.75f;
        compute.ubo.destY = 0.0f;
        memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));
    }

    void updateGraphicsUniformBuffers() {
        graphics.ubo.projection = camera.matrices.perspective;
        graphics.ubo.view = camera.matrices.view;
        graphics.ubo.screenDim = glm::vec2((float)size.width, (float)size.height);
        memcpy(graphics.uniformBuffer.mapped, &graphics.ubo, sizeof(graphics.ubo));
    }

    void draw() override {
        // Submit graphics commands
        ExampleBase::draw();

        static std::once_flag once;
        std::call_once(once, [&] { addRenderWaitSemaphore(compute.semaphores.complete, vk::PipelineStageFlagBits::eComputeShader); });
        static const std::vector<vk::PipelineStageFlags> waitStages{ vk::PipelineStageFlagBits::eComputeShader };
        compute.submit();
    }

    void prepare() override {
        ExampleBase::prepare();
        compute.prepare();
        renderSignalSemaphores.push_back(compute.semaphores.ready);

        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void update(float deltaTime) override {
        vkx::ExampleBase::update(deltaTime);
        updateUniformBuffers();
    }

    void viewChanged() override { updateGraphicsUniformBuffers(); }
};

VULKAN_EXAMPLE_MAIN()
