/*
 * Vulkan Example - Screen space ambient occlusion example
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <common/random.hpp>
#include <examples/offscreen.hpp>

#include <shaders/ssao/blur.frag.inl>
#include <shaders/ssao/composition.frag.inl>
#include <shaders/ssao/fullscreen.vert.inl>
#include <shaders/ssao/gbuffer.frag.inl>
#include <shaders/ssao/gbuffer.vert.inl>
#include <shaders/ssao/ssao.frag.inl>

#define SSAO_KERNEL_SIZE 32
#define SSAO_RADIUS 0.5f

#if defined(__ANDROID__)
#define SSAO_NOISE_DIM 8
#else
#define SSAO_NOISE_DIM 4
#endif
#define SSAO_NOISE_COUNT (SSAO_NOISE_DIM * SSAO_NOISE_DIM)

// Vertex layout for the models
static const vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_UV,
    vks::model::VERTEX_COMPONENT_COLOR,
    vks::model::VERTEX_COMPONENT_NORMAL,
} };

#if defined(__ANDROID__)
constexpr uint32_t ssaoSizeDivisor = 2;
#else
constexpr uint32_t ssaoSizeDivisor = 1;
#endif

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::texture::Texture2D ssaoNoise;
    } textures;

    struct {
        vks::model::Model scene;
    } models;

    struct UBOSceneMatrices {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
    } uboSceneMatrices;

    struct UBOSSAOParams {
        glm::mat4 projection;
        int32_t ssao = true;
        int32_t ssaoOnly = false;
        int32_t ssaoBlur = true;
    } uboSSAOParams;

    struct {
        vk::Pipeline offscreen;
        vk::Pipeline composition;
        vk::Pipeline ssao;
        vk::Pipeline ssaoBlur;
    } pipelines;

    struct {
        vk::PipelineLayout gBuffer;
        vk::PipelineLayout ssao;
        vk::PipelineLayout ssaoBlur;
        vk::PipelineLayout composition;
    } pipelineLayouts;

    struct {
        const uint32_t count = 5;
        vk::DescriptorSet model;
        vk::DescriptorSet floor;
        vk::DescriptorSet ssao;
        vk::DescriptorSet ssaoBlur;
        vk::DescriptorSet composition;
    } descriptorSets;

    struct {
        vk::DescriptorSetLayout gBuffer;
        vk::DescriptorSetLayout ssao;
        vk::DescriptorSetLayout ssaoBlur;
        vk::DescriptorSetLayout composition;
    } descriptorSetLayouts;

    struct {
        vks::Buffer sceneMatrices;
        vks::Buffer ssaoKernel;
        vks::Buffer ssaoParams;
    } uniformBuffers;

    struct {
        vkx::offscreen::Renderer gbuffer;
        vkx::offscreen::Renderer ssao;
        vkx::offscreen::Renderer ssaoBlur;
        vk::CommandBuffer cmdBuffer;

        void destroy() {
            const auto& device = vks::Context::get().device;
            gbuffer.destroy();
            ssao.destroy();
            ssaoBlur.destroy();
            cmdBuffer.reset();
        }
    } offscreen;

    VulkanExample() {
        title = "Screen space ambient occlusion";
        settings.overlay = true;
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
        camera.rotationSpeed = 0.25f;
#endif
        camera.position = { 7.5f, -6.75f, 0.0f };
        camera.setRotation(glm::vec3(5.0f, 90.0f, 0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 64.0f);
    }

    ~VulkanExample() {
        offscreen.destroy();
        device.destroy(pipelines.offscreen);
        device.destroy(pipelines.composition);
        device.destroy(pipelines.ssao);
        device.destroy(pipelines.ssaoBlur);

        device.destroy(pipelineLayouts.gBuffer);
        device.destroy(pipelineLayouts.ssao);
        device.destroy(pipelineLayouts.ssaoBlur);
        device.destroy(pipelineLayouts.composition);

        device.destroy(descriptorSetLayouts.gBuffer);
        device.destroy(descriptorSetLayouts.ssao);
        device.destroy(descriptorSetLayouts.ssaoBlur);
        device.destroy(descriptorSetLayouts.composition);

        // Meshes
        models.scene.destroy();

        // Uniform buffers
        uniformBuffers.sceneMatrices.destroy();
        uniformBuffers.ssaoKernel.destroy();
        uniformBuffers.ssaoParams.destroy();

        // Misc
        textures.ssaoNoise.destroy();
    }

    void prepareOffscreen() {
        vk::SamplerCreateInfo sampler;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        sampler.maxAnisotropy = 1.0f;
        sampler.maxLod = 1.0f;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;

        // G-Buffer
        {
            vkx::offscreen::Builder builder{ size };
            // position
            builder.appendColorFormat(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eSampled);
            // normals
            builder.appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eSampled);
            // albedo
            builder.appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eSampled);
            builder.withSamplerCreateInfo(sampler);
            builder.withDepthFormat(defaultDepthStencilFormat);
            offscreen.gbuffer.prepare(builder);
        }

        // SSAO
        {
            const vk::Extent2D ssaoSize{ size.width / ssaoSizeDivisor, size.height / ssaoSizeDivisor };
            vkx::offscreen::Builder builder{ ssaoSize };
            builder.appendColorFormat(vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eSampled, defaultClearDepth);
            builder.withSamplerCreateInfo(sampler);
            offscreen.ssao.prepare(builder);
        }

        // SSAO blur
        {
            vkx::offscreen::Builder builder{ size };
            builder.appendColorFormat(vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eSampled, defaultClearDepth);
            builder.withSamplerCreateInfo(sampler);
            offscreen.ssaoBlur.prepare(builder);
        }
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildDeferredCommandBuffer() {
        vk::DeviceSize offsets = { 0 };
        if (!offscreen.cmdBuffer) {
            offscreen.cmdBuffer = graphicsQueue.createCommandBuffer();
        }

        // Create a semaphore used to synchronize offscreen rendering and usage

        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        using namespace vks::util;

        // First pass: Fill G-Buffer components (positions+depth, normals, albedo) using MRT
        // -------------------------------------------------------------------------------------------------------
        offscreen.gbuffer.setLayout(offscreen.cmdBuffer, ImageTransitionState::COLOR_ATTACHMENT, ImageTransitionState::DEPTH_ATTACHMENT);
        offscreen.cmdBuffer.beginRendering(offscreen.gbuffer.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.gbuffer.size));
        offscreen.cmdBuffer.setScissor(0, rect2D(offscreen.gbuffer.size));
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.gBuffer, 0, descriptorSets.floor, {});
        offscreen.cmdBuffer.bindVertexBuffers(0, models.scene.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(models.scene.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRendering();

        // Second pass: SSAO generation
        // -------------------------------------------------------------------------------------------------------
        offscreen.gbuffer.setLayout(offscreen.cmdBuffer, ImageTransitionState::SAMPLED, ImageTransitionState::SAMPLED);

        offscreen.ssao.setLayout(offscreen.cmdBuffer, ImageTransitionState::COLOR_ATTACHMENT);
        offscreen.cmdBuffer.beginRendering(offscreen.ssao.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.ssao.size));
        offscreen.cmdBuffer.setScissor(0, rect2D(offscreen.ssao.size));
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.ssao, 0, descriptorSets.ssao, {});
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ssao);
        offscreen.cmdBuffer.draw(3, 1, 0, 0);
        offscreen.cmdBuffer.endRendering();
        offscreen.ssao.setLayout(offscreen.cmdBuffer, ImageTransitionState::SAMPLED);

        // Third pass: SSAO blur
        // -------------------------------------------------------------------------------------------------------
        offscreen.ssaoBlur.setLayout(offscreen.cmdBuffer, ImageTransitionState::COLOR_ATTACHMENT);
        offscreen.cmdBuffer.beginRendering(offscreen.ssaoBlur.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.ssaoBlur.size));
        offscreen.cmdBuffer.setScissor(0, rect2D(offscreen.ssaoBlur.size));
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.ssaoBlur, 0, descriptorSets.ssaoBlur, {});
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ssaoBlur);
        offscreen.cmdBuffer.draw(3, 1, 0, 0);
        offscreen.cmdBuffer.endRendering();
        offscreen.ssaoBlur.setLayout(offscreen.cmdBuffer, ImageTransitionState::SAMPLED);

        offscreen.cmdBuffer.end();
    }

    void loadAssets() override {
        vks::model::ModelCreateInfo modelCreateInfo;
        modelCreateInfo.scale = glm::vec3(0.5f);
        modelCreateInfo.uvscale = glm::vec2(1.0f);
        modelCreateInfo.center = glm::vec3(0.0f, 0.0f, 0.0f);
        models.scene.loadFromFile(getAssetPath() + "models/sibenik/sibenik.dae", vertexLayout, modelCreateInfo);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCommandBuffer) override {
        vk::Viewport viewport;
        viewport.width = (float)size.width;
        viewport.height = (float)size.height;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;
        drawCommandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor;
        scissor.extent = size;
        drawCommandBuffer.setScissor(0, scissor);
        // Final composition pass
        drawCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, {});
        drawCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
        drawCommandBuffer.draw(3, 1, 0, 0);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 10 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 12 },
        };
        descriptorPool =
            device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, descriptorSets.count, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupLayoutsAndDescriptors() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
        vk::DescriptorSetLayoutCreateInfo setLayoutCreateInfo;
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
        vk::DescriptorSetAllocateInfo descriptorAllocInfo{ descriptorPool, 1 };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

        // G-Buffer creation (offscreen scene rendering)
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.gBuffer = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.gBuffer = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.gBuffer });
        descriptorSets.floor = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayouts.gBuffer })[0];
        writeDescriptorSets = { { descriptorSets.floor, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.sceneMatrices.descriptor } };
        device.updateDescriptorSets(writeDescriptorSets, {});
        // SSAO Generation
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Position+Depth
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Normals
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS SSAO Noise
            { 3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },         // FS SSAO Kernel UBO
            { 4, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },         // FS Params UBO
        };

        descriptorSetLayouts.ssao = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.ssao = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.ssao });
        descriptorSets.ssao = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayouts.ssao })[0];

        std::vector<vk::DescriptorImageInfo> imageDescriptors{
            { offscreen.gbuffer.sampler, offscreen.gbuffer.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.gbuffer.sampler, offscreen.gbuffer.colorTargets[1].imageView, vk::ImageLayout::eReadOnlyOptimal },
            textures.ssaoNoise.makeDescriptor(defaultSampler),
        };
        writeDescriptorSets = {
            { descriptorSets.ssao, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[0] },                     // FS Position+Depth
            { descriptorSets.ssao, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[1] },                     // FS Normals
            { descriptorSets.ssao, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[2] },                     // FS SSAO Noise
            { descriptorSets.ssao, 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.ssaoKernel.descriptor },  // FS SSAO Kernel UBO
            { descriptorSets.ssao, 4, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.ssaoParams.descriptor },  // FS SSAO Params UBO
        };
        device.updateDescriptorSets(writeDescriptorSets, {});

        // SSAO Blur
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Sampler SSAO
        };
        descriptorSetLayouts.ssaoBlur = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.ssaoBlur = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.ssaoBlur });
        descriptorSets.ssaoBlur = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.ssaoBlur })[0];

        imageDescriptors = {
            { offscreen.ssao.sampler, offscreen.ssao.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal },
        };
        writeDescriptorSets = {
            { descriptorSets.ssaoBlur, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[0] },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});

        // Composition
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Position+Depth
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Normals
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Albedo
            { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS SSAO
            { 4, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS SSAO blurred
            { 5, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },         // FS Lights UBO
        };

        descriptorSetLayouts.composition = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.composition = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.composition });
        descriptorSets.composition = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.composition })[0];

        imageDescriptors = {
            { offscreen.gbuffer.sampler, offscreen.gbuffer.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.gbuffer.sampler, offscreen.gbuffer.colorTargets[1].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.gbuffer.sampler, offscreen.gbuffer.colorTargets[2].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.gbuffer.sampler, offscreen.ssao.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.gbuffer.sampler, offscreen.ssaoBlur.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal },
        };

        writeDescriptorSets = {
            { descriptorSets.composition, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[0] },  // FS Sampler Position+Depth
            { descriptorSets.composition, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[1] },  // FS Sampler Normals
            { descriptorSets.composition, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[2] },  // FS Sampler Albedo
            { descriptorSets.composition, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[3] },  // FS Sampler SSAO
            { descriptorSets.composition, 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[4] },  // FS Sampler SSAO blurred
            { descriptorSets.composition, 5, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.ssaoParams.descriptor },  // FS SSAO Params UBO
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayouts.composition };
        builder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // Final composition pass pipeline
        {
            builder.loadShader(vkx::shaders::ssao::fullscreen::vert, vk::ShaderStageFlagBits::eVertex);
            builder.loadShader(vkx::shaders::ssao::composition::frag, vk::ShaderStageFlagBits::eFragment);
            pipelines.composition = builder.create(context.pipelineCache);
        }

        // SSAO Pass
        {
            offscreen.ssao.setupDynamicRendering(builder);
            builder.layout = pipelineLayouts.ssao;
            // Destroy the fragment shader, but not the vertex shader
            device.destroy(builder.shaderStages[1].module);
            builder.shaderStages.resize(1);
            builder.loadShader(vkx::shaders::ssao::ssao::frag, vk::ShaderStageFlagBits::eFragment);

            struct SpecializationData {
                uint32_t kernelSize = SSAO_KERNEL_SIZE;
                float radius = SSAO_RADIUS;
            } specializationData;

            // Set constant parameters via specialization constants
            std::array<vk::SpecializationMapEntry, 2> specializationMapEntries{
                vk::SpecializationMapEntry{ 0, offsetof(SpecializationData, kernelSize), sizeof(uint32_t) },  // SSAO Kernel size
                vk::SpecializationMapEntry{ 1, offsetof(SpecializationData, radius), sizeof(float) },         // SSAO radius
            };

            vk::SpecializationInfo specializationInfo{ 2, specializationMapEntries.data(), sizeof(SpecializationData), &specializationData };
            builder.shaderStages[1].pSpecializationInfo = &specializationInfo;
            pipelines.ssao = builder.create(context.pipelineCache);
        }

        // SSAO blur pass
        {
            offscreen.ssaoBlur.setupDynamicRendering(builder);
            builder.layout = pipelineLayouts.ssaoBlur;
            // Destroy the fragment shader, but not the vertex shader
            device.destroy(builder.shaderStages[1].module);
            builder.shaderStages.resize(1);
            builder.loadShader(vkx::shaders::ssao::blur::frag, vk::ShaderStageFlagBits::eFragment);
            pipelines.ssaoBlur = builder.create(context.pipelineCache);
        }

        // Fill G-Buffer
        {
            builder.destroyShaderModules();
            offscreen.gbuffer.setupDynamicRendering(builder);
            builder.depthStencilState = true;
            builder.layout = pipelineLayouts.gBuffer;
            vertexLayout.appendVertexLayout(builder.vertexInputState);
            builder.loadShader(vkx::shaders::ssao::gbuffer::vert, vk::ShaderStageFlagBits::eVertex);
            builder.loadShader(vkx::shaders::ssao::gbuffer::frag, vk::ShaderStageFlagBits::eFragment);
            // Blend attachment states required for all color attachments
            // This is important, as color write mask will otherwise be 0x0 and you
            // won't see anything rendered to the attachment
            builder.colorBlendState.blendAttachmentStates.resize(3);
            pipelines.offscreen = builder.create(context.pipelineCache);
        }
    }

    float lerp(float a, float b, float f) { return a + f * (b - a); }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Scene matrices
        uniformBuffers.sceneMatrices = loader.createUniformBuffer(uboSceneMatrices);
        // SSAO parameters
        uniformBuffers.ssaoParams = loader.createUniformBuffer(uboSSAOParams);

        // Update
        updateUniformBufferMatrices();
        updateUniformBufferSSAOParams();

        // SSAO
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
        std::default_random_engine rndGen;

        // Sample kernel
        std::array<glm::vec4, SSAO_KERNEL_SIZE> ssaoKernel;
        for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i) {
            glm::vec3 sample(rndDist(rndGen) * 2.0 - 1.0, rndDist(rndGen) * 2.0 - 1.0, rndDist(rndGen));
            sample = glm::normalize(sample);
            sample *= rndDist(rndGen);
            float scale = float(i) / float(SSAO_KERNEL_SIZE);
            scale = lerp(0.1f, 1.0f, scale * scale);
            ssaoKernel[i] = glm::vec4(sample * scale, 0.0f);
        }

        // Upload as UBO
        uniformBuffers.ssaoKernel = loader.createUniformBuffer(ssaoKernel);

        // Random noise
        std::vector<glm::vec4> ssaoNoise(SSAO_NOISE_DIM * SSAO_NOISE_DIM);
        for (uint32_t i = 0; i < static_cast<uint32_t>(ssaoNoise.size()); i++) {
            ssaoNoise[i] = glm::vec4(rndDist(rndGen) * 2.0f - 1.0f, rndDist(rndGen) * 2.0f - 1.0f, 0.0f, 0.0f);
        }
        // Upload as texture
        vks::Image::Builder imageBuilder{ SSAO_NOISE_DIM };
        imageBuilder.withFormat(vk::Format::eR32G32B32A32Sfloat);
        imageBuilder.withUsage(vk::ImageUsageFlagBits::eSampled);
        textures.ssaoNoise.fromBuffer(ssaoNoise, imageBuilder);
    }

    void updateUniformBufferMatrices() {
        uboSceneMatrices.projection = camera.matrices.perspective;
        uboSceneMatrices.view = camera.matrices.view;
        uboSceneMatrices.model = glm::mat4(1.0f);
        uniformBuffers.sceneMatrices.copy(uboSceneMatrices);
    }

    void updateUniformBufferSSAOParams() {
        uboSSAOParams.projection = camera.matrices.perspective;
        uniformBuffers.ssaoParams.copy(uboSSAOParams);
    }

    void draw() override {
        prepareFrame();

        queueCommandBuffer(offscreen.cmdBuffer, vkx::RenderStates::OFFSCREEN_PRERENDER,
                           vk::PipelineStageFlagBits2::eColorAttachmentOutput | vk::PipelineStageFlagBits2::eAllGraphics);
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        loadAssets();
        prepareOffscreen();
        prepareUniformBuffers();
        setupDescriptorPool();
        setupLayoutsAndDescriptors();
        preparePipelines();
        buildCommandBuffers();
        buildDeferredCommandBuffer();
        prepared = true;
    }

    void viewChanged() override {
        updateUniformBufferMatrices();
        updateUniformBufferSSAOParams();
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Enable SSAO", &uboSSAOParams.ssao)) {
                updateUniformBufferSSAOParams();
            }
            if (ui.checkBox("SSAO blur", &uboSSAOParams.ssaoBlur)) {
                updateUniformBufferSSAOParams();
            }
            if (ui.checkBox("SSAO pass only", &uboSSAOParams.ssaoOnly)) {
                updateUniformBufferSSAOParams();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
