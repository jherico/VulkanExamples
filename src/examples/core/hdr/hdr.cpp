/*
 * Vulkan Example - HDR
 *
 * Note: Requires the separate asset pack (see data/README.md)
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */
#include <examples/offscreen.hpp>

#include <shaders/hdr/bloom.frag.inl>
#include <shaders/hdr/bloom.vert.inl>
#include <shaders/hdr/composition.frag.inl>
#include <shaders/hdr/composition.vert.inl>
#include <shaders/hdr/gbuffer.frag.inl>
#include <shaders/hdr/gbuffer.vert.inl>

class VulkanExample : public vkx::ExampleBase {
public:
    bool bloom = true;
    bool displaySkybox = true;

    // Vertex layout for the models

    struct {
        vks::texture::TextureCubeMap envmap;
    } textures;

    struct Models {
        vks::model::VertexLayout vertexLayout{ {
            vks::model::VERTEX_COMPONENT_POSITION,
            vks::model::VERTEX_COMPONENT_NORMAL,
            vks::model::VERTEX_COMPONENT_UV,
        } };
        vks::model::Model skybox;
        std::vector<vks::model::Model> objects;
        int32_t objectIndex = 1;
    } models;

    struct {
        vks::Buffer matrices;
        vks::Buffer params;
    } uniformBuffers;

    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 modelview;
    } uboVS;

    struct UBOParams {
        float exposure = 1.0f;
    } uboParams;

    struct {
        vk::Pipeline skybox;
        vk::Pipeline reflect;
        vk::Pipeline composition;
        vk::Pipeline bloom[2];
    } pipelines;

    struct {
        vk::PipelineLayout models;
        vk::PipelineLayout composition;
        vk::PipelineLayout bloomFilter;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet object;
        vk::DescriptorSet skybox;
        vk::DescriptorSet composition;
        vk::DescriptorSet bloomFilter;
    } descriptorSets;

    struct {
        vk::DescriptorSetLayout models;
        vk::DescriptorSetLayout composition;
        vk::DescriptorSetLayout bloomFilter;
    } descriptorSetLayouts;

    struct {
        vkx::offscreen::Renderer scene;
        vkx::offscreen::Renderer filter;
        vk::CommandBuffer cmdBuffer;
        vk::Semaphore semaphore;
    } offscreen;

    // struct {
    //     vk::Extent2D extent;
    //     VkFramebuffer frameBuffer;
    //     vks::Image color[1];
    //     VkRenderPass renderPass;
    //     VkSampler sampler;
    // } filterPass;

    std::vector<std::string> objectNames;

    VulkanExample() {
        title = "Hight dynamic range rendering";
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -4.0f));
        camera.setRotation(glm::vec3(0.0f, 180.0f, 0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroy(pipelines.skybox);
        device.destroy(pipelines.reflect);
        device.destroy(pipelines.composition);
        device.destroy(pipelines.bloom[0]);
        device.destroy(pipelines.bloom[1]);

        device.destroy(pipelineLayouts.models);
        device.destroy(pipelineLayouts.composition);
        device.destroy(pipelineLayouts.bloomFilter);

        device.destroy(descriptorSetLayouts.models);
        device.destroy(descriptorSetLayouts.composition);
        device.destroy(descriptorSetLayouts.bloomFilter);

        offscreen.filter.destroy();
        offscreen.scene.destroy();
        graphicsQueue.freeCommandBuffer(offscreen.cmdBuffer);
        device.destroy(offscreen.semaphore);

        for (auto& model : models.objects) {
            model.destroy();
        }
        models.skybox.destroy();
        uniformBuffers.matrices.destroy();
        uniformBuffers.params.destroy();
        textures.envmap.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) override {
        // Final composition
        // Scene
        commandBuffer.setViewport(0, viewport());
        commandBuffer.setScissor(0, scissor());
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, nullptr);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
        commandBuffer.draw(3, 1, 0, 0);

        // Bloom
        if (bloom) {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.bloom[0]);
            commandBuffer.draw(3, 1, 0, 0);
        }
    }

    // Prepare a new framebuffer and attachments for offscreen rendering (G-Buffer)
    void prepareOffscreen() {
        {
            vkx::offscreen::Builder builder{ size };
            builder.appendColorFormat(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eSampled, vks::util::clearColor(glm::vec4(0)));
            builder.appendColorFormat(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eSampled, vks::util::clearColor(glm::vec4(0)));
            builder.withDepthFormat(defaultDepthStencilFormat);
            offscreen.scene.prepare(builder);
        }

        // Bloom separable filter pass
        {
            vkx::offscreen::Builder builder{ size };
            builder.appendColorFormat(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eSampled);
            offscreen.filter.prepare(builder);
        }
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildOffscreenCommandBuffer() {
        if (!offscreen.cmdBuffer) {
            offscreen.cmdBuffer = graphicsQueue.createCommandBuffer();
        }

        // Create a semaphore used to synchronize offscreen rendering and usage
        if (!offscreen.semaphore) {
            offscreen.semaphore = device.createSemaphore({});
        }

        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        offscreen.scene.setLayout(offscreen.cmdBuffer, vks::util::ImageTransitionState::COLOR_ATTACHMENT, vks::util::ImageTransitionState::DEPTH_ATTACHMENT);

        // Clear values for all attachments written in the fragment sahder
        offscreen.cmdBuffer.beginRendering(offscreen.scene.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.scene.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.scene.size));
        // Skybox
        if (displaySkybox) {
            offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.models, 0, descriptorSets.skybox, nullptr);
            offscreen.cmdBuffer.bindVertexBuffers(0, models.skybox.vertices.buffer, { 0 });
            offscreen.cmdBuffer.bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
            offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
            offscreen.cmdBuffer.drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
        }

        // 3D object
        const auto& model = models.objects[models.objectIndex];
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.models, 0, descriptorSets.object, nullptr);
        offscreen.cmdBuffer.bindVertexBuffers(0, model.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(model.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.reflect);
        offscreen.cmdBuffer.drawIndexed(model.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();

        // Bloom filter

        offscreen.scene.setLayout(offscreen.cmdBuffer, vks::util::ImageTransitionState::SAMPLED);
        offscreen.filter.setLayout(offscreen.cmdBuffer, vks::util::ImageTransitionState::SAMPLED);
        offscreen.cmdBuffer.beginRendering(offscreen.scene.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.filter.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.filter.size));
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.bloomFilter, 0, descriptorSets.bloomFilter, nullptr);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.bloom[1]);
        offscreen.cmdBuffer.draw(3, 1, 0, 0);
        offscreen.cmdBuffer.endRenderPass();
        offscreen.cmdBuffer.end();
    }

    void loadAssets() override {
        // Models
        models.skybox.loadFromFile(getAssetPath() + "models/cube.obj", models.vertexLayout, 0.05f);
        std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
        objectNames = { "Sphere", "Teapot", "Torusknot", "Venus" };
        models.objects.resize(filenames.size());
        for (size_t i = 0; i < filenames.size(); ++i) {
            auto& model = models.objects[i];
            const auto& file = filenames[i];
            model.loadFromFile(getAssetPath() + "models/" + file, models.vertexLayout, 0.05f * (file == "venus.fbx" ? 3.0f : 1.0f));
        }

        // Load HDR cube map
        textures.envmap.loadFromFile(getAssetPath() + "textures/hdr/uffizi_cube.ktx", vk::Format::eR16G16B16A16Sfloat);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 4 },
            { vk::DescriptorType::eCombinedImageSampler, 6 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 4, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.models = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.models = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.models });

        // Bloom filter
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.bloomFilter = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.bloomFilter = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.bloomFilter });

        // G-Buffer composition
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.composition = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.composition = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.composition });
    }

    void setupDescriptorSets() {
        // 3D object descriptor set
        descriptorSets.object = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.models })[0];
        // Sky box descriptor set
        descriptorSets.skybox = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.models })[0];
        // Bloom filter
        descriptorSets.bloomFilter = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.bloomFilter })[0];
        // Composition descriptor set
        descriptorSets.composition = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.composition })[0];

        std::vector<vk::DescriptorImageInfo> colorDescriptors = {
            { offscreen.scene.sampler, offscreen.scene.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.scene.sampler, offscreen.scene.colorTargets[1].imageView, vk::ImageLayout::eReadOnlyOptimal },
            { offscreen.filter.sampler, offscreen.filter.colorTargets[1].imageView, vk::ImageLayout::eReadOnlyOptimal },
        };

        auto envMapDescriptor = textures.envmap.makeDescriptor(defaultSampler);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            vk::WriteDescriptorSet{ descriptorSets.object, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.matrices.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.object, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &envMapDescriptor },
            vk::WriteDescriptorSet{ descriptorSets.object, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.skybox, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.matrices.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.skybox, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &envMapDescriptor },
            vk::WriteDescriptorSet{ descriptorSets.skybox, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.bloomFilter, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[0] },
            vk::WriteDescriptorSet{ descriptorSets.bloomFilter, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[1] },
            vk::WriteDescriptorSet{ descriptorSets.composition, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[0] },
            vk::WriteDescriptorSet{ descriptorSets.composition, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorDescriptors[2] },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Final fullscreen composition pass pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.composition };
        pipelineBuilder.depthStencilState = { false };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        // Empty vertex input state, full screen triangles are generated by the vertex shader
        pipelineBuilder.loadShader(vkx::shaders::hdr::composition::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::hdr::composition::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.composition = pipelineBuilder.create(context.pipelineCache);

        // Bloom pass
        {
            pipelineBuilder.destroyShaderModules();
            auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
            blendAttachmentState.blendEnable = VK_TRUE;
            blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
            blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
            blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
            blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
            blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;
            pipelineBuilder.loadShader(vkx::shaders::hdr::bloom::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(vkx::shaders::hdr::bloom::frag, vk::ShaderStageFlagBits::eFragment);
            // Set constant parameters via specialization constants
            uint32_t dir = 1;
            vk::SpecializationMapEntry specializationMapEntry{ 0, 0, sizeof(uint32_t) };
            vk::SpecializationInfo specializationInfo{ 1, &specializationMapEntry, sizeof(dir), &dir };
            pipelineBuilder.shaderStages[1].pSpecializationInfo = &specializationInfo;
            pipelines.bloom[0] = pipelineBuilder.create(context.pipelineCache);
            // Second blur pass (into separate framebuffer)
            dir = 0;
            pipelines.bloom[1] = pipelineBuilder.create(context.pipelineCache);
            pipelineBuilder.destroyShaderModules();
        }

        // Object rendering pipelines

        {
            // Binding description
            models.vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
            // Skybox pipeline (background cube)
            auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
            blendAttachmentState.blendEnable = VK_FALSE;
            pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
            pipelineBuilder.layout = pipelineLayouts.models;
            pipelineBuilder.colorBlendState.blendAttachmentStates.resize(2);
            pipelineBuilder.loadShader(vkx::shaders::hdr::gbuffer::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(vkx::shaders::hdr::gbuffer::frag, vk::ShaderStageFlagBits::eFragment);
            // Set constant parameters via specialization constants
            uint32_t shadertype = 0;
            vk::SpecializationMapEntry specializationMapEntry{ 0, 0, sizeof(uint32_t) };
            vk::SpecializationInfo specializationInfo{ 1, &specializationMapEntry, sizeof(shadertype), &shadertype };
            pipelineBuilder.shaderStages[0].pSpecializationInfo = &specializationInfo;
            pipelineBuilder.shaderStages[1].pSpecializationInfo = &specializationInfo;
            pipelines.skybox = pipelineBuilder.create(context.pipelineCache);
            // Object rendering pipeline
            shadertype = 1;
            // Enable depth test and write
            pipelineBuilder.depthStencilState = { true };
            // Flip cull mode
            pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
            pipelines.reflect = pipelineBuilder.create(context.pipelineCache);
        }
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Matrices vertex shader uniform buffer
        uniformBuffers.matrices = loader.createUniformBuffer(uboVS);
        // Params
        uniformBuffers.params = loader.createUniformBuffer(uboParams);

        updateUniformBuffers();
        updateParams();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.modelview = camera.matrices.view;
        uniformBuffers.matrices.copy(uboVS);
    }

    void updateParams() { uniformBuffers.params.copy(uboParams); }

    void preRender() override {
        vks::frame::QueuedCommandBuilder builder{ offscreen.cmdBuffer, vkx::RenderStates::OFFSCREEN_PRERENDER,
                                                  vk::PipelineStageFlagBits2::eColorAttachmentOutput };
        queueCommandBuffer(builder);
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareUniformBuffers();
        prepareOffscreen();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.comboBox("Object type", &models.objectIndex, objectNames)) {
                updateUniformBuffers();
                buildOffscreenCommandBuffer();
            }
            if (ui.inputFloat("Exposure", &uboParams.exposure, 0.025f)) {
                updateParams();
            }
            if (ui.checkBox("Bloom", &bloom)) {
                buildCommandBuffers();
            }
            if (ui.checkBox("Skybox", &displaySkybox)) {
                buildOffscreenCommandBuffer();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
