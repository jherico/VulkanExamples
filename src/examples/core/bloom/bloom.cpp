/*
 * Vulkan Example - Multi pass offscreen rendering (bloom)
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/offscreen.hpp>
#include <rendering/model.hpp>
#include <rendering/texture.hpp>

#include <shaders/bloom/colorpass.frag.inl>
#include <shaders/bloom/colorpass.vert.inl>
#include <shaders/bloom/gaussblur.frag.inl>
#include <shaders/bloom/gaussblur.vert.inl>
#include <shaders/bloom/phongpass.frag.inl>
#include <shaders/bloom/phongpass.vert.inl>
#include <shaders/bloom/skybox.frag.inl>
#include <shaders/bloom/skybox.vert.inl>

// Texture properties
#define TEX_DIM 256

// Offscreen frame buffer properties
#define FB_DIM TEX_DIM
#define FB_COLOR_FORMAT TEX_FORMAT

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
} };

class VulkanExample : public vkx::OffscreenExampleBase {
    using Parent = OffscreenExampleBase;

public:
    bool& bloom = offscreen.active;

    struct {
        vks::texture::TextureCubeMap cubemap;
    } textures;

    struct {
        vks::model::Model ufo;
        vks::model::Model ufoGlow;
        vks::model::Model skyBox;
    } meshes;

    struct {
        vks::Buffer scene;
        vks::Buffer skyBox;
        vks::Buffer blurParams;
    } uniformBuffers;

    struct UBO {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
    };

    struct UBOBlurParams {
        float blurScale = 1.0f;
        float blurStrength = 1.5f;
    };

    struct {
        UBO scene, skyBox;
        UBOBlurParams blurParams;
    } ubos;

    struct {
        vk::Pipeline blurVert;
        vk::Pipeline blurHorz;
        vk::Pipeline glowPass;
        vk::Pipeline phongPass;
        vk::Pipeline skyBox;
    } pipelines;

    struct {
        vk::PipelineLayout blur;
        vk::PipelineLayout scene;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet blurVert;
        vk::DescriptorSet blurHorz;
        vk::DescriptorSet scene;
        vk::DescriptorSet skyBox;
    } descriptorSets;

    // Descriptor set layout is shared amongst
    // all descriptor sets
    struct {
        vk::DescriptorSetLayout blur;
        vk::DescriptorSetLayout scene;
    } descriptorSetLayouts;

    VulkanExample()
        : vkx::OffscreenExampleBase() {
        timerSpeed *= 0.5f;
        title = "Vulkan Example - Bloom";
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -10.25f));
        camera.setRotation(glm::vec3(7.5f, -343.0f, 0.0f));
        camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroy(pipelines.blurVert);
        device.destroy(pipelines.blurHorz);
        device.destroy(pipelines.phongPass);
        device.destroy(pipelines.glowPass);
        device.destroy(pipelines.skyBox);

        device.destroy(pipelineLayouts.blur);
        device.destroy(pipelineLayouts.scene);

        device.destroy(descriptorSetLayouts.blur);
        device.destroy(descriptorSetLayouts.scene);

        // Assets
        meshes.ufo.destroy();
        meshes.ufoGlow.destroy();
        meshes.skyBox.destroy();
        textures.cubemap.destroy();

        // Uniform buffers
        uniformBuffers.scene.destroy();
        uniformBuffers.skyBox.destroy();
        uniformBuffers.blurParams.destroy();
    }

    // Render the 3D scene into a texture target
    void buildOffscreenCommandBuffer() override {
        vk::DeviceSize offset = 0;

        // Horizontal blur
        offscreen.cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        // Draw the unblurred geometry to framebuffer 1
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));

        auto& renderingInfo = offscreen.renderingInfo;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &offscreen.colorAttachmentsInfo[0];

        // Draw the bloom geometry.
        {
            offscreen.cmdBuffer.beginRendering(renderingInfo);
            offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
            offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.glowPass);
            offscreen.cmdBuffer.bindVertexBuffers(0, meshes.ufoGlow.vertices.buffer, offset);
            offscreen.cmdBuffer.bindIndexBuffer(meshes.ufoGlow.indices.buffer, 0, vk::IndexType::eUint32);
            for (const auto& part : meshes.ufoGlow.parts) {
                offscreen.cmdBuffer.drawIndexed(part.indexCount, 1, part.indexBase, 0, 0);
            }
            offscreen.cmdBuffer.endRendering();
        }

        renderingInfo.pColorAttachments = &offscreen.colorAttachmentsInfo[1];
        {
            // Draw a vertical blur pass from framebuffer 1's texture into framebuffer 2
            offscreen.cmdBuffer.beginRendering(renderingInfo);
            offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, descriptorSets.blurVert, nullptr);
            offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.blurVert);
            offscreen.cmdBuffer.draw(3, 1, 0, 0);
            offscreen.cmdBuffer.endRendering();
        }
        offscreen.cmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::DeviceSize offset = 0;
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        // Skybox
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.skyBox, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skyBox);
        cmdBuffer.bindVertexBuffers(0, meshes.skyBox.vertices.buffer, offset);
        cmdBuffer.bindIndexBuffer(meshes.skyBox.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.skyBox.indexCount, 1, 0, 0, 0);

        // 3D scene
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phongPass);
        cmdBuffer.bindVertexBuffers(0, meshes.ufo.vertices.buffer, offset);
        cmdBuffer.bindIndexBuffer(meshes.ufo.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.ufo.indexCount, 1, 0, 0, 0);

        // Render vertical blurred scene applying a horizontal blur
        if (bloom) {
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, descriptorSets.blurHorz, nullptr);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.blurHorz);
            cmdBuffer.draw(3, 1, 0, 0);
        }
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 8 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 6 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 5, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Quad pipeline layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.blur = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.blur = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.blur });

        setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.scene = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.scene = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.scene });
    }

    void setupDescriptorSet() {
        descriptorSets.blurVert = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.blur })[0];
        descriptorSets.blurHorz = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.blur })[0];
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.scene })[0];
        descriptorSets.skyBox = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.scene })[0];

        // Vertical blur
        vk::DescriptorImageInfo texDescriptorVert{ offscreen.sampler, offscreen.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal };
        // Horizontal blur
        vk::DescriptorImageInfo texDescriptorHorz{ offscreen.sampler, offscreen.colorTargets[1].imageView, vk::ImageLayout::eReadOnlyOptimal };
        auto cubemapDesc = textures.cubemap.makeDescriptor(defaultSampler);
        device.updateDescriptorSets(
            {
                vk::WriteDescriptorSet{ descriptorSets.blurVert, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.blurParams.descriptor },
                vk::WriteDescriptorSet{ descriptorSets.blurVert, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorVert },
                vk::WriteDescriptorSet{ descriptorSets.blurHorz, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.blurParams.descriptor },
                vk::WriteDescriptorSet{ descriptorSets.blurHorz, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorHorz },
                // 3D scene
                vk::WriteDescriptorSet{ descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.scene.descriptor },
                // Skybox
                vk::WriteDescriptorSet{ descriptorSets.skyBox, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.skyBox.descriptor },
                vk::WriteDescriptorSet{ descriptorSets.skyBox, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cubemapDesc },
            },
            nullptr);
    }

    void preparePipelines() {
        vks::pipelines::PipelineVertexInputStateCreateInfo vertexInputState;
        vertexLayout.appendVertexLayout(vertexInputState);

        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.blur };
            // No depth or stencil testing enabled
            pipelineBuilder.dynamicRendering(offscreen.colorTargets[0].image.createInfo.format);
            pipelineBuilder.depthStencilState.depthTestEnable = VK_FALSE;
            pipelineBuilder.depthStencilState.depthWriteEnable = VK_FALSE;
            pipelineBuilder.depthStencilState.stencilTestEnable = VK_FALSE;
            pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
            pipelineBuilder.colorBlendState.blendAttachmentStates.resize(1);
            auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
            // Additive blending
            blendAttachmentState.blendEnable = VK_TRUE;
            blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
            blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
            blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
            blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
            blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.loadShader(vkx::shaders::bloom::gaussblur::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(vkx::shaders::bloom::gaussblur::frag, vk::ShaderStageFlagBits::eFragment);

            // Specialization info to compile two versions of the shader without
            // relying on shader branching at runtime
            uint32_t blurdirection = 0;
            vk::SpecializationMapEntry specializationMapEntry{ 0, 0, sizeof(uint32_t) };
            vk::SpecializationInfo specializationInfo{ 1, &specializationMapEntry, sizeof(uint32_t), &blurdirection };

            // Vertical blur pipeline
            pipelineBuilder.shaderStages[1].pSpecializationInfo = &specializationInfo;
            pipelines.blurVert = pipelineBuilder.create(context.pipelineCache);
            // Horizontal blur pipeline
            blurdirection = 1;
            pipelines.blurHorz = pipelineBuilder.create(context.pipelineCache);
        }

        // Vertical gauss blur
        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.scene };
            pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
            pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.loadShader(vkx::shaders::bloom::phongpass::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(vkx::shaders::bloom::phongpass::frag, vk::ShaderStageFlagBits::eFragment);
            pipelines.phongPass = pipelineBuilder.create(context.pipelineCache);
        }

        // Color only pass (offscreen blur base)
        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.scene };
            pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
            pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.loadShader(vkx::shaders::bloom::colorpass::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(vkx::shaders::bloom::colorpass::frag, vk::ShaderStageFlagBits::eFragment);
            pipelines.glowPass = pipelineBuilder.create(context.pipelineCache);
        }

        // Skybox (cubemap
        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.scene };
            pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.depthStencilState = { false };
            pipelineBuilder.loadShader(vkx::shaders::bloom::skybox::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(vkx::shaders::bloom::skybox::frag, vk::ShaderStageFlagBits::eFragment);
            pipelines.skyBox = pipelineBuilder.create(context.pipelineCache);
        }
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Phong and color pass vertex shader uniform buffer
        uniformBuffers.scene = loader.createUniformBuffer(ubos.scene);
        // Fullscreen quad fragment shader uniform buffers
        uniformBuffers.blurParams = loader.createUniformBuffer(ubos.blurParams);
        // Skybox
        uniformBuffers.skyBox = loader.createUniformBuffer(ubos.skyBox);

        // Intialize uniform buffers
        updateUniformBuffersScene();
        updateUniformBuffersBlur();
    }

    // Update uniform buffers for rendering the 3D scene
    void updateUniformBuffersScene() {
        // UFO
        ubos.scene.projection = camera.matrices.perspective;
        ubos.scene.view = camera.matrices.view;

        ubos.scene.model =
            glm::translate(glm::mat4(1.0f), glm::vec3(sin(glm::radians(timer * 360.0f)) * 0.25f, -1.0f, cos(glm::radians(timer * 360.0f)) * 0.25f));
        ubos.scene.model = glm::rotate(ubos.scene.model, -sinf(glm::radians(timer * 360.0f)) * 0.15f, glm::vec3(1.0f, 0.0f, 0.0f));
        ubos.scene.model = glm::rotate(ubos.scene.model, glm::radians(timer * 360.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        uniformBuffers.scene.copy(ubos.scene);

        // Skybox
        ubos.skyBox.projection = camera.matrices.perspective;
        ubos.skyBox.view = glm::mat4(glm::mat3(camera.matrices.view));
        ubos.skyBox.model = glm::mat4(1.0f);
        uniformBuffers.skyBox.copy(ubos.skyBox);
    }

    // Update uniform buffers for the fullscreen quad
    void updateUniformBuffersBlur() { uniformBuffers.blurParams.copy(ubos.blurParams); }

    void loadAssets() override {
        meshes.ufoGlow.loadFromFile(getAssetPath() + "models/retroufo_glow.dae", vertexLayout, 0.05f);
        meshes.ufo.loadFromFile(getAssetPath() + "models/retroufo.dae", vertexLayout, 0.05f);
        meshes.skyBox.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout, 1.0f);
        textures.cubemap.loadFromFile(getAssetPath() + "textures/cubemap_space.ktx", vk::Format::eR8G8B8A8Unorm);
    }

    // void draw() override {
    //     prepareFrame();

    //    // Offscreen rendering
    //    if (bloom) {
    //        graphicsQueue.submit(offscreen.cmdBuffer, semaphores.imageAcquired, vk::PipelineStageFlagBits::eBottomOfPipe,
    //                             offscreen.semaphores.swapchainFilled);
    //        renderWaitSemaphores = { offscreen.semaphores.swapchainFilled };
    //    } else {
    //        renderWaitSemaphores = { semaphores.imageAcquired };
    //    }

    //    // Scene rendering
    //    drawCurrentCommandBuffer();
    //    submitFrame();
    //}

    void prepareOffscreen() override {
        vkx::offscreen::Builder builder{ TEX_DIM };
        builder.appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eSampled)
            .appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eSampled)
            .withDepthFormat(defaultDepthStencilFormat);
        offscreen.prepare(builder);
    }

    void prepare() override {
        Parent::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void update(float deltaTime) override {
        Parent::update(deltaTime);
        if (!paused) {
            updateUniformBuffersScene();
        }
    }

    void viewChanged() override { updateUniformBuffersScene(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Bloom", &bloom)) {
                buildCommandBuffers();
            }
            if (ui.inputFloat("Scale", &ubos.blurParams.blurScale, 0.1f, "%.2f")) {
                updateUniformBuffersBlur();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
