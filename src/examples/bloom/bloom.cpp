/*
* Vulkan Example - Multi pass offscreen rendering (bloom)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanOffscreenExampleBase.hpp>
#include <vks/model.hpp>
#include <vks/texture.hpp>

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
    bool bloom = true;

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

        device.destroyPipeline(pipelines.blurVert);
        device.destroyPipeline(pipelines.blurHorz);
        device.destroyPipeline(pipelines.phongPass);
        device.destroyPipeline(pipelines.glowPass);
        device.destroyPipeline(pipelines.skyBox);

        device.destroyPipelineLayout(pipelineLayouts.blur);
        device.destroyPipelineLayout(pipelineLayouts.scene);

        device.destroyDescriptorSetLayout(descriptorSetLayouts.blur);
        device.destroyDescriptorSetLayout(descriptorSetLayouts.scene);

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
        vk::Viewport viewport = vks::util::viewport(offscreen.size);
        vk::Rect2D scissor = vks::util::rect2D(offscreen.size);
        vk::DeviceSize offset = 0;

        // Horizontal blur
        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 1.0f });
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        offscreen.cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        offscreen.cmdBuffer.begin(cmdBufInfo);

        // Draw the unblurred geometry to framebuffer 1
        offscreen.cmdBuffer.setViewport(0, viewport);
        offscreen.cmdBuffer.setScissor(0, scissor);

        // Draw the bloom geometry.
        {
            vk::RenderPassBeginInfo renderPassBeginInfo;
            renderPassBeginInfo.renderPass = offscreen.renderPass;
            renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
            renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
            renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
            renderPassBeginInfo.clearValueCount = 2;
            renderPassBeginInfo.pClearValues = clearValues;
            offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
            offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
            offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.glowPass);
            offscreen.cmdBuffer.bindVertexBuffers(0, meshes.ufoGlow.vertices.buffer, offset);
            offscreen.cmdBuffer.bindIndexBuffer(meshes.ufoGlow.indices.buffer, 0, vk::IndexType::eUint32);

            for (const auto& part : meshes.ufoGlow.parts) {
                offscreen.cmdBuffer.drawIndexed(part.indexCount, 1, part.indexBase, 0, 0);
            }
            offscreen.cmdBuffer.endRenderPass();
        }

        {
            vk::RenderPassBeginInfo renderPassBeginInfo;
            renderPassBeginInfo.renderPass = offscreen.renderPass;
            renderPassBeginInfo.framebuffer = offscreen.framebuffers[1].framebuffer;
            renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
            renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
            renderPassBeginInfo.clearValueCount = 2;
            renderPassBeginInfo.pClearValues = clearValues;
            // Draw a vertical blur pass from framebuffer 1's texture into framebuffer 2
            offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
            offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.blur, 0, descriptorSets.blurVert, nullptr);
            offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.blurVert);
            offscreen.cmdBuffer.draw(3, 1, 0, 0);
            offscreen.cmdBuffer.endRenderPass();
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
        vk::DescriptorImageInfo texDescriptorVert{ offscreen.framebuffers[0].colors[0].sampler, offscreen.framebuffers[0].colors[0].view,
                                                   vk::ImageLayout::eShaderReadOnlyOptimal };
        // Horizontal blur
        vk::DescriptorImageInfo texDescriptorHorz{ offscreen.framebuffers[1].colors[0].sampler, offscreen.framebuffers[1].colors[0].view,
                                                   vk::ImageLayout::eShaderReadOnlyOptimal };
        // vk::Image descriptor for the cube map texture
        vk::DescriptorImageInfo cubeMapDescriptor{ textures.cubemap.sampler, textures.cubemap.view, vk::ImageLayout::eGeneral };

        device.updateDescriptorSets(
            {
                { descriptorSets.blurVert, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.blurParams.descriptor },
                { descriptorSets.blurVert, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorVert },
                { descriptorSets.blurHorz, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.blurParams.descriptor },
                { descriptorSets.blurHorz, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorHorz },
                // 3D scene
                { descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.scene.descriptor },
                // Skybox
                { descriptorSets.skyBox, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.skyBox.descriptor },
                { descriptorSets.skyBox, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cubeMapDescriptor },
            },
            nullptr);
    }

    void preparePipelines() {
        vks::pipelines::PipelineVertexInputStateCreateInfo vertexInputState;
        vertexInputState.appendVertexLayout(vertexLayout);

        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.blur, offscreen.renderPass };
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
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/gaussblur.vert.spv", vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/gaussblur.frag.spv", vk::ShaderStageFlagBits::eFragment);

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
            pipelineBuilder.renderPass = renderPass;
            pipelines.blurHorz = pipelineBuilder.create(context.pipelineCache);
        }

        // Vertical gauss blur
        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.scene, renderPass };
            pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/phongpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/phongpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipelines.phongPass = pipelineBuilder.create(context.pipelineCache);
        }

        // Color only pass (offscreen blur base)
        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.scene, renderPass };
            pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipelines.glowPass = pipelineBuilder.create(context.pipelineCache);
        }

        // Skybox (cubemap
        {
            vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.scene, renderPass };
            pipelineBuilder.vertexInputState = vertexInputState;
            pipelineBuilder.depthStencilState = { false };
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/skybox.vert.spv", vk::ShaderStageFlagBits::eVertex);
            pipelineBuilder.loadShader(getAssetPath() + "shaders/bloom/skybox.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipelines.skyBox = pipelineBuilder.create(context.pipelineCache);
        }
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Phong and color pass vertex shader uniform buffer
        uniformBuffers.scene = context.createUniformBuffer(ubos.scene);
        // Fullscreen quad fragment shader uniform buffers
        uniformBuffers.blurParams = context.createUniformBuffer(ubos.blurParams);
        // Skybox
        uniformBuffers.skyBox = context.createUniformBuffer(ubos.skyBox);

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
    void updateUniformBuffersBlur() {
        uniformBuffers.blurParams.copy(ubos.blurParams);
    }

    void loadAssets() override {
        meshes.ufoGlow.loadFromFile(context, getAssetPath() + "models/retroufo_glow.dae", vertexLayout, 0.05f);
        meshes.ufo.loadFromFile(context, getAssetPath() + "models/retroufo.dae", vertexLayout, 0.05f);
        meshes.skyBox.loadFromFile(context, getAssetPath() + "models/cube.obj", vertexLayout, 1.0f);
        textures.cubemap.loadFromFile(context, getAssetPath() + "textures/cubemap_space.ktx", vk::Format::eR8G8B8A8Unorm);
    }

    void draw() override {
        prepareFrame();

        // Offscreen rendering
        if (bloom) {
            context.submit(offscreen.cmdBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreen.renderComplete);
            renderWaitSemaphores = { offscreen.renderComplete };
        } else {
            renderWaitSemaphores = { semaphores.acquireComplete };
        }

        // Scene rendering
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void prepare() override {
        offscreen.framebuffers.resize(2);
        offscreen.size = glm::uvec2(TEX_DIM);
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

    void viewChanged() override {
        updateUniformBuffersScene();
    }

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
