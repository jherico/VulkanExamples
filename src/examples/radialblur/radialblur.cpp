/*
* Vulkan Example - Fullscreen radial blur (Single pass offscreen effect)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanOffscreenExampleBase.hpp>

// Texture properties
#define TEX_DIM 128

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_UV,
    vks::model::VERTEX_COMPONENT_COLOR,
    vks::model::VERTEX_COMPONENT_NORMAL,
} };

class VulkanExample : public vkx::OffscreenExampleBase {
public:
    bool blur = true;
    bool displayTexture = false;

    struct {
        vks::model::Model example;
        vks::model::Model quad;
    } meshes;

    struct {
        vks::Buffer vsScene;
        vks::Buffer vsQuad;
        vks::Buffer fsQuad;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVS;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboQuadVS;

    struct UboQuadFS {
        int32_t texWidth = TEX_DIM;
        int32_t texHeight = TEX_DIM;
        float radialBlurScale = 0.25f;
        float radialBlurStrength = 0.75f;
        glm::vec2 radialOrigin = glm::vec2(0.5f, 0.5f);
    } uboQuadFS;

    struct {
        vk::Pipeline radialBlur;
        vk::Pipeline colorPass;
        vk::Pipeline phongPass;
        vk::Pipeline fullScreenOnly;
    } pipelines;

    struct {
        vk::PipelineLayout radialBlur;
        vk::PipelineLayout scene;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet scene;
        vk::DescriptorSet quad;
    } descriptorSets;

    // Descriptor set layout is shared amongst
    // all descriptor sets
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.dolly(-12.0f);
        camera.setRotation({ -16.25f, -28.75f, 0.0f });
        timerSpeed *= 0.5f;
        title = "Vulkan Example - Radial blur";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroyPipeline(pipelines.radialBlur);
        device.destroyPipeline(pipelines.phongPass);
        device.destroyPipeline(pipelines.colorPass);
        device.destroyPipeline(pipelines.fullScreenOnly);

        device.destroyPipelineLayout(pipelineLayouts.radialBlur);
        device.destroyPipelineLayout(pipelineLayouts.scene);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // Meshes
        meshes.example.destroy();
        meshes.quad.destroy();

        // Uniform buffers
        uniformData.vsScene.destroy();
        uniformData.vsQuad.destroy();
        uniformData.fsQuad.destroy();
    }

    // The command buffer to copy for rendering
    // the offscreen scene and blitting it into
    // the texture target is only build once
    // and gets resubmitted
    void buildOffscreenCommandBuffer() override {
        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor();
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
        renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        offscreen.cmdBuffer.begin(cmdBufInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.colorPass);

        vk::DeviceSize offsets = 0;
        offscreen.cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, offsets);
        offscreen.cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();

        offscreen.cmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        // 3D scene
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phongPass);

        cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);

        // Fullscreen quad with radial blur
        if (blur) {
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.radialBlur, 0, descriptorSets.quad, nullptr);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, (displayTexture) ? pipelines.fullScreenOnly : pipelines.radialBlur);
            cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
        }
    }

    void loadMeshes() { meshes.example.loadFromFile(context, getAssetPath() + "models/glowsphere.dae", vertexLayout, 0.05f); }

    // Setup vertices for a single uv-mapped quad
    void generateQuad() {
        struct Vertex {
            float pos[3];
            float uv[2];
            float col[3];
            float normal[3];
        };

#define QUAD_COLOR_NORMAL { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }
        std::vector<Vertex> vertexBuffer = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }, QUAD_COLOR_NORMAL },
                                             { { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f }, QUAD_COLOR_NORMAL },
                                             { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, QUAD_COLOR_NORMAL },
                                             { { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }, QUAD_COLOR_NORMAL } };
#undef QUAD_COLOR_NORMAL

        meshes.quad.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        // Example uses three ubos and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 4 },
            { vk::DescriptorType::eCombinedImageSampler, 2 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Textured quad pipeline layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 2 : Fragment shader uniform buffer
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.radialBlur = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        // Offscreen pipeline layout
        pipelineLayouts.scene = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        // Textured quad descriptor set
        descriptorSets.quad = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texDescriptor{ offscreen.framebuffers[0].colors[0].sampler, offscreen.framebuffers[0].colors[0].view,
                                               vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.quad, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsScene.descriptor },
            // Binding 1 : Fragment shader texture sampler
            { descriptorSets.quad, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
            // Binding 2 : Fragment shader uniform buffer
            { descriptorSets.quad, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.fsQuad.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Offscreen 3D scene descriptor set
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> offscreenWriteDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsQuad.descriptor },
        };
        device.updateDescriptorSets(offscreenWriteDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.radialBlur, renderPass };
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        // Radial blur pipeline
        pipelineBuilder.depthStencilState = { false };
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
        // Load shaders
        pipelineBuilder.loadShader(getAssetPath() + "shaders/radialblur/radialblur.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/radialblur/radialblur.frag.spv", vk::ShaderStageFlagBits::eFragment);

        pipelines.radialBlur = pipelineBuilder.create(context.pipelineCache);

        // No blending (for debug display)
        blendAttachmentState.blendEnable = VK_FALSE;
        pipelines.fullScreenOnly = pipelineBuilder.create(context.pipelineCache);

        // Phong pass
        pipelineBuilder.destroyShaderModules();
        pipelineBuilder.loadShader(getAssetPath() + "shaders/radialblur/phongpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/radialblur/phongpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.layout = pipelineLayouts.scene;
        blendAttachmentState.blendEnable = VK_FALSE;
        pipelineBuilder.depthStencilState = { true };
        pipelines.phongPass = pipelineBuilder.create(context.pipelineCache);

        // Color only pass (offscreen blur base)
        pipelineBuilder.destroyShaderModules();
        pipelineBuilder.loadShader(getAssetPath() + "shaders/radialblur/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/radialblur/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.colorPass = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Phong and color pass vertex shader uniform buffer
        uniformData.vsScene = context.createUniformBuffer(uboVS);

        // Fullscreen quad vertex shader uniform buffer
        uniformData.vsQuad = context.createUniformBuffer(uboVS);

        // Fullscreen quad fragment shader uniform buffer
        uniformData.fsQuad = context.createUniformBuffer(uboVS);

        updateUniformBuffersScene();
        updateUniformBuffersScreen();
    }

    // Update uniform buffers for rendering the 3D scene
    void updateUniformBuffersScene() {
        uboQuadVS.projection = getProjection();

        uboQuadVS.model = glm::mat4();
        uboQuadVS.model = camera.matrices.view;
        uboQuadVS.model = glm::rotate(uboQuadVS.model, glm::radians(timer * 360.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        uniformData.vsQuad.copy(uboQuadVS);
    }

    // Update uniform buffers for the fullscreen quad
    void updateUniformBuffersScreen() {
        // Vertex shader
        uboVS.projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        uboVS.model = glm::mat4();
        uniformData.vsScene.copy(uboVS);

        // Fragment shader
        uniformData.fsQuad.copy(uboQuadFS);
    }

    void prepare() override {
        offscreen.size = glm::uvec2(TEX_DIM);
        OffscreenExampleBase::prepare();
        generateQuad();
        loadMeshes();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildOffscreenCommandBuffer();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            updateUniformBuffersScene();
        }
    }

    void viewChanged() override {
        updateUniformBuffersScene();
        updateUniformBuffersScreen();
    }

    void keyPressed(uint32_t keyCode) override {
        switch (keyCode) {
            case KEY_B:
            case GAMEPAD_BUTTON_A:
                toggleBlur();
                break;
            case KEY_T:
            case GAMEPAD_BUTTON_X:
                toggleTextureDisplay();
                break;
        }
    }

    void toggleBlur() {
        blur = !blur;
        updateUniformBuffersScene();
        buildCommandBuffers();
    }

    void toggleTextureDisplay() {
        displayTexture = !displayTexture;
        buildCommandBuffers();
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Radial blur", &blur)) {
                buildCommandBuffers();
            }
            if (ui.checkBox("Dsiplay render target", &displayTexture)) {
                buildCommandBuffers();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
