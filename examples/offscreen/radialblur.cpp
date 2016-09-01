/*
* Vulkan Example - Fullscreen radial blur (Single pass offscreen effect)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanOffscreenExampleBase.hpp"


// Texture properties
#define TEX_DIM 128

// Vertex layout for this example
std::vector<vkx::VertexLayout> vertexLayout =
{
    vkx::VertexLayout::VERTEX_LAYOUT_POSITION,
    vkx::VertexLayout::VERTEX_LAYOUT_UV,
    vkx::VertexLayout::VERTEX_LAYOUT_COLOR,
    vkx::VertexLayout::VERTEX_LAYOUT_NORMAL
};

class VulkanExample : public vkx::OffscreenExampleBase {
public:
    bool blur = true;
    bool displayTexture = false;

    struct {
        vkx::MeshBuffer example;
        vkx::MeshBuffer quad;
    } meshes;

    struct {
        vk::PipelineVertexInputStateCreateInfo inputState;
        std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    struct {
        vkx::UniformData vsScene;
        vkx::UniformData vsQuad;
        vkx::UniformData fsQuad;
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

    VulkanExample() : vkx::OffscreenExampleBase(ENABLE_VALIDATION) {
        camera.setZoom(-12.0f);
        camera.setRotation({ -16.25f, -28.75f, 0.0f });
        timerSpeed *= 0.5f;
        enableTextOverlay = true;
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
        clearValues[0].color = vkx::clearColor({ 0, 0, 0, 0 });
        clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
        renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        offscreen.cmdBuffer.begin(cmdBufInfo);
        offscreen.cmdBuffer.setViewport(0, vkx::viewport(offscreen.size));
        offscreen.cmdBuffer.setScissor(0, vkx::rect2D(offscreen.size));
        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.colorPass);

        vk::DeviceSize offsets = 0;
        offscreen.cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.example.vertices.buffer, offsets);
        offscreen.cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();

        offscreen.cmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.setViewport(0, vkx::viewport(size));
        cmdBuffer.setScissor(0, vkx::rect2D(size));
        // 3D scene
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phongPass);

        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.example.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);

        // Fullscreen quad with radial blur
        if (blur) {
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.radialBlur, 0, descriptorSets.quad, nullptr);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, (displayTexture) ? pipelines.fullScreenOnly : pipelines.radialBlur);
            cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.quad.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
        }
    }

    void loadMeshes() {
        meshes.example = loadMesh(getAssetPath() + "models/glowsphere.dae", vertexLayout, 0.05f);
    }

    // Setup vertices for a single uv-mapped quad
    void generateQuad() {
        struct Vertex {
            float pos[3];
            float uv[2];
            float col[3];
            float normal[3];
        };

#define QUAD_COLOR_NORMAL { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }
        std::vector<Vertex> vertexBuffer =
        {
            { { 1.0f, 1.0f, 0.0f },{ 1.0f, 1.0f }, QUAD_COLOR_NORMAL },
            { { 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f }, QUAD_COLOR_NORMAL },
            { { 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f }, QUAD_COLOR_NORMAL },
            { { 1.0f, 0.0f, 0.0f },{ 1.0f, 0.0f }, QUAD_COLOR_NORMAL }
        };
#undef QUAD_COLOR_NORMAL

        meshes.quad.vertices = stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        std::vector<uint32_t> indexBuffer = { 0,1,2, 2,3,0 };
        meshes.quad.indexCount = indexBuffer.size();
        meshes.quad.indices = stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupVertexDescriptions() {
        // Binding description
        vertices.bindingDescriptions.resize(1);
        vertices.bindingDescriptions[0] =
            vkx::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vkx::vertexSize(vertexLayout), vk::VertexInputRate::eVertex);

        // Attribute descriptions
        vertices.attributeDescriptions.resize(4);
        // Location 0 : Position
        vertices.attributeDescriptions[0] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0,  vk::Format::eR32G32B32Sfloat, 0);
        // Location 1 : Texture coordinates
        vertices.attributeDescriptions[1] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1,  vk::Format::eR32G32Sfloat, sizeof(float) * 3);
        // Location 2 : Color
        vertices.attributeDescriptions[2] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2,  vk::Format::eR32G32B32Sfloat, sizeof(float) * 5);
        // Location 3 : Normal
        vertices.attributeDescriptions[3] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 3,  vk::Format::eR32G32B32Sfloat, sizeof(float) * 8);

        vertices.inputState = vk::PipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
        vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
        vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
        vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
    }

    void setupDescriptorPool() {
        // Example uses three ubos and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes =
        {
            vkx::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
            vkx::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo =
            vkx::descriptorPoolCreateInfo(poolSizes.size(), poolSizes.data(), 2);

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        // Textured quad pipeline layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex,
                0),
            // Binding 1 : Fragment shader image sampler
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                1),
            // Binding 2 : Fragment shader uniform buffer
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eFragment,
                2)
        };

        vk::DescriptorSetLayoutCreateInfo descriptorLayout =
            vkx::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), setLayoutBindings.size());

        descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vkx::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

        pipelineLayouts.radialBlur = device.createPipelineLayout(pPipelineLayoutCreateInfo);

        // Offscreen pipeline layout
        pipelineLayouts.scene = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    }

    void setupDescriptorSet() {
        // Textured quad descriptor set
        vk::DescriptorSetAllocateInfo allocInfo =
            vkx::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

        descriptorSets.quad = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texDescriptor =
            vkx::descriptorImageInfo(offscreen.framebuffers[0].colors[0].sampler, offscreen.framebuffers[0].colors[0].view, vk::ImageLayout::eShaderReadOnlyOptimal);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::writeDescriptorSet(
                descriptorSets.quad,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.vsScene.descriptor),
            // Binding 1 : Fragment shader texture sampler
            vkx::writeDescriptorSet(
                descriptorSets.quad,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                &texDescriptor),
            // Binding 2 : Fragment shader uniform buffer
            vkx::writeDescriptorSet(
                descriptorSets.quad,
                vk::DescriptorType::eUniformBuffer,
                2,
                &uniformData.fsQuad.descriptor)
        };

        device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

        // Offscreen 3D scene descriptor set
        descriptorSets.scene = device.allocateDescriptorSets(allocInfo)[0];

        std::vector<vk::WriteDescriptorSet> offscreenWriteDescriptorSets =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::writeDescriptorSet(
                descriptorSets.scene,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.vsQuad.descriptor)
        };
        device.updateDescriptorSets(offscreenWriteDescriptorSets.size(), offscreenWriteDescriptorSets.data(), 0, NULL);
    }

    void preparePipelines() {
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vkx::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

        vk::PipelineRasterizationStateCreateInfo rasterizationState =
            vkx::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise);

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

        // Radial blur pipeline
        // Load shaders
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = loadShader(getAssetPath() + "shaders/radialblur/radialblur.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/radialblur/radialblur.frag.spv", vk::ShaderStageFlagBits::eFragment);

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
            vkx::pipelineCreateInfo(pipelineLayouts.radialBlur, renderPass);

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

        // Additive blending
        blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;

        pipelines.radialBlur = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];

        // No blending (for debug display)
        blendAttachmentState.blendEnable = VK_FALSE;
        pipelines.fullScreenOnly = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];

        // Phong pass
        shaderStages[0] = loadShader(getAssetPath() + "shaders/radialblur/phongpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/radialblur/phongpass.frag.spv", vk::ShaderStageFlagBits::eFragment);

        pipelineCreateInfo.layout = pipelineLayouts.scene;
        blendAttachmentState.blendEnable = VK_FALSE;
        depthStencilState.depthWriteEnable = VK_TRUE;

        pipelines.phongPass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];

        // Color only pass (offscreen blur base)
        shaderStages[0] = loadShader(getAssetPath() + "shaders/radialblur/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/radialblur/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);

        pipelines.colorPass = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Phong and color pass vertex shader uniform buffer
        uniformData.vsScene = createUniformBuffer(uboVS);

        // Fullscreen quad vertex shader uniform buffer
        uniformData.vsQuad = createUniformBuffer(uboVS);

        // Fullscreen quad fragment shader uniform buffer
        uniformData.fsQuad = createUniformBuffer(uboVS);

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

    void prepare() {
        offscreen.size = glm::uvec2(TEX_DIM);
        OffscreenExampleBase::prepare();
        generateQuad();
        loadMeshes();
        setupVertexDescriptions();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildOffscreenCommandBuffer();
        updateDrawCommandBuffers();
        prepared = true;
    }

    void render() override  {
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
        case GLFW_KEY_B:
        case GAMEPAD_BUTTON_A:
            toggleBlur();
            break;
        case GLFW_KEY_T:
        case GAMEPAD_BUTTON_X:
            toggleTextureDisplay();
            break;
        }
    }

    void getOverlayText(vkx::TextOverlay *textOverlay) override  {
#if defined(__ANDROID__)
        textOverlay->addText("Press \"Button A\" to toggle blur", 5.0f, 85.0f, vkx::TextOverlay::alignLeft);
        textOverlay->addText("Press \"Button X\" to display offscreen texture", 5.0f, 105.0f, vkx::TextOverlay::alignLeft);
#else
        textOverlay->addText("Press \"B\" to toggle blur", 5.0f, 85.0f, vkx::TextOverlay::alignLeft);
        textOverlay->addText("Press \"T\" to display offscreen texture", 5.0f, 105.0f, vkx::TextOverlay::alignLeft);
#endif
    }

    void toggleBlur() {
        blur = !blur;
        updateUniformBuffersScene();
        updateDrawCommandBuffers();
    }

    void toggleTextureDisplay() {
        displayTexture = !displayTexture;
        updateDrawCommandBuffers();
    }

};

RUN_EXAMPLE(VulkanExample)
