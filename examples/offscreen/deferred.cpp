/*
* Vulkan Example - Deferred shading multiple render targets (aka G-vk::Buffer) example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/



#include "vulkanOffscreenExampleBase.hpp"


// Texture properties
#define TEX_DIM 1024

// Vertex layout for this example
std::vector<vkx::VertexLayout> vertexLayout =
{
    vkx::VertexLayout::VERTEX_LAYOUT_POSITION,
    vkx::VertexLayout::VERTEX_LAYOUT_UV,
    vkx::VertexLayout::VERTEX_LAYOUT_COLOR,
    vkx::VertexLayout::VERTEX_LAYOUT_NORMAL
};

class VulkanExample : public vkx::OffscreenExampleBase {
    using Parent = OffscreenExampleBase;
public:
    bool debugDisplay = true;

    struct {
        vkx::Texture colorMap;
    } textures;

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
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
    } uboVS, uboOffscreenVS;

    struct Light {
        glm::vec4 position;
        glm::vec4 color;
        float radius;
        float quadraticFalloff;
        float linearFalloff;
        float _pad;
    };

    struct {
        Light lights[5];
        glm::vec4 viewPos;
    } uboFragmentLights;

    struct {
        vkx::UniformData vsFullScreen;
        vkx::UniformData vsOffscreen;
        vkx::UniformData fsLights;
    } uniformData;

    struct {
        vk::Pipeline deferred;
        vk::Pipeline offscreen;
        vk::Pipeline debug;
    } pipelines;

    struct {
        vk::PipelineLayout deferred;
        vk::PipelineLayout offscreen;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet offscreen;
    } descriptorSets;

    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::CommandBuffer offscreenCmdBuffer;
    
    VulkanExample() : vkx::OffscreenExampleBase(ENABLE_VALIDATION) {
        
        camera.setZoom(-8.0f);
        size.width = 1024;
        size.height = 1024;
        title = "Vulkan Example - Deferred shading";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources 
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroyPipeline(pipelines.deferred);
        device.destroyPipeline(pipelines.offscreen);
        device.destroyPipeline(pipelines.debug);

        device.destroyPipelineLayout(pipelineLayouts.deferred);
        device.destroyPipelineLayout(pipelineLayouts.offscreen);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // Meshes
        meshes.example.destroy();
        meshes.quad.destroy(); 

        // Uniform buffers
        uniformData.vsOffscreen.destroy();
        uniformData.vsFullScreen.destroy();
        uniformData.fsLights.destroy();
        device.freeCommandBuffers(cmdPool, offscreenCmdBuffer);
        textures.colorMap.destroy();
    }


    // Build command buffer for rendering the scene to the offscreen frame buffer 
    // and blitting it to the different texture targets
    void buildOffscreenCommandBuffer() override {
        // Create separate command buffer for offscreen 
        // rendering
        if (!offscreenCmdBuffer) {
            vk::CommandBufferAllocateInfo cmd = vkx::commandBufferAllocateInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1);
            offscreenCmdBuffer = device.allocateCommandBuffers(cmd)[0];
        }

        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        // Clear values for all attachments written in the fragment sahder
        std::array<vk::ClearValue, 4> clearValues;
        clearValues[0].color = vkx::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
        clearValues[1].color = vkx::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
        clearValues[2].color = vkx::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
        clearValues[3].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = offscreen.renderPass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
        renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
        renderPassBeginInfo.clearValueCount = clearValues.size();
        renderPassBeginInfo.pClearValues = clearValues.data();

        offscreenCmdBuffer.begin(cmdBufInfo);
        offscreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        vk::Viewport viewport = vkx::viewport(offscreen.size);
        offscreenCmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor = vkx::rect2D(offscreen.size);
        offscreenCmdBuffer.setScissor(0, scissor);

        offscreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.offscreen, nullptr);
        offscreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);

        vk::DeviceSize offsets = { 0 };
        offscreenCmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.example.vertices.buffer, { 0 });
        offscreenCmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        offscreenCmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
        offscreenCmdBuffer.endRenderPass();
        offscreenCmdBuffer.end();
    }

    void loadTextures() {
        textures.colorMap = textureLoader->loadTexture(
            getAssetPath() + "models/armor/colormap.ktx",
             vk::Format::eBc3UnormBlock);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        vk::Viewport viewport = vkx::viewport(size);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, vkx::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.deferred, 0, descriptorSet, nullptr);
        if (debugDisplay) {
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
            cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.quad.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 1);
            // Move viewport to display final composition in lower right corner
            viewport.x = viewport.width * 0.5f;
            viewport.y = viewport.height * 0.5f;
        } 

        cmdBuffer.setViewport(0, viewport);
        // Final composition as full screen quad
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.deferred);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.quad.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(6, 1, 0, 0, 1);
    }

    void draw() override {
        prepareFrame();
        {
            vk::SubmitInfo submitInfo;
            submitInfo.pWaitDstStageMask = this->submitInfo.pWaitDstStageMask;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &offscreenCmdBuffer;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &semaphores.acquireComplete;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &offscreen.renderComplete;
            queue.submit(submitInfo, VK_NULL_HANDLE);
        } 
        drawCurrentCommandBuffer(offscreen.renderComplete);
        submitFrame();
    }

    void loadMeshes() {
        meshes.example = loadMesh(getAssetPath() + "models/armor/armor.dae", vertexLayout, 1.0f);
    }

    void generateQuads() {
        // Setup vertices for multiple screen aligned quads
        // Used for displaying final result and debug 
        struct Vertex {
            float pos[3];
            float uv[2];
            float col[3];
            float normal[3];
        };

        std::vector<Vertex> vertexBuffer;

        float x = 0.0f;
        float y = 0.0f;
        for (uint32_t i = 0; i < 3; i++) {
            // Last component of normal is used for debug display sampler index
            vertexBuffer.push_back({ { x + 1.0f, y + 1.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            vertexBuffer.push_back({ { x,      y + 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            vertexBuffer.push_back({ { x,      y,      0.0f }, { 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            vertexBuffer.push_back({ { x + 1.0f, y,      0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            x += 1.0f;
            if (x > 1.0f) {
                x = 0.0f;
                y += 1.0f;
            }
        }
        meshes.quad.vertices = stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0,1,2, 2,3,0 };
        for (uint32_t i = 0; i < 3; ++i) {
            uint32_t indices[6] = { 0,1,2, 2,3,0 };
            for (auto index : indices) {
                indexBuffer.push_back(i * 4 + index);
            }
        }
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
        std::vector<vk::DescriptorPoolSize> poolSizes =
        {
            vkx::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 8),
            vkx::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 8)
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo =
            vkx::descriptorPoolCreateInfo(poolSizes.size(), poolSizes.data(), 2);

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        // Deferred shading layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex,
                0),
            // Binding 1 : Position texture target / Scene colormap
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                1),
            // Binding 2 : Normals texture target
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                2),
            // Binding 3 : Albedo texture target
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                3),
            // Binding 4 : Fragment shader uniform buffer
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eFragment,
                4),
        };

        vk::DescriptorSetLayoutCreateInfo descriptorLayout =
            vkx::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), setLayoutBindings.size());

        descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);


        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vkx::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

        pipelineLayouts.deferred = device.createPipelineLayout(pPipelineLayoutCreateInfo);


        // Offscreen (scene) rendering pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout(pPipelineLayoutCreateInfo);

    }

    void setupDescriptorSet() {
        // Textured quad descriptor set
        vk::DescriptorSetAllocateInfo allocInfo =
            vkx::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the offscreen texture targets
        vk::DescriptorImageInfo texDescriptorPosition =
            vkx::descriptorImageInfo(offscreen.framebuffers[0].colors[0].sampler, offscreen.framebuffers[0].colors[0].view, vk::ImageLayout::eGeneral);

        vk::DescriptorImageInfo texDescriptorNormal =
            vkx::descriptorImageInfo(offscreen.framebuffers[0].colors[1].sampler, offscreen.framebuffers[0].colors[1].view, vk::ImageLayout::eGeneral);

        vk::DescriptorImageInfo texDescriptorAlbedo =
            vkx::descriptorImageInfo(offscreen.framebuffers[0].colors[2].sampler, offscreen.framebuffers[0].colors[2].view, vk::ImageLayout::eGeneral);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::writeDescriptorSet(
            descriptorSet,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.vsFullScreen.descriptor),
            // Binding 1 : Position texture target
            vkx::writeDescriptorSet(
                descriptorSet,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                &texDescriptorPosition),
            // Binding 2 : Normals texture target
            vkx::writeDescriptorSet(
                descriptorSet,
                vk::DescriptorType::eCombinedImageSampler,
                2,
                &texDescriptorNormal),
            // Binding 3 : Albedo texture target
            vkx::writeDescriptorSet(
                descriptorSet,
                vk::DescriptorType::eCombinedImageSampler,
                3,
                &texDescriptorAlbedo),
            // Binding 4 : Fragment shader uniform buffer
            vkx::writeDescriptorSet(
                descriptorSet,
                vk::DescriptorType::eUniformBuffer,
                4,
                &uniformData.fsLights.descriptor),
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Offscreen (scene)
        descriptorSets.offscreen = device.allocateDescriptorSets(allocInfo)[0];

        vk::DescriptorImageInfo texDescriptorSceneColormap =
            vkx::descriptorImageInfo(textures.colorMap.sampler, textures.colorMap.view, vk::ImageLayout::eGeneral);

        std::vector<vk::WriteDescriptorSet> offscreenWriteDescriptorSets =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::writeDescriptorSet(
                descriptorSets.offscreen,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.vsOffscreen.descriptor),
            // Binding 1 : Scene color map
            vkx::writeDescriptorSet(
                descriptorSets.offscreen,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                &texDescriptorSceneColormap)
        };
        device.updateDescriptorSets(offscreenWriteDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vkx::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList);

        vk::PipelineRasterizationStateCreateInfo rasterizationState =
            vkx::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);

        vk::PipelineColorBlendAttachmentState blendAttachmentState =
            vkx::pipelineColorBlendAttachmentState();

        vk::PipelineColorBlendStateCreateInfo colorBlendState =
            vkx::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        vk::PipelineDepthStencilStateCreateInfo depthStencilState =
            vkx::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

        vk::PipelineViewportStateCreateInfo viewportState =
            vkx::pipelineViewportStateCreateInfo(1, 1);

        vk::PipelineMultisampleStateCreateInfo multisampleState;

        std::vector<vk::DynamicState> dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState;
        dynamicState.dynamicStateCount = dynamicStateEnables.size();
        dynamicState.pDynamicStates = dynamicStateEnables.data();

        // Final fullscreen pass pipeline
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = loadShader(getAssetPath() + "shaders/deferred/deferred.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/deferred/deferred.frag.spv", vk::ShaderStageFlagBits::eFragment);

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo = vkx::pipelineCreateInfo(pipelineLayouts.deferred, renderPass);
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

        pipelines.deferred = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];


        // Debug display pipeline
        shaderStages[0] = loadShader(getAssetPath() + "shaders/deferred/debug.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/deferred/debug.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.debug = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];


        // Offscreen pipeline
        shaderStages[0] = loadShader(getAssetPath() + "shaders/deferred/mrt.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/deferred/mrt.frag.spv", vk::ShaderStageFlagBits::eFragment);

        // Separate render pass
        pipelineCreateInfo.renderPass = offscreen.renderPass;

        // Separate layout
        pipelineCreateInfo.layout = pipelineLayouts.offscreen;

        // Blend attachment states required for all color attachments
        // This is important, as color write mask will otherwise be 0x0 and you
        // won't see anything rendered to the attachment
        std::array<vk::PipelineColorBlendAttachmentState, 3> blendAttachmentStates = {
            vkx::pipelineColorBlendAttachmentState(),
            vkx::pipelineColorBlendAttachmentState(),
            vkx::pipelineColorBlendAttachmentState()
        };

        colorBlendState.attachmentCount = blendAttachmentStates.size();
        colorBlendState.pAttachments = blendAttachmentStates.data();

        pipelines.offscreen = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Fullscreen vertex shader
        uniformData.vsFullScreen = createUniformBuffer(uboVS);
        // Deferred vertex shader
        uniformData.vsOffscreen = createUniformBuffer(uboOffscreenVS);
        // Deferred fragment shader
        uniformData.fsLights = createUniformBuffer(uboFragmentLights);

        // Update
        updateUniformBuffersScreen();
        updateUniformBufferDeferredMatrices();
        updateUniformBufferDeferredLights();
    }

    void updateUniformBuffersScreen() {
        if (debugDisplay) {
            uboVS.projection = glm::ortho(0.0f, 2.0f, 0.0f, 2.0f, -1.0f, 1.0f);
        } else {
            uboVS.projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        }
        uboVS.model = glm::mat4();
        uniformData.vsFullScreen.copy(uboVS);
    }

    void updateUniformBufferDeferredMatrices() {
        uboOffscreenVS.projection = camera.matrices.perspective;
        uboOffscreenVS.view = camera.matrices.view;
        uboOffscreenVS.model = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.25f, 0.0f));
        uniformData.vsOffscreen.copy(uboOffscreenVS);
    }

    // Update fragment shader light position uniform block
    void updateUniformBufferDeferredLights() {
        // White light from above
        uboFragmentLights.lights[0].position = glm::vec4(0.0f, 3.0f, 1.0f, 0.0f);
        uboFragmentLights.lights[0].color = glm::vec4(1.5f);
        uboFragmentLights.lights[0].radius = 15.0f;
        uboFragmentLights.lights[0].linearFalloff = 0.3f;
        uboFragmentLights.lights[0].quadraticFalloff = 0.4f;
        // Red light
        uboFragmentLights.lights[1].position = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[1].color = glm::vec4(1.5f, 0.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[1].radius = 15.0f;
        uboFragmentLights.lights[1].linearFalloff = 0.4f;
        uboFragmentLights.lights[1].quadraticFalloff = 0.3f;
        // Blue light
        uboFragmentLights.lights[2].position = glm::vec4(2.0f, 1.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[2].color = glm::vec4(0.0f, 0.0f, 2.5f, 0.0f);
        uboFragmentLights.lights[2].radius = 10.0f;
        uboFragmentLights.lights[2].linearFalloff = 0.45f;
        uboFragmentLights.lights[2].quadraticFalloff = 0.35f;
        // Belt glow
        uboFragmentLights.lights[3].position = glm::vec4(0.0f, 0.7f, 0.5f, 0.0f);
        uboFragmentLights.lights[3].color = glm::vec4(2.5f, 2.5f, 0.0f, 0.0f);
        uboFragmentLights.lights[3].radius = 5.0f;
        uboFragmentLights.lights[3].linearFalloff = 8.0f;
        uboFragmentLights.lights[3].quadraticFalloff = 6.0f;
        // Green light
        uboFragmentLights.lights[4].position = glm::vec4(3.0f, 2.0f, 1.0f, 0.0f);
        uboFragmentLights.lights[4].color = glm::vec4(0.0f, 1.5f, 0.0f, 0.0f);
        uboFragmentLights.lights[4].radius = 10.0f;
        uboFragmentLights.lights[4].linearFalloff = 0.8f;
        uboFragmentLights.lights[4].quadraticFalloff = 0.6f;

        // Current view position
        uboFragmentLights.viewPos = glm::vec4(0.0f, 0.0f, -camera.position.z, 0.0f);

        uniformData.fsLights.copy(uboFragmentLights);
    }


    void prepare() override {
        offscreen.size = glm::uvec2(TEX_DIM);
        offscreen.colorFormats = std::vector<vk::Format>{ {
            vk::Format::eR16G16B16A16Sfloat,
            vk::Format::eR16G16B16A16Sfloat,
            vk::Format::eR8G8B8A8Unorm
        } };
        Parent::prepare();
        loadTextures();
        generateQuads();
        loadMeshes();
        setupVertexDescriptions();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        updateDrawCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void viewChanged() override {
        updateUniformBufferDeferredMatrices();
    }

    void toggleDebugDisplay() {
        debugDisplay = !debugDisplay;
        updateDrawCommandBuffers();
        buildOffscreenCommandBuffer();
        updateUniformBuffersScreen();
    }

    void keyPressed(uint32_t key) override {
        Parent::keyPressed(key);
        switch (key) {
        case GLFW_KEY_D:
            toggleDebugDisplay();
            break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)

