/*
* Vulkan Example - Deferred shading multiple render targets (aka G-vk::Buffer) example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanOffscreenExampleBase.hpp>
#include <vks/model.hpp>

// Texture properties
#define TEX_DIM 1024

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
    bool debugDisplay = true;

    struct {
        vks::texture::Texture2D colorMap;
    } textures;

    struct {
        vks::model::Model example;
        vks::model::Model quad;
    } meshes;

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
        vks::Buffer vsFullScreen;
        vks::Buffer vsOffscreen;
        vks::Buffer fsLights;
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

    VulkanExample() {
        camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
        camera.rotationSpeed = 0.25f;
#endif
        camera.position = { 2.15f, 0.3f, -8.0f };
        camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
        camera.setPerspective(60.0f, size, 0.1f, 256.0f);
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
        textures.colorMap.destroy();
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer
    // and blitting it to the different texture targets
    void buildOffscreenCommandBuffer() override {
        // Create separate command buffer for offscreen rendering
        if (!offscreen.cmdBuffer) {
            offscreen.cmdBuffer = device.allocateCommandBuffers({ cmdPool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        }

        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        // Clear values for all attachments written in the fragment sahder
        std::array<vk::ClearValue, 4> clearValues;
        clearValues[0].color = vks::util::clearColor();
        clearValues[1].color = vks::util::clearColor();
        clearValues[2].color = vks::util::clearColor();
        clearValues[3].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = offscreen.renderPass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
        renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
        renderPassBeginInfo.clearValueCount = (uint32_t)clearValues.size();
        renderPassBeginInfo.pClearValues = clearValues.data();

        offscreen.cmdBuffer.begin(cmdBufInfo);
        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        vk::Viewport viewport = vks::util::viewport(offscreen.size);
        offscreen.cmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor = vks::util::rect2D(offscreen.size);
        offscreen.cmdBuffer.setScissor(0, scissor);

        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.offscreen, nullptr);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);

        vk::DeviceSize offsets = { 0 };
        offscreen.cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();
        offscreen.cmdBuffer.end();
    }

    void loadAssets() override {
        textures.colorMap.loadFromFile(context, getAssetPath() + "models/armor/colormap.ktx", vk::Format::eBc3UnormBlock);
        meshes.example.loadFromFile(context, getAssetPath() + "models/armor/armor.dae", vertexLayout, 1.0f);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::Viewport viewport = vks::util::viewport(size);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.deferred, 0, descriptorSet, nullptr);
        if (debugDisplay) {
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
            cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 1);
            // Move viewport to display final composition in lower right corner
            viewport.x = viewport.width * 0.5f;
            viewport.y = viewport.height * 0.5f;
        }

        cmdBuffer.setViewport(0, viewport);
        // Final composition as full screen quad
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.deferred);
        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(6, 1, 0, 0, 1);
    }

    void draw() override {
        prepareFrame();
        if (offscreen.active) {
            context.submit(offscreen.cmdBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreen.renderComplete);
            renderWaitSemaphores = { offscreen.renderComplete };
        } else {
            renderWaitSemaphores = { semaphores.acquireComplete };
        }
        drawCurrentCommandBuffer();
        submitFrame();
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
            vertexBuffer.push_back({ { x, y + 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            vertexBuffer.push_back({ { x, y, 0.0f }, { 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            vertexBuffer.push_back({ { x + 1.0f, y, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, (float)i } });
            x += 1.0f;
            if (x > 1.0f) {
                x = 0.0f;
                y += 1.0f;
            }
        }
        meshes.quad.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        for (uint32_t i = 0; i < 3; ++i) {
            uint32_t indices[6] = { 0, 1, 2, 2, 3, 0 };
            for (auto index : indices) {
                indexBuffer.push_back(i * 4 + index);
            }
        }
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 8 },
            { vk::DescriptorType::eCombinedImageSampler, 8 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Deferred shading layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Position texture target / Scene colormap
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 2 : Normals texture target
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 3 : Albedo texture target
            { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 4 : Fragment shader uniform buffer
            { 4, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.deferred = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        // Offscreen (scene) rendering pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        // Textured quad descriptor set
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the offscreen texture targets
        vk::DescriptorImageInfo texDescriptorPosition{ offscreen.framebuffers[0].colors[0].sampler, offscreen.framebuffers[0].colors[0].view,
                                                       vk::ImageLayout::eGeneral };

        vk::DescriptorImageInfo texDescriptorNormal{ offscreen.framebuffers[0].colors[1].sampler, offscreen.framebuffers[0].colors[1].view,
                                                     vk::ImageLayout::eGeneral };

        vk::DescriptorImageInfo texDescriptorAlbedo{ offscreen.framebuffers[0].colors[2].sampler, offscreen.framebuffers[0].colors[2].view,
                                                     vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsFullScreen.descriptor },
            // Binding 1 : Position texture target
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorPosition },
            // Binding 2 : Normals texture target
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorNormal },
            // Binding 3 : Albedo texture target
            { descriptorSet, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorAlbedo },
            // Binding 4 : Fragment shader uniform buffer
            { descriptorSet, 4, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.fsLights.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Offscreen (scene)
        descriptorSets.offscreen = device.allocateDescriptorSets(allocInfo)[0];

        vk::DescriptorImageInfo texDescriptorSceneColormap{ textures.colorMap.sampler, textures.colorMap.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> offscreenWriteDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.offscreen, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsOffscreen.descriptor },
            // Binding 1 : Scene color map
            { descriptorSets.offscreen, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorSceneColormap },
        };
        device.updateDescriptorSets(offscreenWriteDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.deferred, renderPass };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);

        // Final fullscreen pass pipeline
        pipelineBuilder.loadShader(getAssetPath() + "shaders/deferred/deferred.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/deferred/deferred.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.deferred = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Debug display pipeline
        pipelineBuilder.loadShader(getAssetPath() + "shaders/deferred/debug.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/deferred/debug.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.debug = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Offscreen pipeline
        pipelineBuilder.loadShader(getAssetPath() + "shaders/deferred/mrt.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/deferred/mrt.frag.spv", vk::ShaderStageFlagBits::eFragment);

        // Separate layout & render pass
        pipelineBuilder.renderPass = offscreen.renderPass;
        pipelineBuilder.layout = pipelineLayouts.offscreen;
        // Blend attachment states required for all color attachments
        // This is important, as color write mask will otherwise be 0x0 and you
        // won't see anything rendered to the attachment
        pipelineBuilder.colorBlendState.blendAttachmentStates = {
            {},
            {},
            {},
        };
        pipelines.offscreen = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Fullscreen vertex shader
        uniformData.vsFullScreen = context.createUniformBuffer(uboVS);
        // Deferred vertex shader
        uniformData.vsOffscreen = context.createUniformBuffer(uboOffscreenVS);
        // Deferred fragment shader
        uniformData.fsLights = context.createUniformBuffer(uboFragmentLights);

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
        offscreen.colorFormats = std::vector<vk::Format>{ { vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm } };
        Parent::prepare();
        generateQuads();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void viewChanged() override { updateUniformBufferDeferredMatrices(); }

    void toggleDebugDisplay() {
        debugDisplay = !debugDisplay;
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        updateUniformBuffersScreen();
    }

    void keyPressed(uint32_t key) override {
        Parent::keyPressed(key);
        switch (key) {
            case KEY_D:
                toggleDebugDisplay();
                break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
