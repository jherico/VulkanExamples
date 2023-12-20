/*
* Vulkan Example - Offscreen rendering using a separate framebuffer
*
*    p - Toggle light source animation
*    l - Toggle between scene and light's POV
*    s - Toggle shadowmap display
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanOffscreenExampleBase.hpp>

// Texture properties

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
    bool displayShadowMap = false;
    bool lightPOV = false;

    // Keep depth range as small as possible
    // for better shadow map precision
    float zNear = 1.0f;
    float zFar = 96.0f;

    // Constant depth bias factor (always applied)
    float depthBiasConstant = 1.25f;
    // Slope depth bias factor, applied depending on polygon's slope
    float depthBiasSlope = 1.75f;

    glm::vec3 lightPos = glm::vec3();
    float lightFOV = 45.0f;

    struct {
        vks::model::Model scene;
        vks::model::Model quad;
    } meshes;

    vks::Buffer uniformDataVS, uniformDataOffscreenVS;

    struct {
        vks::Buffer scene;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVSquad;

    struct {
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
        glm::mat4 depthBiasMVP;
        glm::vec3 lightPos;
    } uboVSscene;

    struct {
        glm::mat4 depthMVP;
    } uboOffscreenVS;

    struct {
        vk::Pipeline quad;
        vk::Pipeline offscreen;
        vk::Pipeline scene;
    } pipelines;

    struct {
        vk::PipelineLayout quad;
        vk::PipelineLayout offscreen;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet offscreen;
        vk::DescriptorSet scene;
    } descriptorSets;

    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        enableVsync = true;
        camera.type = Camera::lookat;
        camera.setRotation({ -15.0f, -390.0f, 0.0f });
        camera.dolly(-10.0f);
        title = "Vulkan Example - Projected shadow mapping";
        timerSpeed *= 0.5f;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroyPipeline(pipelines.quad);
        device.destroyPipeline(pipelines.offscreen);
        device.destroyPipeline(pipelines.scene);

        device.destroyPipelineLayout(pipelineLayouts.quad);
        device.destroyPipelineLayout(pipelineLayouts.offscreen);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // Meshes
        meshes.scene.destroy();
        meshes.quad.destroy();

        // Uniform buffers
        uniformDataVS.destroy();
        uniformDataOffscreenVS.destroy();
    }

    void buildOffscreenCommandBuffer() override {
        // Create separate command buffer for offscreen
        // rendering
        if (offscreen.cmdBuffer) {
            std::vector<vk::CommandBuffer> buffers{ { offscreen.cmdBuffer } };
            context.trashCommandBuffers(context.getCommandPool(), buffers);
        }
        offscreen.cmdBuffer = device.allocateCommandBuffers({ cmdPool, vk::CommandBufferLevel::ePrimary, 1 })[0];

        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
        offscreen.cmdBuffer.begin(cmdBufInfo);

        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor();
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = offscreen.renderPass;
        renderPassBeginInfo.framebuffer = offscreen.framebuffers[0].framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreen.size.x;
        renderPassBeginInfo.renderArea.extent.height = offscreen.size.y;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        // Set depth bias (aka "Polygon offset")
        offscreen.cmdBuffer.setDepthBias(depthBiasConstant, 0.0f, depthBiasSlope);
        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.offscreen, nullptr);
        offscreen.cmdBuffer.bindVertexBuffers(0, meshes.scene.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(meshes.scene.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(meshes.scene.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();
        offscreen.cmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.quad, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.quad);

        // Visualize shadow map
        if (displayShadowMap) {
            cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
        }

        // 3D scene
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.quad, 0, descriptorSets.scene, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.scene);

        cmdBuffer.bindVertexBuffers(0, meshes.scene.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.scene.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.scene.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override { meshes.scene.loadFromFile(context, getAssetPath() + "models/vulkanscene_shadow.dae", vertexLayout, 4.0f); }

    void generateQuad() {
        // Setup vertices for a single uv-mapped quad
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
        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        // Example uses three ubos and two image samplers
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 6 },
            { vk::DescriptorType::eCombinedImageSampler, 4 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        // Textured quad pipeline layout
        pipelineLayouts.quad = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        // Offscreen pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSets() {
        // Textured quad descriptor set
        vk::DescriptorSetAllocateInfo allocInfo{
            descriptorPool,
            1,
            &descriptorSetLayout,
        };

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the shadow map texture
        vk::DescriptorImageInfo texDescriptor{ offscreen.framebuffers[0].depth.sampler, offscreen.framebuffers[0].depth.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor },
            // Binding 1 : Fragment shader texture sampler
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Offscreen
        descriptorSets.offscreen = device.allocateDescriptorSets(allocInfo)[0];

        std::vector<vk::WriteDescriptorSet> offscreenWriteDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.offscreen, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataOffscreenVS.descriptor },
        };
        device.updateDescriptorSets(offscreenWriteDescriptorSets, nullptr);

        // 3D scene
        descriptorSets.scene = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the shadow map texture
        texDescriptor.sampler = offscreen.framebuffers[0].depth.sampler;
        texDescriptor.imageView = offscreen.framebuffers[0].depth.view;

        std::vector<vk::WriteDescriptorSet> sceneDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.scene.descriptor },
            // Binding 1 : Fragment shader shadow sampler
            { descriptorSets.scene, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };
        device.updateDescriptorSets(sceneDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Solid rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayouts.quad, renderPass };
        pipelineCreator.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineCreator.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineCreator.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineCreator.loadShader(getAssetPath() + "shaders/shadowmapping/quad.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/shadowmapping/quad.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.quad = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();

        // 3D scene
        pipelineCreator.loadShader(getAssetPath() + "shaders/shadowmapping/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/shadowmapping/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.scene = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();

        // Offscreen pipeline
        pipelineCreator.loadShader(getAssetPath() + "shaders/shadowmapping/offscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/shadowmapping/offscreen.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineCreator.layout = pipelineLayouts.offscreen;
        pipelineCreator.renderPass = offscreen.renderPass;
        pipelineCreator.depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
        pipelineCreator.rasterizationState.depthBiasEnable = VK_TRUE;
        // Add depth bias to dynamic state, so we can change it at runtime
        pipelineCreator.dynamicState.dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
        pipelines.offscreen = pipelineCreator.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Debug quad vertex shader uniform buffer block
        uniformDataVS = context.createUniformBuffer(uboVSscene);
        // Offsvreen vertex shader uniform buffer block
        uniformDataOffscreenVS = context.createUniformBuffer(uboOffscreenVS);
        // Scene vertex shader uniform buffer block
        uniformData.scene = context.createUniformBuffer(uboVSscene);

        updateLight();
        updateUniformBufferOffscreen();
        updateUniformBuffers();
    }

    void updateLight() {
        // Animate the light source
        lightPos.x = cos(glm::radians(timer * 360.0f)) * 40.0f;
        lightPos.y = -50.0f + sin(glm::radians(timer * 360.0f)) * 20.0f;
        lightPos.z = 25.0f + sin(glm::radians(timer * 360.0f)) * 5.0f;
    }

    void updateUniformBuffers() {
        // Shadow map debug quad
        float AR = (float)size.height / (float)size.width;

        uboVSquad.projection = glm::ortho(2.5f / AR, 0.0f, 0.0f, 2.5f, -1.0f, 1.0f);
        uboVSquad.model = glm::mat4();

        uniformDataVS.copy(uboVSquad);

        // 3D scene
        uboVSscene.projection = glm::perspective(glm::radians(45.0f), (float)size.width / (float)size.height, zNear, zFar);

        uboVSscene.view = camera.matrices.view;
        uboVSscene.model = glm::mat4();
        uboVSscene.lightPos = lightPos;
        // Render scene from light's point of view
        if (lightPOV) {
            uboVSscene.projection = glm::perspective(glm::radians(lightFOV), (float)size.width / (float)size.height, zNear, zFar);
            uboVSscene.view = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0, 1, 0));
        }
        uboVSscene.depthBiasMVP = uboOffscreenVS.depthMVP;
        uniformData.scene.copy(uboVSscene);
    }

    void updateUniformBufferOffscreen() {
        // Matrix from light's point of view
        glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
        glm::mat4 depthViewMatrix = glm::lookAt(lightPos, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        glm::mat4 depthModelMatrix = glm::mat4();

        uboOffscreenVS.depthMVP = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;
        uniformDataOffscreenVS.copy(uboOffscreenVS);
    }

#define TEX_FILTER vk::Filter::eLinear

    void prepare() override {
        offscreen.size = glm::uvec2(2048);
        offscreen.depthFormat = vk::Format::eD16Unorm;
        offscreen.depthFinalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        offscreen.colorFinalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        offscreen.attachmentUsage = vk::ImageUsageFlags();
        offscreen.depthAttachmentUsage = vk::ImageUsageFlagBits::eSampled;
        OffscreenExampleBase::prepare();
        generateQuad();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void update(float deltaTime) override {
        Parent::update(deltaTime);
        if (!paused) {
            updateLight();
            updateUniformBufferOffscreen();
            updateUniformBuffers();
        }
    }

    void viewChanged() override {
        updateUniformBufferOffscreen();
        updateUniformBuffers();
    }

    void toggleShadowMapDisplay() {
        displayShadowMap = !displayShadowMap;
        buildCommandBuffers();
    }

    void toogleLightPOV() {
        lightPOV = !lightPOV;
        viewChanged();
    }

    void keyPressed(uint32_t key) override {
        switch (key) {
            case KEY_S:
                toggleShadowMapDisplay();
                break;
            case KEY_L:
                toogleLightPOV();
                break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
