/*
* Vulkan Example - Rendering outlines using the stencil buffer
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase {
public:
    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_COLOR,
        vks::model::VERTEX_COMPONENT_NORMAL,
    });

    vks::model::Model model;

    struct UBO {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0f, -2.0f, 1.0f, 0.0f);
        // Vertex shader extrudes model by this value along normals for outlining
        float outlineWidth = 0.05f;
    } uboVS;

    vks::Buffer uniformBufferVS;

    struct {
        vk::Pipeline stencil;
        vk::Pipeline outline;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        title = "Stencil buffer outlines";
        timerSpeed *= 0.25f;
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(2.5f, -35.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.08f, 3.6f, -8.4f));
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroy(pipelines.stencil);
        device.destroy(pipelines.outline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        model.destroy();
        uniformBufferVS.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        vk::Viewport viewport{ 0, 0, (float)size.width, (float)size.height, 0.0f, 1.0f };
        drawCmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{ { 0, 0 }, size };
        drawCmdBuffer.setScissor(0, scissor);

        drawCmdBuffer.bindVertexBuffers(0, model.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(model.indices.buffer, 0, vk::IndexType::eUint32);

        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

        // First pass renders object (toon shaded) and fills stencil buffer
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.stencil);
        drawCmdBuffer.drawIndexed(model.indexCount, 1, 0, 0, 0);

        // Second pass renders scaled object only where stencil was not set by first pass
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.outline);
        drawCmdBuffer.drawIndexed(model.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override { model.loadFromFile(context, getAssetPath() + "models/venus.fbx", vertexLayout, 0.3f); }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Deferred shading layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferVS.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        auto& depthStencilState = builder.depthStencilState;
        depthStencilState.stencilTestEnable = VK_TRUE;
        depthStencilState.back.compareOp = vk::CompareOp::eAlways;
        depthStencilState.back.failOp = vk::StencilOp::eReplace;
        depthStencilState.back.depthFailOp = vk::StencilOp::eReplace;
        depthStencilState.back.passOp = vk::StencilOp::eReplace;
        depthStencilState.back.compareMask = 0xff;
        depthStencilState.back.writeMask = 0xff;
        depthStencilState.back.reference = 1;
        depthStencilState.front = depthStencilState.back;
        // Vertex bindings and attributes
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        // Toon render and stencil fill pass
        builder.loadShader(getAssetPath() + "shaders/stencilbuffer/toon.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/stencilbuffer/toon.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.stencil = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Outline pass
        depthStencilState.back.compareOp = vk::CompareOp::eNotEqual;
        depthStencilState.back.failOp = vk::StencilOp::eKeep;
        depthStencilState.back.depthFailOp = vk::StencilOp::eKeep;
        depthStencilState.back.passOp = vk::StencilOp::eReplace;
        depthStencilState.front = depthStencilState.back;
        depthStencilState.depthTestEnable = VK_FALSE;
        builder.loadShader(getAssetPath() + "shaders/stencilbuffer/outline.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/stencilbuffer/outline.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.outline = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Mesh vertex shader uniform buffer block
        uniformBufferVS = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.inputFloat("Outline width", &uboVS.outlineWidth, 0.05f, "%.2f")) {
                updateUniformBuffers();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
