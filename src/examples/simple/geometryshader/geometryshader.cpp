/*
 * Vulkan Example - Geometry shader (vertex normal debugging)
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/geometryshader/base.frag.inl>
#include <shaders/geometryshader/base.vert.inl>
#include <shaders/geometryshader/mesh.frag.inl>
#include <shaders/geometryshader/mesh.vert.inl>
#include <shaders/geometryshader/normaldebug.geom.inl>

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;
public:
    struct {
        vks::model::Model object;
    } meshes;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVS;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboGS;

    struct {
        vks::Buffer VS;
        vks::Buffer GS;
    } uniformData;

    struct {
        vk::Pipeline solid;
        vk::Pipeline normals;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    bool displayNormals = true;

    VulkanExample() {
        camera.setRotation({ 0.0f, -25.0f, 0.0f });
        camera.translate({ 0.0f, 0.0f, -9.0f });
        title = "Vulkan Example - Geometry shader";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.solid);
        device.destroy(pipelines.normals);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        meshes.object.destroy();

        uniformData.VS.destroy();
        uniformData.GS.destroy();
    }

    void getEnabledFeatures() override {
        Parent::getEnabledFeatures();
        auto& deviceFeatures = deviceInfo.features.core10;
        auto& enabledFeatures = context.enabledFeatures.core10;
        enabledFeatures.geometryShader = deviceFeatures.geometryShader;
        // Geometry shader support is required for this example
        if (!enabledFeatures.geometryShader) {
            throw std::runtime_error("Selected GPU does not support geometry shaders!");
        }
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.setLineWidth(1.0f);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.object.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);
        // Solid shading
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
        // Normal debugging
        if (displayNormals) {
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.normals);
            cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
        }
    }

    void loadAssets() override { meshes.object.loadFromFile(getAssetPath() + "models/suzanne.obj", vertexLayout, 0.25f); }

    void setupDescriptorPool() {
        // Example uses two ubos
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 2 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader ubo
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Geometry shader ubo
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eGeometry },
        };

        descriptorSetLayout =
            device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader shader ubo
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.VS.descriptor },
            // Binding 1 : Geometry shader ubo
            vk::WriteDescriptorSet{ descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.GS.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.dynamicState.dynamicStateEnables = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eLineWidth };

        // Normal debugging pipeline
        pipelineBuilder.loadShader(vkx::shaders::geometryshader::base::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::geometryshader::base::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.loadShader(vkx::shaders::geometryshader::normaldebug::geom, vk::ShaderStageFlagBits::eGeometry);
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelines.normals = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Solid rendering pipeline
        pipelineBuilder.loadShader(vkx::shaders::geometryshader::mesh::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::geometryshader::mesh::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.VS = loader.createUniformBuffer(uboVS);
        // Geometry shader uniform buffer block
        uniformData.GS = loader.createUniformBuffer(uboGS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        uboVS.projection = getProjection();
        uboVS.model = camera.matrices.view;
        uniformData.VS.copy(uboVS);

        // Geometry shader
        uboGS.model = uboVS.model;
        uboGS.projection = uboVS.projection;
        uniformData.GS.copy(uboGS);
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

    void render() override {
        if (!prepared)
            return;
        draw();
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Display normals", &displayNormals)) {
                buildCommandBuffers();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
