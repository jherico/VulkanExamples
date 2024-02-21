/*
 * Vulkan Example - Using different pipelines in one single renderpass
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/pipelines/phong.frag.inl>
#include <shaders/pipelines/phong.vert.inl>
#include <shaders/pipelines/toon.frag.inl>
#include <shaders/pipelines/toon.vert.inl>
#include <shaders/pipelines/wireframe.frag.inl>
#include <shaders/pipelines/wireframe.vert.inl>

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

static vk::PhysicalDeviceFeatures features = [] {
    vk::PhysicalDeviceFeatures features;
    features.wideLines = VK_TRUE;
    return features;
}();

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    struct {
        vks::model::Model cube;
    } meshes;

    vks::Buffer uniformDataVS;

    // Same uniform buffer layout as shader
    struct UboVS {
        glm::mat4 projection;
        glm::mat4 modelView;
        glm::vec4 lightPos = glm::vec4(0.0f, 2.0f, 1.0f, 0.0f);
    } uboVS;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    struct {
        vk::Pipeline phong;
        vk::Pipeline wireframe;
        vk::Pipeline toon;
    } pipelines;

    VulkanExample() {
        camera.dolly(-10.5f);
        camera.setRotation({ -25.0f, 15.0f, 0.0f });
        title = "Vulkan Example - vk::Pipeline state objects";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.phong);
        if (deviceInfo.features.core10.fillModeNonSolid) {
            device.destroy(pipelines.wireframe);
        }
        device.destroy(pipelines.toon);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        meshes.cube.destroy();
        uniformDataVS.destroy();
    }

    void getEnabledFeatures() { 
        Parent::getEnabledFeatures();
        context.enabledFeatures.core10.fillModeNonSolid = deviceInfo.features.core10.fillModeNonSolid;
        context.enabledFeatures.core10.wideLines = deviceInfo.features.core10.wideLines;
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        const auto& features = deviceInfo.features.core10;
        const auto& limits = deviceInfo.properties.core10.limits;
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.cube.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.cube.indices.buffer, 0, vk::IndexType::eUint32);

        // Left : Solid colored
        vk::Viewport viewport = vks::util::viewport((float)size.width / 3, (float)size.height, 0.0f, 1.0f);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phong);

        cmdBuffer.drawIndexed(meshes.cube.indexCount, 1, 0, 0, 0);

        // Center : Toon
        viewport.x += viewport.width;
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.toon);
        cmdBuffer.setLineWidth(2.0f);
        cmdBuffer.drawIndexed(meshes.cube.indexCount, 1, 0, 0, 0);

        auto lineWidthGranularity = limits.lineWidthGranularity;
        auto lineWidthRange = limits.lineWidthRange;

        if (features.fillModeNonSolid) {
            // Right : Wireframe
            viewport.x += viewport.width;
            cmdBuffer.setViewport(0, viewport);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.wireframe);
            cmdBuffer.drawIndexed(meshes.cube.indexCount, 1, 0, 0, 0);
        }
    }

    void loadAssets() override { meshes.cube.loadFromFile(getAssetPath() + "models/treasure_smooth.dae", vertexLayout, 1.0f); }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 1 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout };
        pipelineCreator.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineCreator.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineCreator.rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
        pipelineCreator.depthStencilState = true;
        pipelineCreator.dynamicState.dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eLineWidth,
        };

        // Phong shading pipeline
        vertexLayout.appendVertexLayout(pipelineCreator.vertexInputState);
        pipelineCreator.loadShader(vkx::shaders::pipelines::phong::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::pipelines::phong::frag, vk::ShaderStageFlagBits::eFragment);

        // We are using this pipeline as the base for the other pipelines (derivatives)
        // vk::Pipeline derivatives can be used for pipelines that share most of their state
        // Depending on the implementation this may result in better performance for pipeline
        // switchting and faster creation time
        pipelineCreator.pipelineCreateInfo.flags = vk::PipelineCreateFlagBits::eAllowDerivatives;

        // Textured pipeline
        pipelines.phong = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();

        // All pipelines created after the base pipeline will be derivatives
        pipelineCreator.pipelineCreateInfo.flags = vk::PipelineCreateFlagBits::eDerivative;
        // Base pipeline will be our first created pipeline
        pipelineCreator.pipelineCreateInfo.basePipelineHandle = pipelines.phong;
        // It's only allowed to either use a handle or index for the base pipeline
        // As we use the handle, we must set the index to -1 (see section 9.5 of the specification)
        pipelineCreator.pipelineCreateInfo.basePipelineIndex = -1;

        // Toon shading pipeline
        pipelineCreator.loadShader(vkx::shaders::pipelines::toon::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::pipelines::toon::frag, vk::ShaderStageFlagBits::eFragment);

        pipelines.toon = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();

        // Non solid rendering is not a mandatory Vulkan feature
        if (deviceInfo.features.core10.fillModeNonSolid) {
            // vk::Pipeline for wire frame rendering
            pipelineCreator.rasterizationState.polygonMode = vk::PolygonMode::eLine;
            pipelineCreator.loadShader(vkx::shaders::pipelines::wireframe::vert, vk::ShaderStageFlagBits::eVertex);
            pipelineCreator.loadShader(vkx::shaders::pipelines::wireframe::frag, vk::ShaderStageFlagBits::eFragment);
            pipelines.wireframe = pipelineCreator.create(context.pipelineCache);
            pipelineCreator.destroyShaderModules();
        }
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Create the vertex shader uniform buffer block
        uniformDataVS = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = glm::perspective(glm::radians(60.0f), (float)(size.width / 3.0f) / (float)size.height, 0.001f, 256.0f);
        uboVS.modelView = camera.matrices.view;
        uniformDataVS.copy(uboVS);
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
};

RUN_EXAMPLE(VulkanExample)
