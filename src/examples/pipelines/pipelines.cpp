/*
* Vulkan Example - Using different pipelines in one single renderpass
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

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
        device.destroyPipeline(pipelines.phong);
        if (context.deviceFeatures.fillModeNonSolid) {
            device.destroyPipeline(pipelines.wireframe);
        }
        device.destroyPipeline(pipelines.toon);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        meshes.cube.destroy();

        device.destroyBuffer(uniformDataVS.buffer);
        device.freeMemory(uniformDataVS.memory);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
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

        auto lineWidthGranularity = context.deviceProperties.limits.lineWidthGranularity;
        auto lineWidthRange = context.deviceProperties.limits.lineWidthRange;

        if (context.deviceFeatures.fillModeNonSolid) {
            // Right : Wireframe
            viewport.x += viewport.width;
            cmdBuffer.setViewport(0, viewport);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.wireframe);
            cmdBuffer.drawIndexed(meshes.cube.indexCount, 1, 0, 0, 0);
        }
    }

    void loadAssets() override { meshes.cube.loadFromFile(context, getAssetPath() + "models/treasure_smooth.dae", vertexLayout, 1.0f); }

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
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout, renderPass };

        pipelineCreator.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineCreator.dynamicState.dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eLineWidth,
        };

        // Phong shading pipeline
        pipelineCreator.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineCreator.loadShader(getAssetPath() + "shaders/pipelines/phong.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/pipelines/phong.frag.spv", vk::ShaderStageFlagBits::eFragment);

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
        pipelineCreator.loadShader(getAssetPath() + "shaders/pipelines/toon.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/pipelines/toon.frag.spv", vk::ShaderStageFlagBits::eFragment);

        pipelines.toon = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();

        // Non solid rendering is not a mandatory Vulkan feature
        if (context.deviceFeatures.fillModeNonSolid) {
            // vk::Pipeline for wire frame rendering
            pipelineCreator.rasterizationState.polygonMode = vk::PolygonMode::eLine;
            pipelineCreator.loadShader(getAssetPath() + "shaders/pipelines/wireframe.vert.spv", vk::ShaderStageFlagBits::eVertex);
            pipelineCreator.loadShader(getAssetPath() + "shaders/pipelines/wireframe.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipelines.wireframe = pipelineCreator.create(context.pipelineCache);
            pipelineCreator.destroyShaderModules();
        }
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Create the vertex shader uniform buffer block
        uniformDataVS = context.createUniformBuffer(uboVS);
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
