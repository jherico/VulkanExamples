/*
 * Vulkan Example - Animated gears using multiple uniform buffers
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <rendering/vulkanGear.h>

#include <shaders/gears/gears.vert.inl>
#include <shaders/gears/gears.frag.inl>

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    struct {
        vk::Pipeline solid;
    } pipelines;

    std::vector<VulkanGear> gears;
    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_COLOR,
    } };

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        timerSpeed *= 0.25f;
        camera.translate(glm::vec3(0.0f, 0.0f, -12.0f * zoomSpeed));
        title = "Vulkan Example - Gears";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.solid);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        gears.clear();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        for (auto& gear : gears) {
            gear.draw(cmdBuffer, pipelineLayout);
        }
    }

    void prepareVertices() {
        // Gear definitions
        std::vector<float> innerRadiuses = { 1.0f, 0.5f, 1.3f };
        std::vector<float> outerRadiuses = { 4.0f, 2.0f, 2.0f };
        std::vector<float> widths = { 1.0f, 2.0f, 0.5f };
        std::vector<int32_t> toothCount = { 20, 10, 10 };
        std::vector<float> toothDepth = { 0.7f, 0.7f, 0.7f };
        std::vector<glm::vec3> colors = { glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.2f), glm::vec3(0.0f, 0.0f, 1.0f) };
        std::vector<glm::vec3> positions = { glm::vec3(-3.0, 0.0, 0.0), glm::vec3(3.1, 0.0, 0.0), glm::vec3(-3.1, -6.2, 0.0) };
        std::vector<float> rotationSpeeds = { 1.0f, -2.0f, -2.0f };
        std::vector<float> rotationStarts = { 0.0f, -9.0f, -30.0f };

        gears.resize(positions.size());
        for (int32_t i = 0; i < gears.size(); ++i) {
            gears[i].generate(innerRadiuses[i], outerRadiuses[i], widths[i], toothCount[i], toothDepth[i], colors[i], positions[i], rotationSpeeds[i],
                              rotationStarts[i]);
        }
    }

    void setupDescriptorPool() {
        // One UBO for each gears
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = { // Binding 0 : Vertex shader uniform buffer
                                                                          { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex }
        };
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSets() {
        for (auto& gear : gears) {
            gear.setupDescriptorSet(descriptorPool, descriptorSetLayout);
        }
    }

    void preparePipelines() {
        // Solid rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.dynamicRendering(swapChain.surfaceFormat.format, deviceInfo.supportedDepthFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.colorBlendState.blendAttachmentStates.resize(1);
        pipelineBuilder.depthStencilState = { true };
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::gears::gears::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::gears::gears::frag, vk::ShaderStageFlagBits::eFragment);
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    void updateUniformBuffers() {
        glm::mat4 perspective = glm::perspective(glm::radians(60.0f), (float)size.width / (float)size.height, 0.001f, 256.0f);
        for (auto& gear : gears) {
            gear.updateUniformBuffer(perspective, camera.matrices.view, timer * 360.0f);
        }
    }

    void prepare() override {
        Parent::prepare();
        prepareVertices();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        updateUniformBuffers();
        buildCommandBuffers();
        prepared = true;
    }

    void update(float delta) override {
        Parent::update(delta);
        if (!paused) {
            updateUniformBuffers();
        }
    }

    void viewChanged() override { updateUniformBuffers(); }
};

RUN_EXAMPLE(VulkanExample)
