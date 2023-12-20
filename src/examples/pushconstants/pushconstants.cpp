/*
* Vulkan Example - Push constants example (small shader block accessed outside of uniforms for fast updates)
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

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    struct {
        vks::model::Model scene;
    } meshes;

    struct {
        vks::Buffer vertexShader;
    } uniformData;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0, 0.0, -2.0, 1.0);
    } uboVS;

    struct {
        vk::Pipeline solid;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    // This array holds the light positions
    // and will be updated via a push constant
    std::array<glm::vec4, 6> pushConstants;

    VulkanExample() {
        size.width = 1280;
        size.height = 720;
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        timerSpeed *= 0.5f;
        camera.dolly(-30.0f);
        camera.setRotation({ -32.5, 45.0, 0.0 });
        title = "Vulkan Example - Push constants";
    }

    void initVulkan() override {
        Parent::initVulkan();
        // todo : this crashes on certain Android devices, so commented out for now
#if !defined(__ANDROID__)
        // Check requested push constant size against hardware limit
        // Specs require 128 bytes, so if the device complies our
        // push constant buffer should always fit into memory
        vk::PhysicalDeviceProperties deviceProps;
        deviceProps = context.physicalDevice.getProperties();
        assert(sizeof(pushConstants) <= deviceProps.limits.maxPushConstantsSize);
#endif
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.solid);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        meshes.scene.destroy();
        uniformData.vertexShader.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        // Update light positions
        // w component = light radius scale
#define r 7.5f
#define sin_t sin(glm::radians(timer * 360))
#define cos_t cos(glm::radians(timer * 360))
#define y -4.0f
        pushConstants[0] = glm::vec4(r * 1.1 * sin_t, y, r * 1.1 * cos_t, 1.0f);
        pushConstants[1] = glm::vec4(-r * sin_t, y, -r * cos_t, 1.0f);
        pushConstants[2] = glm::vec4(r * 0.85f * sin_t, y, -sin_t * 2.5f, 1.5f);
        pushConstants[3] = glm::vec4(0.0f, y, r * 1.25f * cos_t, 1.5f);
        pushConstants[4] = glm::vec4(r * 2.25f * cos_t, y, 0.0f, 1.25f);
        pushConstants[5] = glm::vec4(r * 2.5f * cos(glm::radians(timer * 360)), y, r * 2.5f * sin_t, 1.25f);
#undef r
#undef y
#undef sin_t
#undef cos_t

        // Submit via push constant (rather than a UBO)
        cmdBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(pushConstants), pushConstants.data());

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, meshes.scene.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.scene.indices.buffer, 0, vk::IndexType::eUint32);

        cmdBuffer.drawIndexed(meshes.scene.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override { meshes.scene.loadFromFile(context, getAssetPath() + "models/samplescene.dae", vertexLayout, 0.35f); }

    void setupDescriptorPool() {
        // Example uses one ubo
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 1 },
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() };
        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        // Define push constant
        // Example uses six light positions as push constants
        // 6 * 4 * 4 = 96 bytes
        // Spec requires a minimum of 128 bytes, bigger values
        // need to be checked against maxPushConstantsSize
        // But even at only 128 bytes, lots of stuff can fit
        // inside push constants
        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(pushConstants) };

        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout, 1, &pushConstantRange });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Binding 0 : Vertex shader uniform buffer
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vertexShader.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Solid rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pushconstants/lights.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pushconstants/lights.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vertexShader = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        uboVS.projection = getProjection();
        uboVS.model = camera.matrices.view;
        uniformData.vertexShader.copy(uboVS);
    }

    void prepare() override {
        Parent::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void update(float delta) override {
        Parent::update(delta);
        if (!paused) {
            buildCommandBuffers();
        }
    }

    void viewChanged() override { updateUniformBuffers(); }

    void windowResized() override {
        Parent::windowResized();
        preparePipelines();
    }
};

RUN_EXAMPLE(VulkanExample)
