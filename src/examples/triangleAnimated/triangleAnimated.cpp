/*
* Vulkan Example - Basic indexed triangle rendering
*
* Note :
*    This is a "pedal to the metal" example to show off how to get Vulkan up an displaying something
*    Contrary to the other examples, this one won't make use of helper functions or initializers
*    Except in a few cases (swap chain setup e.g.)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    // As before
    vks::Buffer vertices;
    vks::Buffer indices;
    vks::Buffer uniformDataVS;
    uint32_t indexCount{ 0 };

    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;

    struct UboVS {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } uboVS;

    // As before
    VulkanExample() {
        size.width = 1280;
        size.height = 720;
        title = "Vulkan Example - Basic indexed triangle";
    }

    // As before
    ~VulkanExample() {
        vertices.destroy();
        indices.destroy();
        uniformDataVS.destroy();

        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);
    }

    void update(float deltaTime) override {
        Parent::update(deltaTime);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Update matrices
        uboVS.projectionMatrix = getProjection();
        uboVS.viewMatrix = glm::translate(glm::mat4(), camera.position);
        uboVS.modelMatrix = glm::mat4_cast(glm::quat_cast(camera.matrices.view));
        memcpy(uniformDataVS.mapped, &uboVS, sizeof(uboVS));
    }

    ////////////////////////////////////////
    //
    // All as before
    //
    void prepare() override {
        ExampleBase::prepare();
        prepareVertices();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuffer.bindVertexBuffers(0, vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(indexCount, 1, 0, 0, 1);
    }

    struct Vertex {
        float pos[3];
        float col[3];
    };

    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_COLOR,
    } };

    void prepareVertices() {
        // Setup vertices
        std::vector<Vertex> vertexBuffer = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
        vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
        indexCount = (uint32_t)indexBuffer.size();
        indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        // We need to tell the API the number of max. requested descriptors per type
        vk::DescriptorPoolSize typeCounts[1];
        // This example only uses one descriptor type (uniform buffer) and only
        // requests one descriptor of this type
        typeCounts[0].type = vk::DescriptorType::eUniformBuffer;
        typeCounts[0].descriptorCount = 1;
        // For additional types you need to add new entries in the type count list
        // E.g. for two combined image samplers :
        // typeCounts[1].type = vk::DescriptorType::eCombinedImageSampler;
        // typeCounts[1].descriptorCount = 2;

        // Create the global descriptor pool
        // All descriptors used in this example are allocated from this pool
        vk::DescriptorPoolCreateInfo descriptorPoolInfo;
        descriptorPoolInfo.poolSizeCount = 1;
        descriptorPoolInfo.pPoolSizes = typeCounts;
        // Set the max. number of sets that can be requested
        // Requesting descriptors beyond maxSets will result in an error
        descriptorPoolInfo.maxSets = 1;
        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        // Setup layout of descriptors used in this example
        // Basically connects the different shader stages to descriptors
        // for binding uniform buffers, image samplers, etc.
        // So every shader binding should map to one descriptor set layout
        // binding

        // Binding 0 : Uniform buffer (Vertex shader)
        vk::DescriptorSetLayoutBinding layoutBinding;
        layoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        layoutBinding.pImmutableSamplers = NULL;

        vk::DescriptorSetLayoutCreateInfo descriptorLayout;
        descriptorLayout.bindingCount = 1;
        descriptorLayout.pBindings = &layoutBinding;

        descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout, nullptr);

        // Create the pipeline layout that is used to generate the rendering pipelines that
        // are based on this descriptor set layout
        // In a more complex scenario you would have different pipeline layouts for different
        // descriptor set layouts that could be reused
        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo;
        pPipelineLayoutCreateInfo.setLayoutCount = 1;
        pPipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    }

    void setupDescriptorSet() {
        // Allocate a new descriptor set from the global descriptor pool
        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // Update the descriptor set determining the shader binding points
        // For every binding point used in a shader there needs to be one
        // descriptor set matching that binding point

        vk::WriteDescriptorSet writeDescriptorSet;

        // Binding 0 : Uniform buffer
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = vk::DescriptorType::eUniformBuffer;
        writeDescriptorSet.pBufferInfo = &uniformDataVS.descriptor;
        // Binds this uniform buffer to binding point 0
        writeDescriptorSet.dstBinding = 0;

        device.updateDescriptorSets(writeDescriptorSet, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.depthStencilState = { false };

        // Load shaders
        // Shaders are loaded from the SPIR-V format, which can be generated from glsl
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;
        pipelineBuilder.loadShader(getAssetPath() + "shaders/triangle/triangle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/triangle/triangle.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        // Create rendering pipeline
        pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        uboVS.projectionMatrix = getProjection();
        uboVS.viewMatrix = glm::translate(glm::mat4(), camera.position);
        uboVS.modelMatrix = glm::inverse(camera.matrices.skyboxView);
        uniformDataVS = context.createUniformBuffer(uboVS);
    }
};

RUN_EXAMPLE(VulkanExample)
