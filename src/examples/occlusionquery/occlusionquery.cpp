/*
* Vulkan Example - Using occlusion query for visbility testing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

// Vertex layout used in this example
// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::model::Model teapot;
        vks::model::Model plane;
        vks::model::Model sphere;
    } meshes;

    struct {
        vks::Buffer vsScene;
        vks::Buffer teapot;
        vks::Buffer sphere;
    } uniformData;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f);
        float visible;
    } uboVS;

    struct {
        vk::Pipeline solid;
        vk::Pipeline occluder;
        // vk::Pipeline with basic shaders used for occlusion pass
        vk::Pipeline simple;
    } pipelines;

    struct {
        vk::DescriptorSet teapot;
        vk::DescriptorSet sphere;
    } descriptorSets;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Stores occlusion query results
    vks::Buffer queryResult;

    // Pool that stores all occlusion queries
    vk::QueryPool queryPool;

    // Passed query samples
    std::vector<uint64_t> passedSamples{ 1, 1 };

    VulkanExample() {
        passedSamples[0] = passedSamples[1] = 1;
        size = vk::Extent2D{ 1280, 720 };
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ 0.0, -123.75, 0.0 });
        camera.dolly(-35.0f);
        title = "Vulkan Example - Occlusion queries";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.solid);
        device.destroyPipeline(pipelines.occluder);
        device.destroyPipeline(pipelines.simple);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyQueryPool(queryPool);

        device.destroyBuffer(queryResult.buffer);
        device.freeMemory(queryResult.memory);

        uniformData.vsScene.destroy();
        uniformData.sphere.destroy();
        uniformData.teapot.destroy();

        meshes.sphere.destroy();
        meshes.plane.destroy();
        meshes.teapot.destroy();
    }

    // Create a buffer for storing the query result
    // Setup a query pool
    void setupQueryResultBuffer() {
        uint32_t bufSize = 2 * sizeof(uint64_t);
        // Results are saved in a host visible buffer for easy access by the application
        queryResult = context.createBuffer(vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                           vk::MemoryPropertyFlagBits::eHostVisible, bufSize);
        // Query pool will be created for occlusion queries
        queryPool = device.createQueryPool({ {}, vk::QueryType::eOcclusion, 2 });
    }

    // Retrieves the results of the occlusion queries submitted to the command buffer
    void getQueryResults() {
        queue.waitIdle();
        device.waitIdle();
        // Store results a 64 bit values and wait until the results have been finished
        // If you don't want to wait, you can use VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
        // which also returns the state of the result (ready) in the result
        static const auto queryResultFlags = vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait;
        // We use vkGetQueryResults to copy the results into a host visible buffer
        // you can use vk::QueryResultFlagBits::eWithAvailability
        // which also returns the state of the result (ready) in the result
        passedSamples = device.getQueryPoolResults<uint64_t>(queryPool, 0, 2, 2 * sizeof(uint64_t), sizeof(uint64_t), queryResultFlags).value;
        //vk::ArrayProxy<uint64_t>{ passedSamples };
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {
        // Reset query pool
        // Must be done outside of render pass
        cmdBuffer.resetQueryPool(queryPool, 0, 2);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, viewport());
        cmdBuffer.setScissor(0, scissor());

        // Occlusion pass
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.simple);

        // Occluder first
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.plane.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.plane.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.plane.indexCount, 1, 0, 0, 0);

        // Teapot
        cmdBuffer.beginQuery(queryPool, 0, vk::QueryControlFlags());

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.teapot, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.teapot.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.teapot.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.teapot.indexCount, 1, 0, 0, 0);

        cmdBuffer.endQuery(queryPool, 0);

        // Sphere
        cmdBuffer.beginQuery(queryPool, 1, vk::QueryControlFlags());

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sphere, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.sphere.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.sphere.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.sphere.indexCount, 1, 0, 0, 0);

        cmdBuffer.endQuery(queryPool, 1);

        // Visible pass
        // Clear color and depth attachments
        std::array<vk::ClearAttachment, 2> clearAttachments;
        clearAttachments[0].aspectMask = vk::ImageAspectFlagBits::eColor;
        clearAttachments[0].clearValue.color = defaultClearColor;
        clearAttachments[0].colorAttachment = 0;

        clearAttachments[1].aspectMask = vk::ImageAspectFlagBits::eDepth;
        clearAttachments[1].clearValue.depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        vk::ClearRect clearRect;
        clearRect.layerCount = 1;
        clearRect.rect.extent = size;

        cmdBuffer.clearAttachments(clearAttachments, clearRect);

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);

        // Teapot
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.teapot, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.teapot.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.teapot.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.teapot.indexCount, 1, 0, 0, 0);

        // Sphere
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sphere, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.sphere.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.sphere.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.sphere.indexCount, 1, 0, 0, 0);

        // Occluder
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.occluder);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.plane.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.plane.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.plane.indexCount, 1, 0, 0, 0);
    }

    void draw() override {
        prepareFrame();

        drawCurrentCommandBuffer();

        // Read query results for displaying in next frame
        getQueryResults();

        submitFrame();
    }

    void loadMeshes() {
        meshes.plane.loadFromFile(context, getAssetPath() + "models/plane_z.3ds", vertexLayout, 0.4f);
        meshes.teapot.loadFromFile(context, getAssetPath() + "models/teapot.3ds", vertexLayout, 0.3f);
        meshes.sphere.loadFromFile(context, getAssetPath() + "models/sphere.3ds", vertexLayout, 0.3f);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{ // One uniform buffer block for each mesh
                                                       vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3)
        };
        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex }
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        // Occluder (plane)
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsScene.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Teapot
        descriptorSets.teapot = device.allocateDescriptorSets(allocInfo)[0];
        writeDescriptorSets[0].dstSet = descriptorSets.teapot;
        writeDescriptorSets[0].pBufferInfo = &uniformData.teapot.descriptor;
        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Sphere
        descriptorSets.sphere = device.allocateDescriptorSets(allocInfo)[0];
        writeDescriptorSets[0].dstSet = descriptorSets.sphere;
        writeDescriptorSets[0].pBufferInfo = &uniformData.sphere.descriptor;
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };

        // Solid rendering pipeline
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{ {}, vk::PrimitiveTopology::eTriangleList };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/occlusionquery/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/occlusionquery/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Solid rendering pipeline
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Basic pipeline for coloring occluded objects
        pipelineBuilder.loadShader(getAssetPath() + "shaders/occlusionquery/simple.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/occlusionquery/simple.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelines.simple = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Visual pipeline for the occluder
        pipelineBuilder.loadShader(getAssetPath() + "shaders/occlusionquery/occluder.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/occlusionquery/occluder.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Enable blending
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcColor;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor;
        pipelines.occluder = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = context.createUniformBuffer(uboVS);
        // Teapot
        uniformData.teapot = context.createUniformBuffer(uboVS);
        // Sphere
        uniformData.sphere = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;

        // Occluder
        uboVS.visible = 1.0f;
        uniformData.vsScene.copy(uboVS);

        // Teapot
        // Toggle color depending on visibility
        uboVS.visible = (passedSamples[0] > 0) ? 1.0f : 0.0f;
        uboVS.model = camera.matrices.view * glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, -10.0f));
        uniformData.teapot.copy(uboVS);

        // Sphere
        // Toggle color depending on visibility
        uboVS.visible = (passedSamples[1] > 0) ? 1.0f : 0.0f;
        uboVS.model = camera.matrices.view * glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, 10.0f));
        uniformData.sphere.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadMeshes();
        setupQueryResultBuffer();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
    }

    void viewChanged() override { updateUniformBuffers(); }
};

RUN_EXAMPLE(VulkanExample)
