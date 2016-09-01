/*
* Vulkan Example - Using occlusion query for visbility testing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"


// Vertex layout used in this example
// Vertex layout for this example
std::vector<vkx::VertexLayout> vertexLayout =
{
    vkx::VertexLayout::VERTEX_LAYOUT_POSITION,
    vkx::VertexLayout::VERTEX_LAYOUT_NORMAL,
    vkx::VertexLayout::VERTEX_LAYOUT_COLOR,
};

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vk::PipelineVertexInputStateCreateInfo inputState;
        std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    struct {
        vkx::MeshBuffer teapot;
        vkx::MeshBuffer plane;
        vkx::MeshBuffer sphere;
    } meshes;

    struct {
        vkx::UniformData vsScene;
        vkx::UniformData teapot;
        vkx::UniformData sphere;
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
    struct {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } queryResult;

    // Pool that stores all occlusion queries
    vk::QueryPool queryPool;

    // Passed query samples
    uint64_t passedSamples[2];

    VulkanExample() : vkx::ExampleBase(ENABLE_VALIDATION) {
        passedSamples[0] = passedSamples[1] = 1;
        vk::Extent2D tst( 1280, 720 );
        //size = { 1280, 720 };
        size = tst;
        camera.setZoom(-35.0f);
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ 0.0, -123.75, 0.0 });
        enableTextOverlay = true;
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

        vk::MemoryRequirements memReqs;
        vk::MemoryAllocateInfo memAlloc;
        vk::BufferCreateInfo bufferCreateInfo =
            vkx::bufferCreateInfo(vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst, bufSize);

        // Results are saved in a host visible buffer for easy access by the application
        queryResult.buffer = device.createBuffer(bufferCreateInfo);
        memReqs = device.getBufferMemoryRequirements(queryResult.buffer);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        queryResult.memory = device.allocateMemory(memAlloc);
        device.bindBufferMemory(queryResult.buffer, queryResult.memory, 0);

        // Create query pool
        vk::QueryPoolCreateInfo queryPoolInfo;
        // Query pool will be created for occlusion queries
        queryPoolInfo.queryType = vk::QueryType::eOcclusion;
        queryPoolInfo.queryCount = 2;

        queryPool = device.createQueryPool(queryPoolInfo, NULL);
    }

    // Retrieves the results of the occlusion queries submitted to the command buffer
    void getQueryResults() {
        queue.waitIdle();
        device.waitIdle();
        // We use vkGetQueryResults to copy the results into a host visible buffer
        // you can use vk::QueryResultFlagBits::eWithAvailability
            // which also returns the state of the result (ready) in the result
        device.getQueryPoolResults(
            queryPool,
            0,
            2,
            sizeof(passedSamples),
            passedSamples,
            sizeof(uint64_t),
            // Store results a 64 bit values and wait until the results have been finished
            // If you don't want to wait, you can use VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
            // which also returns the state of the result (ready) in the result
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    }

    void updatePrimaryCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        // Reset query pool
        // Must be done outside of render pass
        cmdBuffer.resetQueryPool(queryPool, 0, 2);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.setViewport(0, vkx::viewport(size));
        cmdBuffer.setScissor(0, vkx::rect2D(size));

        // Occlusion pass
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.simple);

        // Occluder first
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.plane.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.plane.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.plane.indexCount, 1, 0, 0, 0);

        // Teapot
        cmdBuffer.beginQuery(queryPool, 0, vk::QueryControlFlags());

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.teapot, nullptr);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.teapot.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.teapot.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.teapot.indexCount, 1, 0, 0, 0);

        cmdBuffer.endQuery(queryPool, 0);

        // Sphere
        cmdBuffer.beginQuery(queryPool, 1, vk::QueryControlFlags());

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sphere, nullptr);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.sphere.vertices.buffer, { 0 });
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
        clearAttachments[1].clearValue.depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

        vk::ClearRect clearRect;
        clearRect.layerCount = 1;
        clearRect.rect.extent = size;

        cmdBuffer.clearAttachments(clearAttachments, clearRect);

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);

        // Teapot
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.teapot, nullptr);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.teapot.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.teapot.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.teapot.indexCount, 1, 0, 0, 0);

        // Sphere
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sphere, nullptr);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.sphere.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.sphere.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.sphere.indexCount, 1, 0, 0, 0);

        // Occluder
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.occluder);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.plane.vertices.buffer, { 0 });
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
        meshes.plane = loadMesh(getAssetPath() + "models/plane_z.3ds", vertexLayout, 0.4f);
        meshes.teapot = loadMesh(getAssetPath() + "models/teapot.3ds", vertexLayout, 0.3f);
        meshes.sphere = loadMesh(getAssetPath() + "models/sphere.3ds", vertexLayout, 0.3f);
    }

    void setupVertexDescriptions() {
        // Binding description
        vertices.bindingDescriptions.resize(1);
        vertices.bindingDescriptions[0] =
            vkx::vertexInputBindingDescription(VERTEX_BUFFER_BIND_ID, vkx::vertexSize(vertexLayout), vk::VertexInputRate::eVertex);

        // Attribute descriptions
        // Describes memory layout and shader positions
        vertices.attributeDescriptions.resize(3);
        // Location 0 : Position
        vertices.attributeDescriptions[0] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 0, vk::Format::eR32G32B32Sfloat, 0);
        // Location 1 : Normal
        vertices.attributeDescriptions[1] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3);
        // Location 3 : Color
        vertices.attributeDescriptions[2] =
            vkx::vertexInputAttributeDescription(VERTEX_BUFFER_BIND_ID, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 6);

        vertices.inputState = vk::PipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
        vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
        vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
        vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes =
        {
            // One uniform buffer block for each mesh
            vkx::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3)
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo =
            vkx::descriptorPoolCreateInfo(poolSizes.size(), poolSizes.data(), 3);

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex,
                0)
        };

        vk::DescriptorSetLayoutCreateInfo descriptorLayout =
            vkx::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), setLayoutBindings.size());

        descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
            vkx::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);

        pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    }

    void setupDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocInfo =
            vkx::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

        // Occluder (plane)
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
        {
            // Binding 0 : Vertex shader uniform buffer
            vkx::writeDescriptorSet(
                descriptorSet,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.vsScene.descriptor)
        };

        device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

        // Teapot
        descriptorSets.teapot = device.allocateDescriptorSets(allocInfo)[0];
        writeDescriptorSets[0].dstSet = descriptorSets.teapot;
        writeDescriptorSets[0].pBufferInfo = &uniformData.teapot.descriptor;
        device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

        // Sphere
        descriptorSets.sphere = device.allocateDescriptorSets(allocInfo)[0];
        writeDescriptorSets[0].dstSet = descriptorSets.sphere;
        writeDescriptorSets[0].pBufferInfo = &uniformData.sphere.descriptor;
        device.updateDescriptorSets(writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
    }

    void preparePipelines() {
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vkx::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::eTriangleList, vk::PipelineInputAssemblyStateCreateFlags(), VK_FALSE);

        vk::PipelineRasterizationStateCreateInfo rasterizationState =
            vkx::pipelineRasterizationStateCreateInfo(vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eClockwise);

        vk::PipelineColorBlendAttachmentState blendAttachmentState =
            vkx::pipelineColorBlendAttachmentState();

        vk::PipelineColorBlendStateCreateInfo colorBlendState =
            vkx::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

        vk::PipelineDepthStencilStateCreateInfo depthStencilState =
            vkx::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, vk::CompareOp::eLessOrEqual);

        vk::PipelineViewportStateCreateInfo viewportState =
            vkx::pipelineViewportStateCreateInfo(1, 1);

        vk::PipelineMultisampleStateCreateInfo multisampleState =
            vkx::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

        std::vector<vk::DynamicState> dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState =
            vkx::pipelineDynamicStateCreateInfo(dynamicStateEnables.data(), dynamicStateEnables.size());

        // Solid rendering pipeline
        // Load shaders
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = loadShader(getAssetPath() + "shaders/occlusionquery/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/occlusionquery/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
            vkx::pipelineCreateInfo(pipelineLayout, renderPass);

        pipelineCreateInfo.pVertexInputState = &vertices.inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.stageCount = shaderStages.size();
        pipelineCreateInfo.pStages = shaderStages.data();

        pipelines.solid = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];

        // Basic pipeline for coloring occluded objects
        shaderStages[0] = loadShader(getAssetPath() + "shaders/occlusionquery/simple.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/occlusionquery/simple.frag.spv", vk::ShaderStageFlagBits::eFragment);
        rasterizationState.cullMode = vk::CullModeFlagBits::eNone;

        pipelines.simple = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];

        // Visual pipeline for the occluder
        shaderStages[0] = loadShader(getAssetPath() + "shaders/occlusionquery/occluder.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/occlusionquery/occluder.frag.spv", vk::ShaderStageFlagBits::eFragment);

        // Enable blending
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcColor;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor;

        pipelines.occluder = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo, nullptr)[0];
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = createUniformBuffer(uboVS);
        // Teapot
        uniformData.teapot = createUniformBuffer(uboVS);
        // Sphere
        uniformData.sphere = createUniformBuffer(uboVS);
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

    void prepare() {
        ExampleBase::prepare();
        loadMeshes();
        setupQueryResultBuffer();
        setupVertexDescriptions();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        updateDrawCommandBuffers();
        prepared = true;
    }

    virtual void render() {
        if (!prepared)
            return;
        draw();
    }

    virtual void viewChanged() {
        updateUniformBuffers();
        ExampleBase::updateTextOverlay();
    }

    virtual void getOverlayText(vkx::TextOverlay *textOverlay) {
        textOverlay->addText("Occlusion queries:", 5.0f, 85.0f, vkx::TextOverlay::alignLeft);
        textOverlay->addText("Teapot: " + std::to_string(passedSamples[0]) + " samples passed", 5.0f, 105.0f, vkx::TextOverlay::alignLeft);
        textOverlay->addText("Sphere: " + std::to_string(passedSamples[1]) + " samples passed", 5.0f, 125.0f, vkx::TextOverlay::alignLeft);
    }
};

RUN_EXAMPLE(VulkanExample)
