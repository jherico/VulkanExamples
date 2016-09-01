/*
* Vulkan Example - Dynamic terrain tessellation
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"
#include "frustum.hpp"

#define VERTEX_BUFFER_BIND_ID 0

// Vertex layout for this example
std::vector<vkx::VertexLayout> vertexLayout =
{
    vkx::VERTEX_LAYOUT_POSITION,
    vkx::VERTEX_LAYOUT_NORMAL,
    vkx::VERTEX_LAYOUT_UV
};

class VulkanExample : public vkx::ExampleBase {
private:
    struct {
        vkx::Texture heightMap;
        vkx::Texture skySphere;
        vkx::Texture terrainArray;
    } textures;
public:
    bool wireframe = false;
    bool tessellation = true;

    struct {
        vk::PipelineVertexInputStateCreateInfo inputState;
        std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
        std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    } vertices;

    struct {
        vkx::MeshBuffer object;
        vkx::MeshBuffer skysphere;
    } meshes;

    struct {
        vkx::UniformData terrainTessellation;
        vkx::UniformData skysphereVertex;
    } uniformData;

    // Shared values for tessellation control and evaluation stages
    struct {
        glm::mat4 projection;
        glm::mat4 modelview;
        glm::vec4 lightPos = glm::vec4(0.0f, -2.0f, 0.0f, 0.0f);
        glm::vec4 frustumPlanes[6];
        float displacementFactor = 32.0f;
        float tessellationFactor = 0.75f;
        glm::vec2 viewportDim;
        // Desired size of tessellated quad patch edge
        float tessellatedEdgeSize = 20.0f;
    } uboTess;

    // Skysphere vertex shader stage
    struct {
        glm::mat4 mvp;
    } uboVS;

    struct {
        vk::Pipeline terrain;
        vk::Pipeline wireframe;
        vk::Pipeline skysphere;
    } pipelines;

    struct {
        vk::DescriptorSetLayout terrain;
        vk::DescriptorSetLayout skysphere;
    } descriptorSetLayouts;

    struct {
        vk::PipelineLayout terrain;
        vk::PipelineLayout skysphere;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet terrain;
        vk::DescriptorSet skysphere;
    } descriptorSets;

    // Pipeline statistics
    CreateBufferResult queryResult;
    vk::QueryPool queryPool;
    std::array<uint64_t, 2> pipelineStats{ 0, 0 };

    // View frustum passed to tessellation control shader for culling
    vkTools::Frustum frustum;

    VulkanExample() : vkx::ExampleBase(ENABLE_VALIDATION) {
        enableTextOverlay = true;
        title = "Vulkan Example - Dynamic terrain tessellation";
        //camera.type = Camera::CameraType::firstperson;
        camera.setRotation({ 240.0f, 6.0f, 0.0f });
        camera.setTranslation({ -45.0f, 14.0f, -28.5f });
        camera.movementSpeed = 7.5f;
        timerSpeed *= 15.0f;
        // Support for tessellation shaders is optional, so check first
        //if (!deviceFeatures.tessellationShader)
        //{
        //	vkTools::exitFatal("Selected GPU does not support tessellation shaders!", "Feature not supported");
        //}
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources 
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.terrain, nullptr);
        device.destroyPipeline(pipelines.wireframe, nullptr);

        device.destroyPipelineLayout(pipelineLayouts.skysphere, nullptr);
        device.destroyPipelineLayout(pipelineLayouts.terrain, nullptr);

        device.destroyDescriptorSetLayout(descriptorSetLayouts.terrain, nullptr);
        device.destroyDescriptorSetLayout(descriptorSetLayouts.skysphere, nullptr);

        meshes.object.destroy();

        uniformData.skysphereVertex.destroy();
        uniformData.terrainTessellation.destroy();
        textures.heightMap.destroy();
        textures.skySphere.destroy();
        textures.terrainArray.destroy();

        device.destroyQueryPool(queryPool);

        queryResult.destroy();
        vkFreeMemory(device, queryResult.memory, nullptr);
    }

    // Setup pool and buffer for storing pipeline statistics results
    void setupQueryResultBuffer() {
        uint32_t bufSize = 2 * sizeof(uint64_t);
        // Results are saved in a host visible buffer for easy access by the application
        queryResult = createBuffer(vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst, bufSize);

        // Create query pool
        vk::QueryPoolCreateInfo queryPoolInfo;
        queryPoolInfo.queryType = vk::QueryType::ePipelineStatistics;
        queryPoolInfo.pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations |
            vk::QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations;
        queryPoolInfo.queryCount = 2;
        queryPool = device.createQueryPool(queryPoolInfo);
    }

    // Retrieves the results of the pipeline statistics query submitted to the command buffer
    void getQueryResults() {
        // We use vkGetQueryResults to copy the results into a host visible buffer
        vk::ArrayProxy<uint64_t> proxy{ pipelineStats };
        device.getQueryPoolResults(queryPool, 0, 1, proxy, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    }

    void loadTextures() {
        textures.skySphere = textureLoader->loadTexture(getAssetPath() + "textures/skysphere_bc3.ktx", vk::Format::eBc3UnormBlock);
        // Height data is stored in a one-channel texture
        textures.heightMap =textureLoader->loadTexture(getAssetPath() + "textures/terrain_heightmap_r16.ktx", vk::Format::eR16Unorm);
        // Terrain textures are stored in a texture array with layers corresponding to terrain height
        textures.terrainArray = textureLoader->loadTextureArray(getAssetPath() + "textures/terrain_texturearray_bc3.ktx", vk::Format::eBc3UnormBlock);

        // Setup a mirroring sampler for the height map
        device.destroySampler(textures.heightMap.sampler);
        vk::SamplerCreateInfo samplerInfo;
        samplerInfo.minFilter = samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.maxLod = (float)textures.heightMap.mipLevels;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        textures.heightMap.sampler = device.createSampler(samplerInfo);
        textures.heightMap.descriptor.sampler = textures.heightMap.sampler;
        textures.heightMap.descriptor.imageView = textures.heightMap.view;
        textures.heightMap.descriptor.imageLayout = textures.heightMap.imageLayout;

        // Setup a repeating sampler for the terrain texture layers
        device.destroySampler(textures.terrainArray.sampler);
        samplerInfo.maxLod = (float)textures.terrainArray.mipLevels;
        if (deviceFeatures.samplerAnisotropy) {
            samplerInfo.maxAnisotropy = 4.0f;
            samplerInfo.anisotropyEnable = VK_TRUE;
        }
        textures.terrainArray.sampler = device.createSampler(samplerInfo);
        textures.terrainArray.descriptor.sampler = textures.terrainArray.sampler;
        textures.terrainArray.descriptor.imageView = textures.terrainArray.view;
        textures.terrainArray.descriptor.imageLayout = textures.terrainArray.imageLayout;
    }

    void updatePrimaryCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.resetQueryPool(queryPool, 0, 2);
    }
    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vkx::viewport(size));
        cmdBuffer.setScissor(0, vkx::rect2D(size));
        cmdBuffer.setLineWidth(1.0f);
        // Skysphere
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skysphere);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.skysphere, 0, descriptorSets.skysphere, {});
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.skysphere.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.skysphere.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.skysphere.indexCount, 1, 0, 0, 0);

        // Terrrain
        // Begin pipeline statistics query			
        cmdBuffer.beginQuery(queryPool, 0, vk::QueryControlFlagBits::ePrecise);
        // Render
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wireframe : pipelines.terrain);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.terrain, 0, descriptorSets.terrain, {});
        cmdBuffer.bindVertexBuffers(VERTEX_BUFFER_BIND_ID, meshes.object.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
        // End pipeline statistics query
        cmdBuffer.endQuery(queryPool, 0);
    }

    void loadMeshes() {
        meshes.skysphere = loadMesh(getAssetPath() + "models/geosphere.obj", vertexLayout, 1.0f);
    }

    // Generate a terrain quad patch for feeding to the tessellation control shader
    void generateTerrain() {
        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec2 uv;
        };

#define PATCH_SIZE 64
#define UV_SCALE 1.0f

        std::vector<Vertex> vertices;
        vertices.resize(PATCH_SIZE * PATCH_SIZE * 4);

        const float wx = 2.0f;
        const float wy = 2.0f;

        for (auto x = 0; x < PATCH_SIZE; x++) {
            for (auto y = 0; y < PATCH_SIZE; y++) {
                uint32_t index = (x + y * PATCH_SIZE);
                vertices[index].pos[0] = x * wx + wx / 2.0f - (float)PATCH_SIZE * wx / 2.0f;
                vertices[index].pos[1] = 0.0f;
                vertices[index].pos[2] = y * wy + wy / 2.0f - (float)PATCH_SIZE * wy / 2.0f;
                vertices[index].normal = glm::vec3(0.0f, 1.0f, 0.0f);
                vertices[index].uv = glm::vec2((float)x / PATCH_SIZE, (float)y / PATCH_SIZE) * UV_SCALE;
            }
        }

        const uint32_t w = (PATCH_SIZE - 1);
        std::vector<uint32_t> indices;
        indices.resize(w * w * 4);
        for (auto x = 0; x < w; x++) {
            for (auto y = 0; y < w; y++) {
                uint32_t index = (x + y * w) * 4;
                indices[index] = (x + y * PATCH_SIZE);
                indices[index + 1] = indices[index] + PATCH_SIZE;
                indices[index + 2] = indices[index + 1] + 1;
                indices[index + 3] = indices[index] + 1;
            }
        }

        meshes.object.vertices = stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertices);
        meshes.object.indices = stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indices);
        meshes.object.indexCount = indices.size();
    }

    void setupVertexDescriptions() {
        // Binding description
        vertices.bindingDescriptions.resize(1);
        vertices.bindingDescriptions[0] =
            vkx::vertexInputBindingDescription(
                VERTEX_BUFFER_BIND_ID,
                vkx::vertexSize(vertexLayout),
                vk::VertexInputRate::eVertex);

        // Attribute descriptions
        // Describes memory layout and shader positions
        vertices.attributeDescriptions.resize(3);

        // Location 0 : Position
        vertices.attributeDescriptions[0] =
            vkx::vertexInputAttributeDescription(
                VERTEX_BUFFER_BIND_ID,
                0,
                vk::Format::eR32G32B32Sfloat,
                0);

        // Location 1 : Normals
        vertices.attributeDescriptions[1] =
            vkx::vertexInputAttributeDescription(
                VERTEX_BUFFER_BIND_ID,
                1,
                vk::Format::eR32G32B32Sfloat,
                sizeof(float) * 3);

        // Location 2 : Texture coordinates
        vertices.attributeDescriptions[2] =
            vkx::vertexInputAttributeDescription(
                VERTEX_BUFFER_BIND_ID,
                2,
                vk::Format::eR32G32Sfloat,
                sizeof(float) * 6);

        vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
        vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
        vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
        vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes =
        {
            vkx::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
            vkx::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 3)
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo =
            vkx::descriptorPoolCreateInfo(
                poolSizes.size(),
                poolSizes.data(),
                2);

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayouts() {
        vk::DescriptorSetLayoutCreateInfo descriptorLayout;
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;

        // Terrain
        setLayoutBindings =
        {
            // Binding 0 : Shared Tessellation shader ubo
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eTessellationControl | vk::ShaderStageFlagBits::eTessellationEvaluation,
                0),
            // Binding 1 : Height map
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eTessellationControl | vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
                1),
            // Binding 3 : Terrain texture array layers
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                2),
        };

        descriptorLayout = vkx::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
        descriptorSetLayouts.terrain = device.createDescriptorSetLayout(descriptorLayout);
        pipelineLayoutCreateInfo = vkx::pipelineLayoutCreateInfo(&descriptorSetLayouts.terrain, 1);
        pipelineLayouts.terrain = device.createPipelineLayout(pipelineLayoutCreateInfo);

        // Skysphere
        setLayoutBindings =
        {
            // Binding 0 : Vertex shader ubo
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eUniformBuffer,
                vk::ShaderStageFlagBits::eVertex,
                0),
            // Binding 1 : Color map
            vkx::descriptorSetLayoutBinding(
                vk::DescriptorType::eCombinedImageSampler,
                vk::ShaderStageFlagBits::eFragment,
                1),
        };

        descriptorLayout = vkx::descriptorSetLayoutCreateInfo(setLayoutBindings.data(), static_cast<uint32_t>(setLayoutBindings.size()));
        descriptorSetLayouts.skysphere = device.createDescriptorSetLayout(descriptorLayout);
        pipelineLayoutCreateInfo = vkx::pipelineLayoutCreateInfo(&descriptorSetLayouts.skysphere, 1);
        pipelineLayouts.skysphere = device.createPipelineLayout(pipelineLayoutCreateInfo);
    }

    void setupDescriptorSets() {
        vk::DescriptorSetAllocateInfo allocInfo;
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

        // Terrain
        allocInfo = vkx::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.terrain, 1);
        descriptorSets.terrain = device.allocateDescriptorSets(allocInfo)[0];

        writeDescriptorSets =
        {
            // Binding 0 : Shared tessellation shader ubo
            vkx::writeDescriptorSet(
                descriptorSets.terrain,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.terrainTessellation.descriptor),
            // Binding 1 : Displacement map
            vkx::writeDescriptorSet(
                descriptorSets.terrain,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                &textures.heightMap.descriptor),
            // Binding 2 : Color map (alpha channel)
            vkx::writeDescriptorSet(
                descriptorSets.terrain,
                vk::DescriptorType::eCombinedImageSampler,
                2,
                &textures.terrainArray.descriptor),
        };
        device.updateDescriptorSets(writeDescriptorSets, {});

        // Skysphere
        allocInfo = vkx::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.skysphere, 1);
        descriptorSets.skysphere = device.allocateDescriptorSets(allocInfo)[0];

        writeDescriptorSets =
        {
            // Binding 0 : Vertex shader ubo
            vkx::writeDescriptorSet(
                descriptorSets.skysphere,
                vk::DescriptorType::eUniformBuffer,
                0,
                &uniformData.skysphereVertex.descriptor),
            // Binding 1 : Fragment shader color map
            vkx::writeDescriptorSet(
                descriptorSets.skysphere,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                &textures.skySphere.descriptor),
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vkx::pipelineInputAssemblyStateCreateInfo(vk::PrimitiveTopology::ePatchList);

        vk::PipelineRasterizationStateCreateInfo rasterizationState =
            vkx::pipelineRasterizationStateCreateInfo(
                vk::PolygonMode::eFill,
                vk::CullModeFlagBits::eBack,
                vk::FrontFace::eClockwise);

        vk::PipelineColorBlendAttachmentState blendAttachmentState =
            vkx::pipelineColorBlendAttachmentState();

        vk::PipelineColorBlendStateCreateInfo colorBlendState =
            vkx::pipelineColorBlendStateCreateInfo(
                1,
                &blendAttachmentState);

        vk::PipelineDepthStencilStateCreateInfo depthStencilState =
            vkx::pipelineDepthStencilStateCreateInfo(
                VK_TRUE,
                VK_TRUE,
                vk::CompareOp::eLessOrEqual);

        vk::PipelineViewportStateCreateInfo viewportState =
            vkx::pipelineViewportStateCreateInfo(1, 1);

        vk::PipelineMultisampleStateCreateInfo multisampleState;

        std::vector<vk::DynamicState> dynamicStateEnables = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eLineWidth
        };
        vk::PipelineDynamicStateCreateInfo dynamicState =
            vkx::pipelineDynamicStateCreateInfo(
                dynamicStateEnables.data(),
                dynamicStateEnables.size());

        // We render the terrain as a grid of quad patches
        vk::PipelineTessellationStateCreateInfo tessellationState =
            vkx::pipelineTessellationStateCreateInfo(4);

        std::array<vk::PipelineShaderStageCreateInfo, 4> shaderStages;

        // Terrain tessellation pipeline
        shaderStages[0] = loadShader(getAssetPath() + "shaders/terraintessellation/terrain.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/terraintessellation/terrain.frag.spv", vk::ShaderStageFlagBits::eFragment);
        shaderStages[2] = loadShader(getAssetPath() + "shaders/terraintessellation/terrain.tesc.spv", vk::ShaderStageFlagBits::eTessellationControl);
        shaderStages[3] = loadShader(getAssetPath() + "shaders/terraintessellation/terrain.tese.spv", vk::ShaderStageFlagBits::eTessellationEvaluation);

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
            vkx::pipelineCreateInfo(
                pipelineLayouts.terrain,
                renderPass);

        pipelineCreateInfo.pVertexInputState = &vertices.inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;
        pipelineCreateInfo.pTessellationState = &tessellationState;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.renderPass = renderPass;

        pipelines.terrain = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

        // Terrain wireframe pipeline
        rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wireframe = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

        // Skysphere pipeline
        rasterizationState.polygonMode = vk::PolygonMode::eFill;
        // Revert to triangle list topology
        inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
        // Reset tessellation state
        pipelineCreateInfo.pTessellationState = nullptr;
        // Don't write to depth buffer
        depthStencilState.depthWriteEnable = VK_FALSE;
        pipelineCreateInfo.stageCount = 2;
        pipelineCreateInfo.layout = pipelineLayouts.skysphere;
        shaderStages[0] = loadShader(getAssetPath() + "shaders/terraintessellation/skysphere.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = loadShader(getAssetPath() + "shaders/terraintessellation/skysphere.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.skysphere = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Shared tessellation shader stages uniform buffer
        uniformData.terrainTessellation = createUniformBuffer(uboTess);
        uniformData.skysphereVertex = createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Tessellation

        uboTess.projection = camera.matrices.perspective;
        uboTess.modelview = camera.matrices.view * glm::mat4();
        uboTess.lightPos.y = -0.5f - uboTess.displacementFactor; // todo: Not uesed yet
        uboTess.viewportDim = glm::vec2((float)size.width, (float)size.height);
        frustum.update(uboTess.projection * uboTess.modelview);
        memcpy(uboTess.frustumPlanes, frustum.planes.data(), sizeof(glm::vec4) * 6);

        float savedFactor = uboTess.tessellationFactor;
        if (!tessellation) {
            // Setting this to zero sets all tessellation factors to 1.0 in the shader
            uboTess.tessellationFactor = 0.0f;
        }
        uniformData.terrainTessellation.copy(uboTess);
        if (!tessellation) {
            uboTess.tessellationFactor = savedFactor;
        }

        // Skysphere vertex shader
        uboVS.mvp = camera.matrices.perspective * glm::mat4(glm::mat3(camera.matrices.view));
        uniformData.skysphereVertex.copy(uboVS);
    }

    void prepare() {
        ExampleBase::prepare();
        loadMeshes();
        loadTextures();
        generateTerrain();
        setupQueryResultBuffer();
        setupVertexDescriptions();
        prepareUniformBuffers();
        setupDescriptorSetLayouts();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        updateDrawCommandBuffers();
        prepared = true;
    }

    virtual void viewChanged() {
        updateUniformBuffers();
    }

    void changeTessellationFactor(float delta) {
        uboTess.tessellationFactor += delta;
        uboTess.tessellationFactor = fmax(0.25f, fmin(uboTess.tessellationFactor, 4.0f));
        updateUniformBuffers();
        updateTextOverlay();
    }

    void toggleWireframe() {
        wireframe = !wireframe;
        updateDrawCommandBuffers();
        updateUniformBuffers();
    }

    void toggleTessellation() {
        tessellation = !tessellation;
        updateUniformBuffers();
    }

    virtual void keyPressed(uint32_t keyCode) {
        switch (keyCode) {
        case GLFW_KEY_KP_ADD:
        case GAMEPAD_BUTTON_R1:
            changeTessellationFactor(0.05f);
            break;
        case GLFW_KEY_KP_SUBTRACT:
        case GAMEPAD_BUTTON_L1:
            changeTessellationFactor(-0.05f);
            break;
        case GLFW_KEY_F:
        case GAMEPAD_BUTTON_A:
            toggleWireframe();
            break;
        case GLFW_KEY_T:
        case GAMEPAD_BUTTON_X:
            toggleTessellation();
            break;
        }
    }

    void getOverlayText(vkx::TextOverlay *textOverlay) override {
        std::stringstream ss;
        ss << std::setprecision(2) << std::fixed << uboTess.tessellationFactor;

#if defined(__ANDROID__)
        textOverlay->addText("Tessellation factor: " + ss.str() + " (Buttons L1/R1)", 5.0f, 85.0f, TextOverlay::alignLeft);
        textOverlay->addText("Press \"Button A\" to toggle wireframe", 5.0f, 100.0f, TextOverlay::alignLeft);
        textOverlay->addText("Press \"Button X\" to toggle tessellation", 5.0f, 115.0f, TextOverlay::alignLeft);
#else
        textOverlay->addText("Tessellation factor: " + ss.str() + " (numpad +/-)", 5.0f, 85.0f, TextOverlay::alignLeft);
        textOverlay->addText("Press \"f\" to toggle wireframe", 5.0f, 100.0f, TextOverlay::alignLeft);
        textOverlay->addText("Press \"t\" to toggle tessellation", 5.0f, 115.0f, TextOverlay::alignLeft);
#endif

        textOverlay->addText("pipeline stats:", size.width - 5.0f, 5.0f, TextOverlay::alignRight);
        textOverlay->addText("VS:" + std::to_string(pipelineStats[0]), size.width - 5.0f, 20.0f, TextOverlay::alignRight);
        textOverlay->addText("TE:" + std::to_string(pipelineStats[1]), size.width - 5.0f, 35.0f, TextOverlay::alignRight);
    }
};

RUN_EXAMPLE(VulkanExample)
