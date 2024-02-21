/*
 * Vulkan Example - Dynamic terrain tessellation
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <rendering/frustum.hpp>

#include <shaders/terraintessellation/skysphere.frag.inl>
#include <shaders/terraintessellation/skysphere.vert.inl>
#include <shaders/terraintessellation/terrain.frag.inl>
#include <shaders/terraintessellation/terrain.tesc.inl>
#include <shaders/terraintessellation/terrain.tese.inl>
#include <shaders/terraintessellation/terrain.vert.inl>

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_UV,
} };

class VulkanExample : public vkx::ExampleBase {
private:
    struct {
        vks::texture::Texture2D heightMap;
        vk::Sampler heightMapSampler;
        vks::texture::Texture2D skySphere;
        vks::texture::Texture2DArray terrainArray;
        vk::Sampler terrainSampler;
    } textures;

public:
    bool wireframe = false;
    bool tessellation = true;

    struct {
        vks::model::Model object;
        vks::model::Model skysphere;
    } meshes;

    struct {
        vks::Buffer terrainTessellation;
        vks::Buffer skysphereVertex;
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
    vks::Buffer queryResult;
    vk::QueryPool queryPool;
    std::array<uint64_t, 2> pipelineStats{ 0, 0 };

    // View frustum passed to tessellation control shader for culling
    vks::Frustum frustum;

    VulkanExample() {
        title = "Vulkan Example - Dynamic terrain tessellation";
        camera.type = Camera::CameraType::firstperson;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-12.0f, 159.0f, 0.0f));
        camera.setTranslation(glm::vec3(18.0f, 22.5f, 57.5f));
        camera.movementSpeed = 7.5f;
    }

    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        const auto& deviceFeatures = context.deviceInfo.features.core10;
        auto& enabledFeatures = context.enabledFeatures.core10;
        enabledFeatures.tessellationShader = deviceFeatures.tessellationShader;
        if (!enabledFeatures.tessellationShader) {
            throw std::runtime_error("Selected GPU does not support tessellation shaders!");
        }
        enabledFeatures.pipelineStatisticsQuery = deviceFeatures.pipelineStatisticsQuery;
        enabledFeatures.fillModeNonSolid = deviceFeatures.fillModeNonSolid;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.terrain);
        device.destroy(pipelines.wireframe);

        device.destroy(pipelineLayouts.skysphere);
        device.destroy(pipelineLayouts.terrain);

        device.destroy(descriptorSetLayouts.terrain);
        device.destroy(descriptorSetLayouts.skysphere);

        meshes.object.destroy();

        uniformData.skysphereVertex.destroy();
        uniformData.terrainTessellation.destroy();
        textures.heightMap.destroy();
        textures.skySphere.destroy();
        textures.terrainArray.destroy();
        device.destroy(queryPool);
        queryResult.destroy();
    }

    // Setup pool and buffer for storing pipeline statistics results
    void setupQueryResultBuffer() {
        uint32_t bufSize = 2 * sizeof(uint64_t);
        // Results are saved in a host visible buffer for easy access by the application
        queryResult = loader.createSizedUniformBuffer(bufSize);

        // Create query pool
        auto pipelineStatistics =
            vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations | vk::QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations;
        queryPool = device.createQueryPool({ {}, vk::QueryType::ePipelineStatistics, 2, pipelineStatistics });
    }

    // Retrieves the results of the pipeline statistics query submitted to the command buffer
    void getQueryResults() {
        // We use vkGetQueryResults to copy the results into a host visible buffer
        auto result = device.getQueryPoolResults(queryPool, 0, 1, 2 * sizeof(uint64_t), pipelineStats.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
        assert((result == vk::Result::eSuccess) || (result == vk::Result::eNotReady));
    }

    void loadAssets() override {
        meshes.skysphere.loadFromFile(getAssetPath() + "models/geosphere.obj", vertexLayout, 1.0f);
        textures.skySphere.loadFromFile(getAssetPath() + "textures/skysphere_bc3.ktx", vk::Format::eBc3UnormBlock);
        // Height data is stored in a one-channel texture
        textures.heightMap.loadFromFile(getAssetPath() + "textures/terrain_heightmap_r16.ktx", vk::Format::eR16Unorm);
        // Terrain textures are stored in a texture array with layers corresponding to terrain height
        textures.terrainArray.loadFromFile(getAssetPath() + "textures/terrain_texturearray_bc3.ktx", vk::Format::eBc3UnormBlock);

        // Setup a mirroring sampler for the height map
        vk::SamplerCreateInfo samplerInfo;
        samplerInfo.minFilter = samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.maxLod = (float)textures.heightMap.image.createInfo.mipLevels;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        textures.heightMapSampler = device.createSampler(samplerInfo);

        // Setup a repeating sampler for the terrain texture layers
        samplerInfo.maxLod = (float)textures.terrainArray.image.createInfo.mipLevels;
        if (samplerInfo.anisotropyEnable = deviceInfo.features.core10.samplerAnisotropy) {
            samplerInfo.maxAnisotropy = 4.0f;
        }
        textures.terrainSampler = device.createSampler(samplerInfo);
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {
        if (deviceInfo.features.core10.pipelineStatisticsQuery) {
            cmdBuffer.resetQueryPool(queryPool, 0, 2);
        }
        ExampleBase::updateCommandBufferPreDraw(cmdBuffer);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.setLineWidth(1.0f);
        // Skysphere
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skysphere);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.skysphere, 0, descriptorSets.skysphere, {});
        cmdBuffer.bindVertexBuffers(0, meshes.skysphere.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.skysphere.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.skysphere.indexCount, 1, 0, 0, 0);

        // Terrrain
        // Begin pipeline statistics query
        if (deviceInfo.features.core10.pipelineStatisticsQuery) {
            cmdBuffer.beginQuery(queryPool, 0, {});
        }
        // Render
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wireframe : pipelines.terrain);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.terrain, 0, descriptorSets.terrain, {});
        cmdBuffer.bindVertexBuffers(0, meshes.object.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
        // End pipeline statistics query
        if (deviceInfo.features.core10.pipelineStatisticsQuery) {
            cmdBuffer.endQuery(queryPool, 0);
        }
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

        meshes.object.vertices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertices);
        meshes.object.indices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indices);
        meshes.object.indexCount = static_cast<uint32_t>(indices.size());
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 3),
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayouts() {
        // Terrain
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Shared Tessellation shader ubo
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eTessellationControl | vk::ShaderStageFlagBits::eTessellationEvaluation },
            // Binding 1 : Height map
            {
                1,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                vk::ShaderStageFlagBits::eTessellationControl | vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment,
            },
            // Binding 3 : Terrain texture array layers
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.terrain = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.terrain = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.terrain });

        // Skysphere
        setLayoutBindings = {
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.skysphere = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.skysphere = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.skysphere });
    }

    void setupDescriptorSets() {
        // Terrain
        descriptorSets.terrain = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.terrain })[0];
        // Skysphere
        descriptorSets.skysphere = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.skysphere })[0];

        vk::DescriptorImageInfo texDescriptorHeightMap{ textures.heightMapSampler, textures.heightMap.imageView, vk::ImageLayout::eReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorTerrainArray{ textures.terrainSampler, textures.terrainArray.imageView, vk::ImageLayout::eReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorSkySphere{ defaultSampler, textures.skySphere.imageView, vk::ImageLayout::eReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Terrain
            // Binding 0 : Shared tessellation shader ubo
            { descriptorSets.terrain, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.terrainTessellation.descriptor },
            // Binding 1 : Displacement map
            { descriptorSets.terrain, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorHeightMap },
            // Binding 2 : Color map (alpha channel)
            { descriptorSets.terrain, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorTerrainArray },

            // Skysphere
            // Binding 0 : Vertex shader ubo
            { descriptorSets.skysphere, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.skysphereVertex.descriptor },
            // Binding 1 : Fragment shader color map
            { descriptorSets.skysphere, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorSkySphere },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Terrain tessellation pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayouts.terrain };
        pipelineCreator.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineCreator.inputAssemblyState.topology = vk::PrimitiveTopology::ePatchList;
        pipelineCreator.dynamicState.dynamicStateEnables.push_back(vk::DynamicState::eLineWidth);
        vertexLayout.appendVertexLayout(pipelineCreator.vertexInputState);
        // We render the terrain as a grid of quad patches
        vk::PipelineTessellationStateCreateInfo tessellationState{ {}, 4 };
        pipelineCreator.loadShader(vkx::shaders::terraintessellation::terrain::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::terraintessellation::terrain::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineCreator.loadShader(vkx::shaders::terraintessellation::terrain::tesc, vk::ShaderStageFlagBits::eTessellationControl);
        pipelineCreator.loadShader(vkx::shaders::terraintessellation::terrain::tese, vk::ShaderStageFlagBits::eTessellationEvaluation);
        pipelineCreator.pipelineCreateInfo.pTessellationState = &tessellationState;
        pipelines.terrain = pipelineCreator.create(context.pipelineCache);

        // Terrain wireframe pipeline
        pipelineCreator.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wireframe = pipelineCreator.create(context.pipelineCache);

        pipelineCreator.destroyShaderModules();

        // Skysphere pipeline
        pipelineCreator.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        // Revert to triangle list topology
        pipelineCreator.inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;
        // Reset tessellation state
        pipelineCreator.pipelineCreateInfo.pTessellationState = nullptr;
        // Don't write to depth buffer
        pipelineCreator.depthStencilState.depthWriteEnable = VK_FALSE;
        pipelineCreator.layout = pipelineLayouts.skysphere;
        pipelineCreator.loadShader(vkx::shaders::terraintessellation::skysphere::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::terraintessellation::skysphere::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.skysphere = pipelineCreator.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Shared tessellation shader stages uniform buffer
        uniformData.terrainTessellation = loader.createUniformBuffer(uboTess);
        uniformData.skysphereVertex = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Tessellation

        uboTess.projection = camera.matrices.perspective;
        uboTess.modelview = camera.matrices.view * glm::mat4();
        uboTess.lightPos.y = -0.5f - uboTess.displacementFactor;  // todo: Not uesed yet
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

    void draw() override {
        ExampleBase::prepareFrame();

        drawCurrentCommandBuffer();
        if (deviceInfo.features.core10.pipelineStatisticsQuery) {
            getQueryResults();
        }

        ExampleBase::submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        generateTerrain();
        if (deviceInfo.features.core10.pipelineStatisticsQuery) {
            setupQueryResultBuffer();
        }
        prepareUniformBuffers();
        setupDescriptorSetLayouts();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Tessellation", &tessellation)) {
                updateUniformBuffers();
            }
            if (ui.inputFloat("Factor", &uboTess.tessellationFactor, 0.05f, "%.2f")) {
                updateUniformBuffers();
            }
            if (deviceInfo.features.core10.fillModeNonSolid) {
                if (ui.checkBox("Wireframe", &wireframe)) {
                    buildCommandBuffers();
                }
            }
        }
        if (deviceInfo.features.core10.pipelineStatisticsQuery) {
            if (ui.header("Pipeline statistics")) {
                ui.text("VS invocations: %d", pipelineStats[0]);
                ui.text("TE invocations: %d", pipelineStats[1]);
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
