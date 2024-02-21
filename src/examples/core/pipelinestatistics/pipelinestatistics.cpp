/*
 * Vulkan Example - Retrieving pipeline statistics
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/pipelinestatistics/scene.frag.inl>
#include <shaders/pipelinestatistics/scene.vert.inl>
#include <shaders/pipelinestatistics/scene.tesc.inl>
#include <shaders/pipelinestatistics/scene.tese.inl>

#define OBJ_DIM 0.05f

class VulkanExample : public vkx::ExampleBase {
public:
    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_COLOR,
    });

    struct Models {
        std::vector<vks::model::Model> objects;
        int32_t objectIndex = 3;
        std::vector<std::string> names;
    } models;

    struct UniformBuffers {
        vks::Buffer VS;
    } uniformBuffers;

    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 modelview;
        glm::vec4 lightPos = glm::vec4(-10.0f, -10.0f, 10.0f, 1.0f);
    } uboVS;

    vk::Pipeline pipeline;

    vk::CullModeFlagBits cullMode = vk::CullModeFlagBits::eBack;
    bool blending = false;
    bool discard = false;
    bool wireframe = false;
    bool tessellation = false;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    vk::QueryPool queryPool;

    // Vector for storing pipeline statistics results
    std::vector<uint64_t> pipelineStats;
    std::vector<std::string> pipelineStatNames;

    int32_t gridSize = 3;

    VulkanExample() {
        title = "Pipeline statistics";
        camera.type = Camera::CameraType::firstperson;
        camera.setPosition(glm::vec3(-4.0f, 3.0f, -3.75f));
        camera.setRotation(glm::vec3(-15.25f, -46.5f, 0.0f));
        camera.movementSpeed = 4.0f;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
        camera.rotationSpeed = 0.25f;
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(queryPool);
        uniformBuffers.VS.destroy();
        for (auto& model : models.objects) {
            model.destroy();
        }
    }

    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        const auto& deviceFeatures = deviceInfo.features.core10;
        auto& enabledFeatures = context.enabledFeatures.core10;
        // Support for pipeline statistics is optional
        if (deviceFeatures.pipelineStatisticsQuery) {
            enabledFeatures.pipelineStatisticsQuery = VK_TRUE;
        } else {
            throw std::runtime_error("Selected GPU does not support pipeline statistics!");
        }
        if (deviceFeatures.fillModeNonSolid) {
            enabledFeatures.fillModeNonSolid = VK_TRUE;
        }
        if (deviceFeatures.tessellationShader) {
            enabledFeatures.tessellationShader = VK_TRUE;
        }
    }

    // Setup a query pool for storing pipeline statistics
    void setupQueryPool() {
        const auto& deviceFeatures = deviceInfo.features.core10;
        pipelineStatNames = {
            "Input assembly vertex count        ", "Input assembly primitives count    ", "Vertex shader invocations          ",
            "Clipping stage primitives processed", "Clipping stage primtives output    ", "Fragment shader invocations        ",
        };
        if (deviceFeatures.tessellationShader) {
            pipelineStatNames.push_back("Tess. control shader patches       ");
            pipelineStatNames.push_back("Tess. eval. shader invocations     ");
        }
        pipelineStats.resize(pipelineStatNames.size());

        vk::QueryPoolCreateInfo queryPoolInfo = {};
        // This query pool will store pipeline statistics
        queryPoolInfo.queryType = vk::QueryType::ePipelineStatistics;

        // Pipeline counters to be returned for this pool
        queryPoolInfo.pipelineStatistics =
            vk::QueryPipelineStatisticFlagBits::eInputAssemblyVertices | vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
            vk::QueryPipelineStatisticFlagBits::eVertexShaderInvocations | vk::QueryPipelineStatisticFlagBits::eClippingInvocations |
            vk::QueryPipelineStatisticFlagBits::eClippingPrimitives | vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
        if (deviceFeatures.tessellationShader) {
            queryPoolInfo.pipelineStatistics |= vk::QueryPipelineStatisticFlagBits::eTessellationControlShaderPatches |
                                                vk::QueryPipelineStatisticFlagBits::eTessellationEvaluationShaderInvocations;
        }
        queryPoolInfo.queryCount = deviceFeatures.tessellationShader ? 8 : 6;
        queryPool = device.createQueryPool(queryPoolInfo);
    }

    // Retrieves the results of the pipeline statistics query submitted to the command buffer
    void getQueryResults() {
        const auto& deviceFeatures = deviceInfo.features.core10;
        // uint32_t count = static_cast<uint32_t>(pipelineStats.size());
        auto queryCount = deviceFeatures.tessellationShader ? 8 : 6;
        auto result =
            device.getQueryPoolResults<uint64_t>(queryPool, 0, queryCount, queryCount * sizeof(uint64_t), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
        if (result.value[0] != 0) {
            pipelineStats = result.value;
        }
        // if (result.result != vk::Result::eNotReady) {
        //
        //     pipelineStats = result.value;
        // }
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& drawCmdBuffer) override {
        // Reset timestamp query pool
        drawCmdBuffer.resetQueryPool(queryPool, 0, static_cast<uint32_t>(pipelineStats.size()));
        ExampleBase::updateCommandBufferPreDraw(drawCmdBuffer);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());

        VkDeviceSize offsets[1] = { 0 };

        // Start capture of pipeline statistics
        drawCmdBuffer.beginQuery(queryPool, 0, vk::QueryControlFlagBits::ePrecise);

        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        drawCmdBuffer.bindVertexBuffers(0, 1, &models.objects[models.objectIndex].vertices.buffer, offsets);
        drawCmdBuffer.bindIndexBuffer(models.objects[models.objectIndex].indices.buffer, 0, vk::IndexType::eUint32);

        for (int32_t y = 0; y < gridSize; y++) {
            for (int32_t x = 0; x < gridSize; x++) {
                glm::vec3 pos = glm::vec3(float(x - (gridSize / 2.0f)) * 2.5f, 0.0f, float(y - (gridSize / 2.0f)) * 2.5f);
                drawCmdBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::vec3), &pos);
                drawCmdBuffer.drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
            }
        }

        // End capture of pipeline statistics
        drawCmdBuffer.endQuery(queryPool, 0);
    }

    void draw() override {
        ExampleBase::prepareFrame();

        drawCurrentCommandBuffer();

        // Read query results for displaying in next frame
        getQueryResults();

        ExampleBase::submitFrame();
    }

    void loadAssets() override {
        // Objects
        std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
        for (auto file : filenames) {
            vks::model::Model model;
            model.loadFromFile(getAssetPath() + "models/" + file, vertexLayout, OBJ_DIM * (file == "venus.fbx" ? 3.0f : 1.0f));
            models.objects.push_back(std::move(model));
        }
        models.names = { "Sphere", "Teapot", "Torusknot", "Venus" };
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 3 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::vec3) };
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout, 1, &pushConstantRange });
    }

    void setupDescriptorSets() {
        // Scene rendering
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.VS.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        builder.rasterizationState.cullMode = cullMode;
        vertexLayout.appendVertexLayout(builder.vertexInputState);
        vk::PipelineTessellationStateCreateInfo tessellationState{ {}, 3 };

        if (blending) {
            auto& blendAttachmentState = builder.colorBlendState.blendAttachmentStates[0];
            blendAttachmentState.blendEnable = VK_TRUE;
            blendAttachmentState.colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
            blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
            blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
            blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
            builder.depthStencilState.depthWriteEnable = VK_FALSE;
        }

        if (discard) {
            builder.rasterizationState.rasterizerDiscardEnable = VK_TRUE;
        }

        if (wireframe) {
            builder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        }

        builder.loadShader(vkx::shaders::pipelinestatistics::scene::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::pipelinestatistics::scene::frag, vk::ShaderStageFlagBits::eFragment);

        if (tessellation) {
            builder.inputAssemblyState.topology = vk::PrimitiveTopology::ePatchList;
            builder.pipelineCreateInfo.pTessellationState = &tessellationState;
            builder.loadShader(vkx::shaders::pipelinestatistics::scene::tesc, vk::ShaderStageFlagBits::eTessellationControl);
            builder.loadShader(vkx::shaders::pipelinestatistics::scene::tese, vk::ShaderStageFlagBits::eTessellationEvaluation);
        }

        pipeline = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        uniformBuffers.VS = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.modelview = camera.matrices.view;
        uniformBuffers.VS.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();
        setupQueryPool();
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

    void rebuildPipelines() {
        graphicsQueue.handle.waitIdle();
        device.waitIdle();
        if (pipeline) {
            device.destroy(pipeline);
            pipeline = nullptr;
        }
        preparePipelines();
        buildCommandBuffers();
    }

    void OnUpdateUIOverlay() override {
        const auto& deviceFeatures = deviceInfo.features.core10;
        if (ui.header("Settings")) {
            if (ui.comboBox("Object type", &models.objectIndex, models.names)) {
                updateUniformBuffers();
                buildCommandBuffers();
            }
            if (ui.sliderInt("Grid size", &gridSize, 1, 10)) {
                buildCommandBuffers();
            }
            std::vector<std::string> cullModeNames = { "None", "Front", "Back", "Back and front" };
            if (ui.comboBox("Cull mode", &reinterpret_cast<int32_t&>(cullMode), cullModeNames)) {
                rebuildPipelines();
            }
            if (ui.checkBox("Blending", &blending)) {
                rebuildPipelines();
            }
            if (deviceFeatures.fillModeNonSolid) {
                if (ui.checkBox("Wireframe", &wireframe)) {
                    rebuildPipelines();
                }
            }
            if (deviceFeatures.tessellationShader) {
                if (ui.checkBox("Tessellation", &tessellation)) {
                    rebuildPipelines();
                }
            }
            if (ui.checkBox("Discard", &discard)) {
                rebuildPipelines();
            }
        }
        if (!pipelineStats.empty()) {
            if (ui.header("Pipeline statistics")) {
                for (auto i = 0; i < pipelineStats.size(); i++) {
                    std::string caption = pipelineStatNames[i] + ": %d";
                    ui.text(caption.c_str(), pipelineStats[i]);
                }
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
