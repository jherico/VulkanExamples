/*
 * Vulkan Example - Indirect drawing
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 *
 * Summary:
 * Use a device local buffer that stores draw commands for instanced rendering of different meshes stored
 * in the same buffer.
 *
 * Indirect drawing offloads draw command generation and offers the ability to update them on the GPU
 * without the CPU having to touch the buffer again, also reducing the number of drawcalls.
 *
 * The example shows how to setup and fill such a buffer on the CPU side, stages it to the device and
 * shows how to render it using only one draw command.
 *
 * See readme.md for details
 *
 */

#include <examples/example.hpp>

#include <shaders/indirectdraw/indirectdraw.vert.inl>
#include <shaders/indirectdraw/indirectdraw.frag.inl>
#include <shaders/indirectdraw/ground.vert.inl>
#include <shaders/indirectdraw/ground.frag.inl>
#include <shaders/indirectdraw/skysphere.vert.inl>
#include <shaders/indirectdraw/skysphere.frag.inl>

// Number of instances per object
#if defined(__ANDROID__)
#define OBJECT_INSTANCE_COUNT 1024
// Circular range of plant distribution
#define PLANT_RADIUS 20.0f
#else
#define OBJECT_INSTANCE_COUNT 2048
// Circular range of plant distribution
#define PLANT_RADIUS 25.0f
#endif

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::texture::Texture2DArray plants;
        vks::texture::Texture2D ground;
    } textures;

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
    });

    struct {
        vks::model::Model plants;
        vks::model::Model ground;
        vks::model::Model skysphere;
    } models;

    // Per-instance data block
    struct InstanceData {
        glm::vec3 pos;
        glm::vec3 rot;
        float scale;
        uint32_t texIndex;
    };

    // Contains the instanced data
    vks::Buffer instanceBuffer;
    // Contains the indirect drawing commands
    vks::Buffer indirectCommandsBuffer;
    uint32_t indirectDrawCount;

    struct {
        glm::mat4 projection;
        glm::mat4 view;
    } uboVS;

    struct {
        vks::Buffer scene;
    } uniformData;

    struct {
        vk::Pipeline plants;
        vk::Pipeline ground;
        vk::Pipeline skysphere;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    vk::Sampler samplerRepeat;

    uint32_t objectCount = 0;

    // Store the indirect draw commands containing index offsets and instance count per object
    std::vector<vk::DrawIndexedIndirectCommand> indirectCommands;

    VulkanExample() {
        title = "Indirect rendering";
        camera.type = Camera::CameraType::firstperson;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-12.0f, 159.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.4f, 1.25f, 0.0f));
        camera.movementSpeed = 5.0f;
        defaultClearColor = vks::util::clearColor({ 0.18f, 0.27f, 0.5f, 0.0f });
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroy(pipelines.plants);
        device.destroy(pipelines.ground);
        device.destroy(pipelines.skysphere);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        models.plants.destroy();
        models.ground.destroy();
        models.skysphere.destroy();
        textures.plants.destroy();
        textures.ground.destroy();
        instanceBuffer.destroy();
        indirectCommandsBuffer.destroy();
        uniformData.scene.destroy();
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        auto& deviceFeatures = context.deviceInfo.features.core10;
        auto& enabledFeatures = context.enabledFeatures.core10;
        // Example uses multi draw indirect if available
        enabledFeatures.multiDrawIndirect = deviceFeatures.multiDrawIndirect;
    };

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());

        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

        // Plants
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.plants);
        // Binding point 0 : Mesh vertex buffer
        drawCmdBuffer.bindVertexBuffers(0, models.plants.vertices.buffer, { 0 });
        // Binding point 1 : Instance data buffer
        drawCmdBuffer.bindVertexBuffers(1, instanceBuffer.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(models.plants.indices.buffer, 0, vk::IndexType::eUint32);

        // If the multi draw feature is supported:
        // One draw call for an arbitrary number of ojects
        // Index offsets and instance count are taken from the indirect buffer
        auto& deviceFeatures = context.deviceInfo.features.core10;
        if (deviceFeatures.multiDrawIndirect) {
            drawCmdBuffer.drawIndexedIndirect(indirectCommandsBuffer.buffer, 0, indirectDrawCount, sizeof(VkDrawIndexedIndirectCommand));
        } else {
            // If multi draw is not available, we must issue separate draw commands
            for (auto j = 0; j < indirectCommands.size(); j++) {
                drawCmdBuffer.drawIndexedIndirect(indirectCommandsBuffer.buffer, j * sizeof(VkDrawIndexedIndirectCommand), 1,
                                                  sizeof(VkDrawIndexedIndirectCommand));
            }
        }

        // Ground
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ground);
        drawCmdBuffer.bindVertexBuffers(0, models.ground.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(models.ground.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(models.ground.indexCount, 1, 0, 0, 0);
        // Skysphere
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skysphere);
        drawCmdBuffer.bindVertexBuffers(0, models.skysphere.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(models.skysphere.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(models.skysphere.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        models.plants.loadFromFile(getAssetPath() + "models/plants.dae", vertexLayout, 0.0025f);
        models.ground.loadFromFile(getAssetPath() + "models/plane_circle.dae", vertexLayout, PLANT_RADIUS + 1.0f);
        models.skysphere.loadFromFile(getAssetPath() + "models/skysphere.dae", vertexLayout, 512.0f / 10.0f);

        // Textures
        std::string texFormatSuffix;
        vk::Format texFormat;
        auto& deviceFeatures = context.deviceInfo.features.core10;
        // Get supported compressed texture format
        if (deviceFeatures.textureCompressionBC) {
            texFormatSuffix = "_bc3_unorm";
            texFormat = vk::Format::eBc3UnormBlock;
        } else if (deviceFeatures.textureCompressionASTC_LDR) {
            texFormatSuffix = "_astc_8x8_unorm";
            texFormat = vk::Format::eAstc8x8UnormBlock;
        } else if (deviceFeatures.textureCompressionETC2) {
            texFormatSuffix = "_etc2_unorm";
            texFormat = vk::Format::eEtc2R8G8B8A8UnormBlock;
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }

        textures.plants.loadFromFile(getAssetPath() + "textures/texturearray_plants" + texFormatSuffix + ".ktx", texFormat);
        textures.ground.loadFromFile(getAssetPath() + "textures/ground_dry" + texFormatSuffix + ".ktx", texFormat);
    }

    void setupDescriptorPool() {
        // Example uses one ubo
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 1 },
            { vk::DescriptorType::eCombinedImageSampler, 2 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        auto plantDesc = textures.plants.makeDescriptor(defaultSampler);
        auto groundDesc = textures.plants.makeDescriptor(defaultSampler);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.scene.descriptor },
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &plantDesc },
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &groundDesc },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout };
        builder.depthStencilState = true;
        builder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        vertexLayout.appendVertexLayout(builder.vertexInputState);

        // Ground
        builder.loadShader(vkx::shaders::indirectdraw::ground::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::indirectdraw::ground::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.ground = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Skysphere
        builder.loadShader(vkx::shaders::indirectdraw::skysphere::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::indirectdraw::skysphere::frag, vk::ShaderStageFlagBits::eFragment);
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        pipelines.skysphere = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Indirect (and instanced) pipeline for the plants
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
        builder.vertexInputState.bindingDescriptions.push_back({ 1, sizeof(InstanceData), vk::VertexInputRate::eInstance });
        builder.vertexInputState.attributeDescriptions.push_back({ 4, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, pos) });
        builder.vertexInputState.attributeDescriptions.push_back({ 5, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, rot) });
        builder.vertexInputState.attributeDescriptions.push_back({ 6, 1, vk::Format::eR32Sfloat, offsetof(InstanceData, scale) });
        builder.vertexInputState.attributeDescriptions.push_back({ 7, 1, vk::Format::eR32Sint, offsetof(InstanceData, texIndex) });
        builder.loadShader(vkx::shaders::indirectdraw::indirectdraw::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::indirectdraw::indirectdraw::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.plants = builder.create(context.pipelineCache);
        builder.destroyShaderModules();
    }

    // Prepare (and stage) a buffer containing the indirect draw commands
    void prepareIndirectData() {
        indirectCommands.clear();

        // Create on indirect command for each mesh in the scene
        uint32_t m = 0;
        for (auto& modelPart : models.plants.parts) {
            VkDrawIndexedIndirectCommand indirectCmd{};
            indirectCmd.instanceCount = OBJECT_INSTANCE_COUNT;
            indirectCmd.firstInstance = m * OBJECT_INSTANCE_COUNT;
            indirectCmd.firstIndex = modelPart.indexBase;
            indirectCmd.indexCount = modelPart.indexCount;

            indirectCommands.push_back(indirectCmd);

            m++;
        }

        objectCount = 0;
        for (auto indirectCmd : indirectCommands) {
            objectCount += indirectCmd.instanceCount;
        }
        indirectDrawCount = static_cast<uint32_t>(indirectCommands.size());
        indirectCommandsBuffer = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndirectBuffer, indirectCommands);
    }

    // Prepare (and stage) a buffer containing instanced data for the mesh draws
    void prepareInstanceData() {
        std::vector<InstanceData> instanceData;
        instanceData.resize(objectCount);

        std::mt19937 rndGenerator((unsigned)time(NULL));
        std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

        for (uint32_t i = 0; i < objectCount; i++) {
            instanceData[i].rot = glm::vec3(0.0f, float(M_PI) * uniformDist(rndGenerator), 0.0f);
            float theta = 2 * float(M_PI) * uniformDist(rndGenerator);
            float phi = acos(1 - 2 * uniformDist(rndGenerator));
            instanceData[i].pos = glm::vec3(sin(phi) * cos(theta), 0.0f, cos(phi)) * PLANT_RADIUS;
            instanceData[i].scale = 1.0f + uniformDist(rndGenerator) * 2.0f;
            instanceData[i].texIndex = i / OBJECT_INSTANCE_COUNT;
        }

        instanceBuffer = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, instanceData);
    }

    void prepareUniformBuffers() {
        uniformData.scene = loader.createUniformBuffer(uboVS);
        updateUniformBuffer(true);
    }

    void updateUniformBuffer(bool viewChanged) {
        if (viewChanged) {
            uboVS.projection = camera.matrices.perspective;
            uboVS.view = camera.matrices.view;
        }

        uniformData.scene.copy(uboVS);
    }

    void prepare() {
        ExampleBase::prepare();
        prepareIndirectData();
        prepareInstanceData();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffer(true); }

    void OnUpdateUIOverlay() override {
        auto& deviceFeatures = context.deviceInfo.features.core10;
        if (!deviceFeatures.multiDrawIndirect) {
            if (ui.header("Info")) {
                ui.text("multiDrawIndirect not supported");
            }
        }
        if (ui.header("Statistics")) {
            ui.text("Objects: %d", objectCount);
        }
    }
};

VULKAN_EXAMPLE_MAIN()
