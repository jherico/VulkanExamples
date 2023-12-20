/*
* Vulkan Example - Shader specialization constants
*
* For details see https://www.khronos.org/registry/vulkan/specs/misc/GL_KHR_vulkan_glsl.txt
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase {
public:
    // Vertex layout for the models
    const vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
    } };

    struct {
        vks::model::Model cube;
    } models;

    struct {
        vks::texture::Texture2D colormap;
    } textures;

    vks::Buffer uniformBuffer;

    // Same uniform buffer layout as shader
    struct UBOVS {
        glm::mat4 projection;
        glm::mat4 modelView;
        glm::vec4 lightPos = glm::vec4(0.0f, -2.0f, 1.0f, 0.0f);
    } uboVS;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    struct {
        vk::Pipeline phong;
        vk::Pipeline toon;
        vk::Pipeline textured;
    } pipelines;

    VulkanExample() {
        title = "Specialization constants";
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, ((float)size.width / 3.0f) / (float)size.height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-40.0f, -90.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroy(pipelines.phong);
        device.destroy(pipelines.textured);
        device.destroy(pipelines.toon);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        models.cube.destroy();
        textures.colormap.destroy();
        uniformBuffer.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        vk::Rect2D scissor;
        scissor.extent = size;
        drawCmdBuffer.setScissor(0, scissor);

        vk::Viewport viewport;
        viewport.width = (float)size.width / 3.0f;
        viewport.height = (float)size.height;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;

        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, { descriptorSet }, {});
        drawCmdBuffer.bindVertexBuffers(0, models.cube.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(models.cube.indices.buffer, 0, vk::IndexType::eUint32);

        // Left
        viewport.x = 0;
        drawCmdBuffer.setViewport(0, viewport);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.phong);
        drawCmdBuffer.drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

        // Center
        viewport.x = (float)size.width / 3.0f;
        drawCmdBuffer.setViewport(0, viewport);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.toon);
        drawCmdBuffer.drawIndexed(models.cube.indexCount, 1, 0, 0, 0);

        // Right
        viewport.x += (float)size.width / 3.0f;
        drawCmdBuffer.setViewport(0, viewport);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.textured);
        drawCmdBuffer.drawIndexed(models.cube.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        models.cube.loadFromFile(context, getAssetPath() + "models/color_teapot_spheres.dae", vertexLayout, 0.1f);
        textures.colormap.loadFromFile(context, getAssetPath() + "textures/metalplate_nomips_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.colormap.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        builder.dynamicState.dynamicStateEnables.push_back(vk::DynamicState::eLineWidth);
        builder.vertexInputState.appendVertexLayout(vertexLayout);

        // Prepare specialization data

        // Host data to take specialization constants from
        struct SpecializationData {
            // Sets the lighting model used in the fragment "uber" shader
            uint32_t lightingModel;
            // Parameter for the toon shading part of the fragment shader
            float toonDesaturationFactor = 0.5f;
        } specializationData;

        // Each shader constant of a shader stage corresponds to one map entry
        std::array<vk::SpecializationMapEntry, 2> specializationMapEntries;
        // Shader bindings based on specialization constants are marked by the new "constant_id" layout qualifier:
        //	layout (constant_id = 0) const int LIGHTING_MODEL = 0;
        //	layout (constant_id = 1) const float PARAM_TOON_DESATURATION = 0.0f;

        // Map entry for the lighting model to be used by the fragment shader
        specializationMapEntries[0].constantID = 0;
        specializationMapEntries[0].size = sizeof(specializationData.lightingModel);
        specializationMapEntries[0].offset = 0;

        // Map entry for the toon shader parameter
        specializationMapEntries[1].constantID = 1;
        specializationMapEntries[1].size = sizeof(specializationData.toonDesaturationFactor);
        specializationMapEntries[1].offset = offsetof(SpecializationData, toonDesaturationFactor);

        // Prepare specialization info block for the shader stage
        vk::SpecializationInfo specializationInfo{};
        specializationInfo.dataSize = sizeof(specializationData);
        specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
        specializationInfo.pMapEntries = specializationMapEntries.data();
        specializationInfo.pData = &specializationData;

        // Create pipelines
        // All pipelines will use the same "uber" shader and specialization constants to change branching and parameters of that shader
        builder.loadShader(getAssetPath() + "shaders/specializationconstants/uber.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/specializationconstants/uber.frag.spv", vk::ShaderStageFlagBits::eFragment);

        // Specialization info is assigned is part of the shader stage (modul) and must be set after creating the module and before creating the pipeline
        builder.shaderStages[1].pSpecializationInfo = &specializationInfo;

        // Solid phong shading
        specializationData.lightingModel = 0;
        pipelines.phong = builder.create(context.pipelineCache);

        // Phong and textured
        specializationData.lightingModel = 1;
        pipelines.toon = builder.create(context.pipelineCache);

        // Textured discard
        specializationData.lightingModel = 2;
        pipelines.textured = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        uniformBuffer = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.modelView = camera.matrices.view;

        memcpy(uniformBuffer.mapped, &uboVS, sizeof(uboVS));
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

VULKAN_EXAMPLE_MAIN()
