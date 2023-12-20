/*
* Vulkan Example - Physical based rendering a textured object (metal/roughness workflow) with image based lighting
*
* Note: Requires the separate asset pack (see data/README.md)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

// For reference see http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

#include <vulkanExampleBase.h>
#include <pbr.hpp>

class VulkanExample : public vkx::ExampleBase {
public:
    bool displaySkybox = true;

    struct Textures {
        vks::texture::TextureCubeMap environmentCube;
        // Generated at runtime
        vks::texture::Texture2D lutBrdf;
        vks::texture::TextureCubeMap irradianceCube;
        vks::texture::TextureCubeMap prefilteredCube;
        // Object texture maps
        vks::texture::Texture2D albedoMap;
        vks::texture::Texture2D normalMap;
        vks::texture::Texture2D aoMap;
        vks::texture::Texture2D metallicMap;
        vks::texture::Texture2D roughnessMap;
    } textures;

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
    });

    struct Meshes {
        vks::model::Model skybox;
        vks::model::Model object;
    } models;

    struct {
        vks::Buffer object;
        vks::Buffer skybox;
        vks::Buffer params;
    } uniformBuffers;

    struct UBOMatrices {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
        glm::vec3 camPos;
    } uboMatrices;

    struct UBOParams {
        glm::vec4 lights[4];
        float exposure = 4.5f;
        float gamma = 2.2f;
    } uboParams;

    struct {
        vk::Pipeline skybox;
        vk::Pipeline pbr;
    } pipelines;

    struct {
        vk::DescriptorSet object;
        vk::DescriptorSet skybox;
    } descriptorSets;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        title = "Textured PBR with IBL";

        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 4.0f;
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        camera.rotationSpeed = 0.25f;

        camera.setRotation({ -10.75f, 153.0f, 0.0f });
        camera.setPosition({ 1.85f, 0.5f, 5.0f });

        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroyPipeline(pipelines.skybox);
        device.destroyPipeline(pipelines.pbr);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        models.object.destroy();
        models.skybox.destroy();

        uniformBuffers.object.destroy();
        uniformBuffers.skybox.destroy();
        uniformBuffers.params.destroy();

        textures.environmentCube.destroy();
        textures.irradianceCube.destroy();
        textures.prefilteredCube.destroy();
        textures.lutBrdf.destroy();
        textures.albedoMap.destroy();
        textures.normalMap.destroy();
        textures.aoMap.destroy();
        textures.metallicMap.destroy();
        textures.roughnessMap.destroy();
    }

    void getEnabledFeatures() override {
        if (context.deviceFeatures.samplerAnisotropy) {
            context.enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuf) override {
        cmdBuf.setViewport(0, viewport());
        cmdBuf.setScissor(0, scissor());

        std::vector<vk::DeviceSize> offsets{ 0 };

        // Skybox
        if (displaySkybox) {
            cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.skybox, nullptr);
            cmdBuf.bindVertexBuffers(0, models.skybox.vertices.buffer, offsets);
            cmdBuf.bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
            cmdBuf.drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
        }

        // Objects
        cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.object, nullptr);
        cmdBuf.bindVertexBuffers(0, models.object.vertices.buffer, offsets);
        cmdBuf.bindIndexBuffer(models.object.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.pbr);

        cmdBuf.drawIndexed(models.object.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        textures.environmentCube.loadFromFile(context, getAssetPath() + "textures/hdr/gcanyon_cube.ktx", vF::eR16G16B16A16Sfloat);
        models.skybox.loadFromFile(context, getAssetPath() + "models/cube.obj", vertexLayout, 1.0f);
        // PBR model
        models.object.loadFromFile(context, getAssetPath() + "models/cerberus/cerberus.fbx", vertexLayout, 0.05f);
        textures.albedoMap.loadFromFile(context, getAssetPath() + "models/cerberus/albedo.ktx", vF::eR8G8B8A8Unorm);
        textures.normalMap.loadFromFile(context, getAssetPath() + "models/cerberus/normal.ktx", vF::eR8G8B8A8Unorm);
        textures.aoMap.loadFromFile(context, getAssetPath() + "models/cerberus/ao.ktx", vF::eR8Unorm);
        textures.metallicMap.loadFromFile(context, getAssetPath() + "models/cerberus/metallic.ktx", vF::eR8Unorm);
        textures.roughnessMap.loadFromFile(context, getAssetPath() + "models/cerberus/roughness.ktx", vF::eR8Unorm);
    }

    void setupDescriptors() {
        // Descriptor Pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vDT::eUniformBuffer, 4 },
            { vDT::eCombinedImageSampler, 16 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });

        // Descriptor set layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vDT::eUniformBuffer, 1, vSS::eVertex | vSS::eFragment }, { 1, vDT::eUniformBuffer, 1, vSS::eFragment },
            { 2, vDT::eCombinedImageSampler, 1, vSS::eFragment },         { 3, vDT::eCombinedImageSampler, 1, vSS::eFragment },
            { 4, vDT::eCombinedImageSampler, 1, vSS::eFragment },         { 5, vDT::eCombinedImageSampler, 1, vSS::eFragment },
            { 6, vDT::eCombinedImageSampler, 1, vSS::eFragment },         { 7, vDT::eCombinedImageSampler, 1, vSS::eFragment },
            { 8, vDT::eCombinedImageSampler, 1, vSS::eFragment },         { 9, vDT::eCombinedImageSampler, 1, vSS::eFragment },
        };
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        // Descriptor sets
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        // Objects
        descriptorSets.object = device.allocateDescriptorSets(allocInfo)[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            { descriptorSets.object, 0, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.object.descriptor },
            { descriptorSets.object, 1, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            { descriptorSets.object, 2, 0, 1, vDT::eCombinedImageSampler, &textures.irradianceCube.descriptor },
            { descriptorSets.object, 3, 0, 1, vDT::eCombinedImageSampler, &textures.lutBrdf.descriptor },
            { descriptorSets.object, 4, 0, 1, vDT::eCombinedImageSampler, &textures.prefilteredCube.descriptor },
            { descriptorSets.object, 5, 0, 1, vDT::eCombinedImageSampler, &textures.albedoMap.descriptor },
            { descriptorSets.object, 6, 0, 1, vDT::eCombinedImageSampler, &textures.normalMap.descriptor },
            { descriptorSets.object, 7, 0, 1, vDT::eCombinedImageSampler, &textures.aoMap.descriptor },
            { descriptorSets.object, 8, 0, 1, vDT::eCombinedImageSampler, &textures.metallicMap.descriptor },
            { descriptorSets.object, 9, 0, 1, vDT::eCombinedImageSampler, &textures.roughnessMap.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Sky box
        descriptorSets.skybox = device.allocateDescriptorSets(allocInfo)[0];
        writeDescriptorSets = {
            { descriptorSets.skybox, 0, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.skybox.descriptor },
            { descriptorSets.skybox, 1, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            { descriptorSets.skybox, 2, 0, 1, vDT::eCombinedImageSampler, &textures.environmentCube.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Pipeline layout
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });

        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.depthStencilState = { false };
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);

        // Skybox pipeline (background cube)
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbrtexture/skybox.vert.spv", vSS::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbrtexture/skybox.frag.spv", vSS::eFragment);
        pipelines.skybox = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // PBR pipeline
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbrtexture/pbrtexture.vert.spv", vSS::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbrtexture/pbrtexture.frag.spv", vSS::eFragment);
        // Enable depth test and write
        pipelineBuilder.depthStencilState = { true };
        pipelines.pbr = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Objact vertex shader uniform buffer
        uniformBuffers.object = context.createUniformBuffer(uboMatrices);
        // Skybox vertex shader uniform buffer
        uniformBuffers.skybox = context.createUniformBuffer(uboMatrices);
        // Shared parameter uniform buffer
        uniformBuffers.params = context.createUniformBuffer(uboParams);
        updateUniformBuffers();
        updateParams();
    }

    void updateUniformBuffers() {
        // 3D object
        uboMatrices.projection = camera.matrices.perspective;
        uboMatrices.view = camera.matrices.view;
        uboMatrices.model = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        uboMatrices.camPos = camera.position * -1.0f;
        memcpy(uniformBuffers.object.mapped, &uboMatrices, sizeof(uboMatrices));

        // Skybox
        uboMatrices.model = glm::mat4(glm::mat3(camera.matrices.view));
        memcpy(uniformBuffers.skybox.mapped, &uboMatrices, sizeof(uboMatrices));
    }

    void updateParams() {
        const float p = 15.0f;
        uboParams.lights[0] = glm::vec4(-p, -p * 0.5f, -p, 1.0f);
        uboParams.lights[1] = glm::vec4(-p, -p * 0.5f, p, 1.0f);
        uboParams.lights[2] = glm::vec4(p, -p * 0.5f, p, 1.0f);
        uboParams.lights[3] = glm::vec4(p, -p * 0.5f, -p, 1.0f);

        memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams));
    }

    void prepare() override {
        ExampleBase::prepare();
        vkx::pbr::generateBRDFLUT(context, textures.lutBrdf);
        vkx::pbr::generateIrradianceCube(context, textures.irradianceCube, models.skybox, vertexLayout, textures.environmentCube.descriptor);
        vkx::pbr::generatePrefilteredCube(context, textures.prefilteredCube, models.skybox, vertexLayout, textures.environmentCube.descriptor);
        prepareUniformBuffers();
        setupDescriptors();
        preparePipelines();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.inputFloat("Exposure", &uboParams.exposure, 0.1f, "%.2f")) {
                updateParams();
            }
            if (ui.inputFloat("Gamma", &uboParams.gamma, 0.1f, "%.2f")) {
                updateParams();
            }
            if (ui.checkBox("Skybox", &displaySkybox)) {
                buildCommandBuffers();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
