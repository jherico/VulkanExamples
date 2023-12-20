/*
* Vulkan Example - Physical based rendering with image based lighting
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

#define GRID_DIM 7

struct Material {
    // Parameter block used as push constant block
    struct PushBlock {
        float roughness = 0.0f;
        float metallic = 0.0f;
        float specular = 0.0f;
        float r, g, b;
    } params;
    std::string name;
    Material(){};
    Material(std::string n, glm::vec3 c)
        : name(n) {
        params.r = c.r;
        params.g = c.g;
        params.b = c.b;
    };
};

// Vertex layout for the models
vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_UV,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    bool displaySkybox = true;

    struct Textures {
        vks::texture::TextureCubeMap environmentCube;
        // Generated at runtime
        vks::texture::Texture2D lutBrdf;
        vks::texture::TextureCubeMap irradianceCube;
        vks::texture::TextureCubeMap prefilteredCube;
    } textures;

    struct Meshes {
        vks::model::Model skybox;
        std::vector<vks::model::Model> objects;
        int32_t objectIndex = 0;
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

    // Default materials to select from
    std::vector<Material> materials;
    int32_t materialIndex = 0;

    std::vector<std::string> materialNames;
    std::vector<std::string> objectNames;

    VulkanExample() {
        title = "PBR with image based lighting";

        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 4.0f;
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        camera.rotationSpeed = 0.25f;

        camera.setRotation({ -3.75f, 180.0f, 0.0f });
        camera.setPosition({ 0.55f, 0.85f, 12.0f });

        // Setup some default materials (source: https://seblagarde.wordpress.com/2011/08/17/feeding-a-physical-based-lighting-mode/)
        materials.push_back(Material("Gold", glm::vec3(1.0f, 0.765557f, 0.336057f)));
        materials.push_back(Material("Copper", glm::vec3(0.955008f, 0.637427f, 0.538163f)));
        materials.push_back(Material("Chromium", glm::vec3(0.549585f, 0.556114f, 0.554256f)));
        materials.push_back(Material("Nickel", glm::vec3(0.659777f, 0.608679f, 0.525649f)));
        materials.push_back(Material("Titanium", glm::vec3(0.541931f, 0.496791f, 0.449419f)));
        materials.push_back(Material("Cobalt", glm::vec3(0.662124f, 0.654864f, 0.633732f)));
        materials.push_back(Material("Platinum", glm::vec3(0.672411f, 0.637331f, 0.585456f)));
        // Testing materials
        materials.push_back(Material("White", glm::vec3(1.0f)));
        materials.push_back(Material("Dark", glm::vec3(0.1f)));
        materials.push_back(Material("Black", glm::vec3(0.0f)));
        materials.push_back(Material("Red", glm::vec3(1.0f, 0.0f, 0.0f)));
        materials.push_back(Material("Blue", glm::vec3(0.0f, 0.0f, 1.0f)));

        settings.overlay = true;

        for (auto material : materials) {
            materialNames.push_back(material.name);
        }
        objectNames = { "Sphere", "Teapot", "Torusknot", "Venus" };

        materialIndex = 9;
    }

    ~VulkanExample() {
        device.destroyPipeline(pipelines.skybox, nullptr);
        device.destroyPipeline(pipelines.pbr, nullptr);

        device.destroyPipelineLayout(pipelineLayout, nullptr);
        device.destroyDescriptorSetLayout(descriptorSetLayout, nullptr);

        for (auto& model : models.objects) {
            model.destroy();
        }
        models.skybox.destroy();

        uniformBuffers.object.destroy();
        uniformBuffers.skybox.destroy();
        uniformBuffers.params.destroy();

        textures.environmentCube.destroy();
        textures.irradianceCube.destroy();
        textures.prefilteredCube.destroy();
        textures.lutBrdf.destroy();
    }

    void getEnabledFeatures() override {
        if (context.deviceFeatures.samplerAnisotropy) {
            context.enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) override {
        commandBuffer.setViewport(0, viewport());
        commandBuffer.setScissor(0, scissor());
        vk::DeviceSize offsets[1] = { 0 };

        // Skybox
        if (displaySkybox) {
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets.skybox, 0, NULL);
            commandBuffer.bindVertexBuffers(0, 1, &models.skybox.vertices.buffer, offsets);
            commandBuffer.bindIndexBuffer(models.skybox.indices.buffer, 0, vk::IndexType::eUint32);
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
            commandBuffer.drawIndexed(models.skybox.indexCount, 1, 0, 0, 0);
        }

        // Objects
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets.object, 0, NULL);
        commandBuffer.bindVertexBuffers(0, 1, &models.objects[models.objectIndex].vertices.buffer, offsets);
        commandBuffer.bindIndexBuffer(models.objects[models.objectIndex].indices.buffer, 0, vk::IndexType::eUint32);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.pbr);

        Material mat = materials[materialIndex];

#define SINGLE_ROW 1
#ifdef SINGLE_ROW
        uint32_t objcount = 10;
        for (uint32_t x = 0; x < objcount; x++) {
            glm::vec3 pos = glm::vec3(float(x - (objcount / 2.0f)) * 2.15f, 0.0f, 0.0f);
            mat.params.roughness = 1.0f - glm::clamp((float)x / (float)objcount, 0.005f, 1.0f);
            mat.params.metallic = glm::clamp((float)x / (float)objcount, 0.005f, 1.0f);
            commandBuffer.pushConstants<glm::vec3>(pipelineLayout, vSS::eVertex, 0, pos);
            commandBuffer.pushConstants<Material::PushBlock>(pipelineLayout, vSS::eFragment, sizeof(glm::vec3), mat.params);
            commandBuffer.drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
        }
#else
        for (uint32_t y = 0; y < GRID_DIM; y++) {
            mat.params.metallic = (float)y / (float)(GRID_DIM);
            for (uint32_t x = 0; x < GRID_DIM; x++) {
                glm::vec3 pos = glm::vec3(float(x - (GRID_DIM / 2.0f)) * 2.5f, 0.0f, float(y - (GRID_DIM / 2.0f)) * 2.5f);
                mat.params.roughness = glm::clamp((float)x / (float)(GRID_DIM), 0.05f, 1.0f);
                commandBuffer.pushConstants<glm::vec3>(pipelineLayout, vSS::eVertex, 0, pos);
                commandBuffer.pushConstants<Material::PushBlock>(pipelineLayout, vSS::eFragment, sizeof(glm::vec3), mat.params);
                commandBuffer.drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
            }
        }
#endif
    }

    void loadAssets() override {
        textures.environmentCube.loadFromFile(context, getAssetPath() + "textures/hdr/pisa_cube.ktx", vF::eR16G16B16A16Sfloat);
        // Skybox
        models.skybox.loadFromFile(context, getAssetPath() + "models/cube.obj", vertexLayout, 1.0f);
        // Objects
        const std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
        auto modelCount = filenames.size();
        models.objects.resize(modelCount);
        for (size_t i = 0; i < modelCount; ++i) {
            auto& model = models.objects[i];
            const auto& file = filenames[i];
            model.loadFromFile(context, getAssetPath() + "models/" + file, vertexLayout, 0.05f * (file == "venus.fbx" ? 3.0f : 1.0f));
        }
    }

    void setupDescriptors() {
        // Descriptor Pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vDT::eUniformBuffer, 4 },
            { vDT::eCombinedImageSampler, 6 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });

        // Descriptor set layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vDT::eUniformBuffer, 1, vSS::eVertex | vSS::eFragment }, { 1, vDT::eUniformBuffer, 1, vSS::eFragment },
            { 2, vDT::eCombinedImageSampler, 1, vSS::eFragment },         { 3, vDT::eCombinedImageSampler, 1, vSS::eFragment },
            { 4, vDT::eCombinedImageSampler, 1, vSS::eFragment },
        };
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        // Descriptor sets
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        // Objects
        descriptorSets.object = device.allocateDescriptorSets(allocInfo)[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSets.object, 0, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.object.descriptor },
            { descriptorSets.object, 1, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
            { descriptorSets.object, 2, 0, 1, vDT::eCombinedImageSampler, &textures.irradianceCube.descriptor },
            { descriptorSets.object, 3, 0, 1, vDT::eCombinedImageSampler, &textures.lutBrdf.descriptor },
            { descriptorSets.object, 4, 0, 1, vDT::eCombinedImageSampler, &textures.prefilteredCube.descriptor },
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
        // Push constant ranges
        std::vector<vk::PushConstantRange> pushConstantRanges{
            { vSS::eVertex, 0, sizeof(glm::vec3) },
            { vSS::eFragment, sizeof(glm::vec3), sizeof(Material::PushBlock) },
        };
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout, (uint32_t)pushConstantRanges.size(), pushConstantRanges.data() });

        // Pipelines
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.depthStencilState = { false };
        // Vertex bindings and attributes
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        // Skybox pipeline (background cube)
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbribl/skybox.vert.spv", vSS::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbribl/skybox.frag.spv", vSS::eFragment);
        pipelines.skybox = pipelineBuilder.create(context.pipelineCache);

        pipelineBuilder.destroyShaderModules();

        // PBR pipeline
        // Enable depth test and write
        pipelineBuilder.depthStencilState = { true };
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbribl/pbribl.vert.spv", vSS::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbribl/pbribl.frag.spv", vSS::eFragment);
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
        uboMatrices.model = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f + (models.objectIndex == 1 ? 45.0f : 0.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
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
            if (ui.comboBox("Material", &materialIndex, materialNames)) {
                buildCommandBuffers();
            }
            if (ui.comboBox("Object type", &models.objectIndex, objectNames)) {
                updateUniformBuffers();
                buildCommandBuffers();
            }
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
