/*
* Vulkan Example - Physical based shading basics
*
* See http://graphicrants.blogspot.de/2013/08/specular-brdf-reference.html for a good reference to the different functions that make up a specular BRDF
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

#define ENABLE_VALIDATION false
#define GRID_DIM 7
#define OBJ_DIM 0.05f

struct Material {
    // Parameter block used as push constant block
    struct PushBlock {
        float roughness;
        float metallic;
        float r, g, b;
    } params;
    std::string name;
    Material(){};
    Material(std::string n, glm::vec3 c, float r, float m)
        : name(n) {
        params.roughness = r;
        params.metallic = m;
        params.r = c.r;
        params.g = c.g;
        params.b = c.b;
    };
};

class VulkanExample : public vkx::ExampleBase {
public:
    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
    } };

    struct Meshes {
        std::vector<vks::model::Model> objects;
        int32_t objectIndex = 0;
    } models;

    struct {
        vks::Buffer object;
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
    } uboParams;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;

    // Default materials to select from
    std::vector<Material> materials;
    int32_t materialIndex = 0;

    std::vector<std::string> materialNames;
    std::vector<std::string> objectNames;

    VulkanExample() {
        title = "Physical based shading basics";
        camera.type = Camera::CameraType::firstperson;
        camera.setPosition(glm::vec3(10.0f, 13.0f, 1.8f));
        camera.setRotation(glm::vec3(-62.5f, 90.0f, 0.0f));
        camera.movementSpeed = 4.0f;
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        camera.rotationSpeed = 0.25f;
        paused = true;
        timerSpeed *= 0.25f;
        settings.overlay = true;

        // Setup some default materials (source: https://seblagarde.wordpress.com/2011/08/17/feeding-a-physical-based-lighting-mode/)
        materials.push_back(Material("Gold", glm::vec3(1.0f, 0.765557f, 0.336057f), 0.1f, 1.0f));
        materials.push_back(Material("Copper", glm::vec3(0.955008f, 0.637427f, 0.538163f), 0.1f, 1.0f));
        materials.push_back(Material("Chromium", glm::vec3(0.549585f, 0.556114f, 0.554256f), 0.1f, 1.0f));
        materials.push_back(Material("Nickel", glm::vec3(0.659777f, 0.608679f, 0.525649f), 0.1f, 1.0f));
        materials.push_back(Material("Titanium", glm::vec3(0.541931f, 0.496791f, 0.449419f), 0.1f, 1.0f));
        materials.push_back(Material("Cobalt", glm::vec3(0.662124f, 0.654864f, 0.633732f), 0.1f, 1.0f));
        materials.push_back(Material("Platinum", glm::vec3(0.672411f, 0.637331f, 0.585456f), 0.1f, 1.0f));
        // Testing materials
        materials.push_back(Material("White", glm::vec3(1.0f), 0.1f, 1.0f));
        materials.push_back(Material("Red", glm::vec3(1.0f, 0.0f, 0.0f), 0.1f, 1.0f));
        materials.push_back(Material("Blue", glm::vec3(0.0f, 0.0f, 1.0f), 0.1f, 1.0f));
        materials.push_back(Material("Black", glm::vec3(0.0f), 0.1f, 1.0f));

        for (auto material : materials) {
            materialNames.push_back(material.name);
        }
        objectNames = { "Sphere", "Teapot", "Torusknot", "Venus" };

        materialIndex = 0;
    }

    ~VulkanExample() {
        vkDestroyPipeline(device, pipeline, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        for (auto& model : models.objects) {
            model.destroy();
        }

        uniformBuffers.object.destroy();
        uniformBuffers.params.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, viewport());
        cmdBuffer.setScissor(0, scissor());

        std::vector<vk::DeviceSize> offsets{ 0 };

        // Objects
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, models.objects[models.objectIndex].vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(models.objects[models.objectIndex].indices.buffer, 0, vk::IndexType::eUint32);

        Material mat = materials[materialIndex];

        //#define SINGLE_ROW 1
#ifdef SINGLE_ROW
        mat.params.metallic = 1.0;

        uint32_t objcount = 10;
        for (uint32_t x = 0; x < objcount; x++) {
            glm::vec3 pos = glm::vec3(float(x - (objcount / 2.0f)) * 2.5f, 0.0f, 0.0f);
            mat.params.roughness = glm::clamp((float)x / (float)objcount, 0.005f, 1.0f);
            vkCmdPushConstants(drawCmdBuffers[i], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::vec3), &pos);
            vkCmdPushConstants(drawCmdBuffers[i], pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::vec3), sizeof(Material::PushBlock), &mat);
            vkCmdDrawIndexed(drawCmdBuffers[i], models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
        }
#else
        for (uint32_t y = 0; y < GRID_DIM; y++) {
            for (uint32_t x = 0; x < GRID_DIM; x++) {
                glm::vec3 pos = glm::vec3(float(x - (GRID_DIM / 2.0f)) * 2.5f, 0.0f, float(y - (GRID_DIM / 2.0f)) * 2.5f);
                cmdBuffer.pushConstants<glm::vec3>(pipelineLayout, vSS::eVertex, 0, pos);
                mat.params.metallic = glm::clamp((float)x / (float)(GRID_DIM - 1), 0.1f, 1.0f);
                mat.params.roughness = glm::clamp((float)y / (float)(GRID_DIM - 1), 0.05f, 1.0f);
                cmdBuffer.pushConstants<Material::PushBlock>(pipelineLayout, vSS::eFragment, sizeof(glm::vec3), mat.params);
                cmdBuffer.drawIndexed(models.objects[models.objectIndex].indexCount, 1, 0, 0, 0);
            }
        }
#endif
    }
#if 0
    void buildCommandBuffers() {
        vk::CommandBufferBeginInfo cmdBufInfo;

        vk::ClearValue clearValues[2];
        clearValues[0].color = defaultClearColor;
        clearValues[1].depthStencil = { 1.0f, 0 };

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent = size;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = framebuffers[i];

            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }
#endif
    void loadAssets() override {
        // Objects
        std::vector<std::string> filenames = { "geosphere.obj", "teapot.dae", "torusknot.obj", "venus.fbx" };
        auto modelCount = filenames.size();
        models.objects.resize(modelCount);
        for (size_t i = 0; i < modelCount; ++i) {
            const auto& file = filenames[i];
            auto& model = models.objects[i];
            model.loadFromFile(context, getAssetPath() + "models/" + file, vertexLayout, OBJ_DIM * (file == "venus.fbx" ? 3.0f : 1.0f));
        }
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vDT::eUniformBuffer, 1, vSS::eVertex | vSS::eFragment },
            { 1, vDT::eUniformBuffer, 1, vSS::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        std::vector<vk::PushConstantRange> pushConstantRanges = {
            { vSS::eVertex, 0, sizeof(glm::vec3) },
            { vSS::eFragment, sizeof(glm::vec3), sizeof(Material::PushBlock) },
        };

        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout, (uint32_t)pushConstantRanges.size(), pushConstantRanges.data() });
    }

    void setupDescriptorSets() {
        // Descriptor Pool
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vDT::eUniformBuffer, 4 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });

        // Descriptor sets
        // 3D object descriptor set
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            { descriptorSet, 0, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.object.descriptor },
            { descriptorSet, 1, 0, 1, vDT::eUniformBuffer, nullptr, &uniformBuffers.params.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbrbasic/pbr.vert.spv", vSS::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/pbrbasic/pbr.frag.spv", vSS::eFragment);
        pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Objact vertex shader uniform buffer
        uniformBuffers.object = context.createUniformBuffer(uboMatrices);
        // Shared parameter uniform buffer
        uniformBuffers.params = context.createUniformBuffer(uboParams);
        updateUniformBuffers();
        updateLights();
    }

    void updateUniformBuffers() {
        // 3D object
        uboMatrices.projection = camera.matrices.perspective;
        uboMatrices.view = camera.matrices.view;
        uboMatrices.model = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f + (models.objectIndex == 1 ? 45.0f : 0.0f)), glm::vec3(0.0f, 1.0f, 0.0f));
        uboMatrices.camPos = camera.position * -1.0f;
        memcpy(uniformBuffers.object.mapped, &uboMatrices, sizeof(uboMatrices));
    }

    void updateLights() {
        const float p = 15.0f;
        uboParams.lights[0] = glm::vec4(-p, -p * 0.5f, -p, 1.0f);
        uboParams.lights[1] = glm::vec4(-p, -p * 0.5f, p, 1.0f);
        uboParams.lights[2] = glm::vec4(p, -p * 0.5f, p, 1.0f);
        uboParams.lights[3] = glm::vec4(p, -p * 0.5f, -p, 1.0f);

        if (!paused) {
            uboParams.lights[0].x = sin(glm::radians(timer * 360.0f)) * 20.0f;
            uboParams.lights[0].z = cos(glm::radians(timer * 360.0f)) * 20.0f;
            uboParams.lights[1].x = cos(glm::radians(timer * 360.0f)) * 20.0f;
            uboParams.lights[1].y = sin(glm::radians(timer * 360.0f)) * 20.0f;
        }

        memcpy(uniformBuffers.params.mapped, &uboParams, sizeof(uboParams));
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorSets();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared) {
            return;
        }
        draw();
        if (!paused) {
            updateLights();
        }
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
        }
    }
};

VULKAN_EXAMPLE_MAIN()
