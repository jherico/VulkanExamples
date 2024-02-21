/*
 * Vulkan Demo Scene
 *
 * Don't take this a an example, it's more of a personal playground
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * Note : Different license than the other examples!
 *
 * This code is licensed under the Mozilla Public License Version 2.0 (http://opensource.org/licenses/MPL-2.0)
 */

#include <examples/example.hpp>
#include <rendering/model.hpp>
#include <shaders/vulkanscene/logo.frag.inl>
#include <shaders/vulkanscene/logo.vert.inl>
#include <shaders/vulkanscene/mesh.frag.inl>
#include <shaders/vulkanscene/mesh.vert.inl>
#include <shaders/vulkanscene/skybox.frag.inl>
#include <shaders/vulkanscene/skybox.vert.inl>

static std::vector<std::string> names{ "logos", "background", "models", "skybox" };

class VulkanExample : public vkx::ExampleBase {
public:
    using DemoMesh = std::pair<vks::model::Model, vk::Pipeline>;

    struct DemoMeshes {
        DemoMesh logos;
        DemoMesh background;
        DemoMesh models;
        DemoMesh skybox;
    } demoMeshes;

    struct {
        vks::Buffer meshVS;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 normal;
        glm::mat4 view;
        glm::vec4 lightPos;
    } uboVS;

    struct {
        vks::texture::TextureCubeMap skybox;
    } textures;

    struct {
        vk::Pipeline logos;
        vk::Pipeline models;
        vk::Pipeline skybox;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    glm::vec4 lightPos = glm::vec4(1.0f, 2.0f, 0.0f, 0.0f);

    VulkanExample() {
        size.width = 1280;
        size.height = 720;
        rotationSpeed = 0.5f;
        camera.setRotation({ 15.0f, 0.f, 0.0f });
        title = "Vulkan Demo Scene - � 2016 by Sascha Willems";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.logos);
        device.destroy(pipelines.models);
        device.destroy(pipelines.skybox);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        uniformData.meshVS.destroy();

        demoMeshes.logos.first.destroy();
        demoMeshes.background.first.destroy();
        demoMeshes.models.first.destroy();
        demoMeshes.skybox.first.destroy();

        textures.skybox.destroy();
    }

    void loadTextures() { textures.skybox.loadFromFile(getAssetPath() + "textures/cubemap_vulkan.ktx", vk::Format::eR8G8B8A8Unorm); }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        std::array<DemoMesh*, 4> meshes{ {
            &demoMeshes.skybox,
            &demoMeshes.background,
            &demoMeshes.logos,
            &demoMeshes.models,
        } };
        for (auto& meshPtr : meshes) {
            const auto& pipeline = meshPtr->second;
            const auto& mesh = meshPtr->first;
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            cmdBuffer.bindVertexBuffers(0, mesh.vertices.buffer, { 0 });
            cmdBuffer.bindIndexBuffer(mesh.indices.buffer, 0, vk::IndexType::eUint32);
            cmdBuffer.drawIndexed(mesh.indexCount, 1, 0, 0, 0);
        }
    }

    vks::model::VertexLayout vertexLayout{ {
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
    } };

    void prepareVertices() {
        struct Vertex {
            float pos[3];
            float normal[3];
            float uv[2];
            float color[3];
        };

        // Load meshes for demos scene

        demoMeshes.logos.first.loadFromFile(getAssetPath() + "models/vulkanscenelogos.dae", vertexLayout);
        demoMeshes.background.first.loadFromFile(getAssetPath() + "models/vulkanscenebackground.dae", vertexLayout);
        demoMeshes.models.first.loadFromFile(getAssetPath() + "models/vulkanscenemodels.dae", vertexLayout);
        demoMeshes.skybox.first.loadFromFile(getAssetPath() + "models/cube.obj", vertexLayout);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 2 },
            { vk::DescriptorType::eCombinedImageSampler, 1 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{ { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
                                                                       { 1, vk::DescriptorType::eCombinedImageSampler, 1,
                                                                         vk::ShaderStageFlagBits::eFragment } };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Cube map image descriptor
        vk::DescriptorImageInfo texDescriptorCubeMap = textures.skybox.makeDescriptor(defaultSampler);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.meshVS.descriptor },
            // Binding 1 : Fragment shader image sampler
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorCubeMap },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

        // Binding description
        pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, vertexLayout.stride(), vk::VertexInputRate::eVertex } };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Location 0 : Position
            { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
            // Location 1 : Normal
            { 1, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3 },
            // Location 2 : Texture coordinates
            { 2, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 6 },
            // Location 3 : Color
            { 3, 0, vk::Format::eR32G32B32Sfloat, sizeof(float) * 8 },
        };

        // Attribute descriptions

        // vk::Pipeline for the meshes (armadillo, bunny, etc.)
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::vulkanscene::mesh::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::vulkanscene::mesh::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.models = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // vk::Pipeline for the logos
        pipelineBuilder.loadShader(vkx::shaders::vulkanscene::logo::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::vulkanscene::logo::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.logos = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // vk::Pipeline for the sky sphere (todo)
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;  // Inverted culling
        pipelineBuilder.depthStencilState.depthWriteEnable = VK_FALSE;               // No depth writes
        pipelineBuilder.loadShader(vkx::shaders::vulkanscene::skybox::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::vulkanscene::skybox::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.vertexInputState.attributeDescriptions.resize(1);
        pipelines.skybox = pipelineBuilder.create(context.pipelineCache);

        // Assign pipelines
        demoMeshes.logos.second = pipelines.logos;
        demoMeshes.models.second = pipelines.models;
        demoMeshes.background.second = pipelines.models;
        demoMeshes.skybox.second = pipelines.skybox;
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.meshVS = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = getProjection();
        uboVS.view = glm::translate(glm::mat4(), glm::vec3(0, 0, camera.position.z));
        uboVS.model = camera.matrices.view;
        uboVS.model[3] = vec4{ 0, 0, 0, 1 };
        uboVS.normal = glm::inverseTranspose(uboVS.view * uboVS.model);
        uboVS.lightPos = lightPos;
        uniformData.meshVS.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadTextures();
        prepareVertices();
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

RUN_EXAMPLE(VulkanExample)
