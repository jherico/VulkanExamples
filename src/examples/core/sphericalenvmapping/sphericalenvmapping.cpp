/*
 * Vulkan Example - Spherical Environment Mapping, using different mat caps
 *
 * Use +/-/space toggle through different material captures
 *
 * Based on https://www.clicktorelease.com/blog/creating-spherical-environment-mapping-shader
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <rendering/model.hpp>

#include <shaders/sphericalenvmapping/sem.frag.inl>
#include <shaders/sphericalenvmapping/sem.vert.inl>

class VulkanExample : public vkx::ExampleBase {
public:
    vks::model::VertexLayout vertexLayout{ {
        vks::model::Component::VERTEX_COMPONENT_POSITION,
        vks::model::Component::VERTEX_COMPONENT_NORMAL,
        vks::model::Component::VERTEX_COMPONENT_UV,
        vks::model::Component::VERTEX_COMPONENT_COLOR,
    } };

    struct {
        vks::model::Model object;
    } meshes;

    struct {
        vks::texture::Texture2DArray matCapArray;
    } textures;

    struct {
        vks::Buffer vertexShader;
    } uniformData;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 normal;
        glm::mat4 view;
        uint32_t texIndex{ 0 };
    } uboVS;

    struct {
        vk::Pipeline sem;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        rotationSpeed = 0.75f;
        zoomSpeed = 0.25f;
        camera.dolly(-0.9f);
        camera.setRotation({ -25.0f, 23.75f, 0.0f });
        title = "Vulkan Example - Spherical Environment Mapping";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.sem);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        meshes.object.destroy();

        uniformData.vertexShader.destroy();

        textures.matCapArray.destroy();
    }

    void loadAssets() override {
        // Several mat caps are stored in a single texture array
        // so they can easily be switched inside the shader
        // just by updating the index in a uniform buffer
        textures.matCapArray.loadFromFile(getAssetPath() + "textures/matcap_array_rgba.ktx", vk::Format::eR8G8B8A8Unorm);
        meshes.object.loadFromFile(getAssetPath() + "models/chinesedragon.dae", vertexLayout, 0.05f);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.sem);
        cmdBuffer.bindVertexBuffers(0, meshes.object.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
                                                          vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader color map image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // Color map image descriptor
        vk::DescriptorImageInfo texDescriptorColorMap{ defaultSampler, textures.matCapArray.imageView, vk::ImageLayout::eReadOnlyOptimal };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vertexShader.descriptor },
            // Binding 1 : Fragment shader image sampler
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorColorMap },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Spherical environment rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::sphericalenvmapping::sem::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::sphericalenvmapping::sem::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.sem = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vertexShader = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.view = camera.matrices.view;
        uboVS.normal = glm::inverseTranspose(uboVS.view * uboVS.model);
        uniformData.vertexShader.copy(uboVS);
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

    void render() override {
        if (!prepared)
            return;
        draw();
    }

    void viewChanged() override { updateUniformBuffers(); }

    void keyPressed(uint32_t keyCode) override {
        switch (keyCode) {
            case KEY_KPADD:
            case KEY_SPACE:
            case GAMEPAD_BUTTON_A:
                changeMatCapIndex(1);
                break;
            case KEY_KPSUB:
            case GAMEPAD_BUTTON_X:
                changeMatCapIndex(-1);
                break;
        }
    }

    void changeMatCapIndex(uint32_t delta) {
        uboVS.texIndex += delta;
        uboVS.texIndex %= textures.matCapArray.image.createInfo.arrayLayers;
        updateUniformBuffers();
    }
};

RUN_EXAMPLE(VulkanExample)
