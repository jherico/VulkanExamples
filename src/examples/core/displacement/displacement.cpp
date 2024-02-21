/*
 * Vulkan Example - Displacement mapping with tessellation shaders
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/displacement/base.frag.inl>
#include <shaders/displacement/base.vert.inl>
#include <shaders/displacement/displacement.tesc.inl>
#include <shaders/displacement/displacement.tese.inl>
#include <shaders/displacement/passthrough.tesc.inl>
#include <shaders/displacement/passthrough.tese.inl>

// Vertex layout for this example
const vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
} };

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

private:
    struct {
        vks::texture::Texture2D colorHeightMap;
    } textures;

public:
    bool splitScreen = true;
    bool displacement = true;

    struct {
        vks::model::Model object;
    } meshes;

    vks::Buffer uniformDataTC, uniformDataTE;

    struct UBOTessControl {
        float tessLevel = 64.0f;
    } uboTessControl;

    struct UBOTessEval {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0, -25.0, 0.0, 0.0);
        float tessAlpha = 1.0f;
        float tessStrength = 0.75f;
    } uboTessEval;

    struct {
        vk::Pipeline solid;
        vk::Pipeline wire;
        vk::Pipeline solidPassThrough;
        vk::Pipeline wirePassThrough;
    } pipelines;
    vk::Pipeline* pipelineLeft = &pipelines.solidPassThrough;
    vk::Pipeline* pipelineRight = &pipelines.solid;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.dolly(-50.25f);
        camera.setRotation(glm::vec3(-20.0f, 45.0f, 0.0f));
        title = "Tessellation shader displacement";
    }

    void initVulkan() override {
        Parent::initVulkan();
        // Support for tessellation shaders is optional, so check first
        if (!context.deviceInfo.features.core10.tessellationShader) {
            throw std::runtime_error("Selected GPU does not support tessellation shaders!");
        }
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.solid);
        device.destroy(pipelines.wire);
        device.destroy(pipelines.solidPassThrough);
        device.destroy(pipelines.wirePassThrough);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        meshes.object.destroy();

        uniformDataTC.destroy();
        uniformDataTE.destroy();

        textures.colorHeightMap.destroy();
    }

    void getEnabledFeatures() override {
        Parent::getEnabledFeatures();
        context.enabledFeatures.core10.tessellationShader = VK_TRUE;
        context.enabledFeatures.core10.fillModeNonSolid = VK_TRUE;
    }

    void loadAssets() override {
        meshes.object.loadFromFile(getAssetPath() + "models/torus.obj", vertexLayout, 0.25f);
        if (deviceInfo.features.core10.textureCompressionBC) {
            textures.colorHeightMap.loadFromFile(getAssetPath() + "textures/stonefloor03_color_bc3_unorm.ktx", vk::Format::eBc3UnormBlock);
        } else if (deviceInfo.features.core10.textureCompressionASTC_LDR) {
            textures.colorHeightMap.loadFromFile(getAssetPath() + "textures/stonefloor03_color_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock);
        } else if (deviceInfo.features.core10.textureCompressionETC2) {
            textures.colorHeightMap.loadFromFile(getAssetPath() + "textures/stonefloor03_color_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8UnormBlock);
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::Viewport viewport = vks::util::viewport(splitScreen ? (float)size.width / 2.0f : (float)size.width, (float)size.height, 0.0f, 1.0f);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.setLineWidth(1.0f);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.object.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);

        if (splitScreen) {
            cmdBuffer.setViewport(0, viewport);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelineLeft);
            cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
            viewport.x += viewport.width;
        }

        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelineRight);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
    }

    void setupDescriptorPool() {
        // Example uses two ubos and two image samplers
        std::vector<vk::DescriptorPoolSize> poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2),
                                                          vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Tessellation control shader ubo
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eTessellationControl },
            // Binding 1 : Tessellation evaluation shader ubo
            { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eTessellationEvaluation },
            // Binding 2 : Tessellation evaluation shader displacement map image sampler
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        vk::DescriptorImageInfo texDescriptor = textures.colorHeightMap.makeDescriptor(defaultSampler);
        device.updateDescriptorSets(
            {
                // Binding 0 : Tessellation control shader ubo
                { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataTC.descriptor },
                // Binding 1 : Tessellation evaluation shader ubo
                { descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataTE.descriptor },
                // Binding 2 : Color and displacement map (alpha channel)
                { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
            },
            nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.dynamicRendering(swapChain.surfaceFormat.format, deviceInfo.supportedDepthFormat);
        pipelineBuilder.inputAssemblyState.topology = vk::PrimitiveTopology::ePatchList;
        pipelineBuilder.depthStencilState = { true };
        pipelineBuilder.dynamicState.dynamicStateEnables = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eLineWidth };

        vk::PipelineTessellationStateCreateInfo tessellationState{ {}, 3 };
        pipelineBuilder.pipelineCreateInfo.pTessellationState = &tessellationState;

        // Tessellation pipeline
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::displacement::base::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::displacement::base::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.loadShader(vkx::shaders::displacement::displacement::tesc, vk::ShaderStageFlagBits::eTessellationControl);
        pipelineBuilder.loadShader(vkx::shaders::displacement::displacement::tese, vk::ShaderStageFlagBits::eTessellationEvaluation);
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        // Solid pipeline
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);

        // Wireframe pipeline
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wire = pipelineBuilder.create(context.pipelineCache);

        // Pass through pipelines
        // Load pass through tessellation shaders (Vert and frag are reused)
        context.device.destroy(pipelineBuilder.shaderStages[2].module);
        context.device.destroy(pipelineBuilder.shaderStages[3].module);
        pipelineBuilder.shaderStages.resize(2);
        pipelineBuilder.loadShader(vkx::shaders::displacement::passthrough::tesc, vk::ShaderStageFlagBits::eTessellationControl);
        pipelineBuilder.loadShader(vkx::shaders::displacement::passthrough::tese, vk::ShaderStageFlagBits::eTessellationEvaluation);

        // Solid
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        pipelines.solidPassThrough = pipelineBuilder.create(context.pipelineCache);

        // Wireframe
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wirePassThrough = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Tessellation evaluation shader uniform buffer
        uniformDataTE = loader.createUniformBuffer(uboTessEval);
        // Tessellation control shader uniform buffer
        uniformDataTC = loader.createUniformBuffer(uboTessControl);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Tessellation eval
        uboTessEval.projection = glm::perspective(glm::radians(45.0f), (float)(size.width * ((splitScreen) ? 0.5f : 1.0f)) / (float)size.height, 0.1f, 256.0f);
        uboTessEval.model = camera.matrices.view;
        // uboTessEval.lightPos.y = -0.5f - uboTessEval.tessStrength;
        uniformDataTE.copy(uboTessEval);

        // Tessellation control
        float savedLevel = uboTessControl.tessLevel;
        if (!displacement) {
            uboTessControl.tessLevel = 1.0f;
        }

        uniformDataTC.copy(uboTessControl);

        if (!displacement) {
            uboTessControl.tessLevel = savedLevel;
        }
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

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Tessellation displacement", &displacement)) {
                updateUniformBuffers();
            }
            if (ui.inputFloat("Strength", &uboTessEval.tessStrength, 0.025f)) {
                updateUniformBuffers();
            }
            if (ui.inputFloat("Level", &uboTessControl.tessLevel, 0.5f, "%.2f")) {
                updateUniformBuffers();
            }
            if (deviceInfo.features.core10.fillModeNonSolid) {
                if (ui.checkBox("Splitscreen", &splitScreen)) {
                    buildCommandBuffers();
                    updateUniformBuffers();
                }
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
