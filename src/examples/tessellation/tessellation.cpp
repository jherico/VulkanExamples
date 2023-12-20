/*
* Vulkan Example - Tessellation shader PN triangles
*
* Based on http://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
* Shaders based on http://onrendering.blogspot.de/2011/12/tessellation-on-gpu-curved-pn-triangles.html
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ { vks::model::VERTEX_COMPONENT_POSITION, vks::model::VERTEX_COMPONENT_NORMAL, vks::model::VERTEX_COMPONENT_UV } };

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    bool splitScreen = true;
    bool wireframe = true;

    struct {
        vks::texture::Texture2D colorMap;
    } textures;

    struct {
        vks::model::Model object;
    } meshes;

    vks::Buffer uniformDataTC, uniformDataTE;

    struct UboTC {
        float tessLevel = 3.0f;
    } uboTC;

    struct UboTE {
        glm::mat4 projection;
        glm::mat4 model;
        float tessAlpha = 1.0f;
    } uboTE;

    struct {
        vk::Pipeline solid;
        vk::Pipeline wire;
        vk::Pipeline solidPassThrough;
        vk::Pipeline wirePassThrough;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.setRotation({ -350.0f, 60.0f, 0.0f });
        camera.setTranslation({ -3.0f, 2.3f, -6.5f });
        title = "Vulkan Example - Tessellation shader (PN Triangles)";
    }

    void getEnabledFeatures() override {
        // Example uses tessellation shaders
        if (deviceFeatures.tessellationShader) {
            enabledFeatures.tessellationShader = VK_TRUE;
        } else {
            throw std::runtime_error("Selected GPU does not support tessellation shaders!");
        }
        // Fill mode non solid is required for wireframe display
        if (deviceFeatures.fillModeNonSolid) {
            enabledFeatures.fillModeNonSolid = VK_TRUE;
        } else {
            wireframe = false;
        }
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.solid);
        device.destroyPipeline(pipelines.wire);
        device.destroyPipeline(pipelines.solidPassThrough);
        device.destroyPipeline(pipelines.wirePassThrough);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        meshes.object.destroy();

        device.destroyBuffer(uniformDataTC.buffer);
        device.freeMemory(uniformDataTC.memory);

        device.destroyBuffer(uniformDataTE.buffer);
        device.freeMemory(uniformDataTE.memory);

        textures.colorMap.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::Viewport viewport = vks::util::viewport(splitScreen ? (float)size.width / 2.0f : (float)size.width, (float)size.height, 0.0f, 1.0f);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.setLineWidth(1.0f);

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, meshes.object.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);

        if (splitScreen) {
            cmdBuffer.setViewport(0, viewport);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wirePassThrough : pipelines.solidPassThrough);
            cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
            viewport.x = float(size.width) / 2;
        }

        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wire : pipelines.solid);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        meshes.object.loadFromFile(context, getAssetPath() + "models/lowpoly/deer.dae", vertexLayout, 1.0f);
        textures.colorMap.loadFromFile(context, getAssetPath() + "textures/deer.ktx", vk::Format::eBc3UnormBlock);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eTessellationControl },
            { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eTessellationEvaluation },
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        vk::DescriptorImageInfo texDescriptor = vk::DescriptorImageInfo(textures.colorMap.sampler, textures.colorMap.view, vk::ImageLayout::eGeneral);
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Tessellation control shader ubo
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataTC.descriptor },
            // Binding 1 : Tessellation evaluation shader ubo
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataTE.descriptor },
            // Binding 2 : Color map
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.inputAssemblyState.topology = vk::PrimitiveTopology::ePatchList;
        builder.dynamicState.dynamicStateEnables.push_back(vk::DynamicState::eLineWidth);
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        vk::PipelineTessellationStateCreateInfo tessellationState{ {}, 3 };
        builder.pipelineCreateInfo.pTessellationState = &tessellationState;

        // Tessellation pipelines
        // Load shaders
        builder.loadShader(getAssetPath() + "shaders/tessellation/base.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/tessellation/base.frag.spv", vk::ShaderStageFlagBits::eFragment);
        builder.loadShader(getAssetPath() + "shaders/tessellation/pntriangles.tesc.spv", vk::ShaderStageFlagBits::eTessellationControl);
        builder.loadShader(getAssetPath() + "shaders/tessellation/pntriangles.tese.spv", vk::ShaderStageFlagBits::eTessellationEvaluation);

        // Tessellation pipelines
        // Solid
        pipelines.solid = builder.create(context.pipelineCache);
        // Wireframe
        builder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wire = builder.create(context.pipelineCache);

        // Pass through pipelines
        // Load pass through tessellation shaders (Vert and frag are reused)
        device.destroyShaderModule(builder.shaderStages[2].module);
        device.destroyShaderModule(builder.shaderStages[3].module);
        builder.shaderStages.resize(2);
        builder.loadShader(getAssetPath() + "shaders/tessellation/passthrough.tesc.spv", vk::ShaderStageFlagBits::eTessellationControl);
        builder.loadShader(getAssetPath() + "shaders/tessellation/passthrough.tese.spv", vk::ShaderStageFlagBits::eTessellationEvaluation);

        // Solid
        builder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        pipelines.solidPassThrough = builder.create(context.pipelineCache);

        // Wireframe
        builder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wirePassThrough = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Tessellation evaluation shader uniform buffer
        uniformDataTE = context.createUniformBuffer(uboTE);
        // Tessellation control shader uniform buffer
        uniformDataTC = context.createUniformBuffer(uboTC);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Tessellation eval
        uboTE.projection = glm::perspective(glm::radians(45.0f), (float)(size.width * ((splitScreen) ? 0.5f : 1.0f)) / (float)size.height, 0.1f, 256.0f);
        uboTE.model = camera.matrices.view;
        uniformDataTE.copy(uboTE);

        // Tessellation control uniform block
        uniformDataTC.copy(uboTC);
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

    virtual void OnUpdateUIOverlay() {
        if (ui.header("Settings")) {
            if (ui.inputFloat("Tessellation level", &uboTC.tessLevel, 0.25f, "%.2f")) {
                updateUniformBuffers();
            }
            if (deviceFeatures.fillModeNonSolid) {
                if (ui.checkBox("Wireframe", &wireframe)) {
                    updateUniformBuffers();
                    buildCommandBuffers();
                }
                if (ui.checkBox("Splitscreen", &splitScreen)) {
                    updateUniformBuffers();
                    buildCommandBuffers();
                }
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
