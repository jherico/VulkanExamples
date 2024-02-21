/*
 * Vulkan Example - Offscreen rendering using a separate framebuffer
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/offscreen.hpp>

#include <shaders/offscreen/mirror.frag.inl>
#include <shaders/offscreen/mirror.vert.inl>
#include <shaders/offscreen/offscreen.frag.inl>
#include <shaders/offscreen/offscreen.vert.inl>
#include <shaders/offscreen/quad.frag.inl>
#include <shaders/offscreen/quad.vert.inl>

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
} };

class VulkanExample : public vkx::OffscreenExampleBase {
public:
    struct {
        vks::texture::Texture2D colorMap;
    } textures;

    struct {
        vks::model::Model example;
        vks::model::Model plane;
    } meshes;

    struct {
        vks::Buffer vsShared;
        vks::Buffer vsMirror;
        vks::Buffer vsOffScreen;
    } uniformData;

    struct UBO {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    };

    struct {
        UBO vsShared;
    } ubos;

    struct {
        vk::Pipeline shaded;
        vk::Pipeline mirror;
    } pipelines;

    struct {
        vk::PipelineLayout quad;
        vk::PipelineLayout offscreen;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet mirror;
        vk::DescriptorSet model;
        vk::DescriptorSet offscreen;
    } descriptorSets;

    vk::DescriptorSetLayout descriptorSetLayout;

    glm::vec3 meshPos = glm::vec3(0.0f, -1.5f, 0.0f);

    VulkanExample() {
        camera.setRotation({ -2.5f, 0.0f, 0.0f });
        camera.setPosition({ 0.0f, 1.0f, 0.0f });
        camera.dolly(-6.5f);
        timerSpeed *= 0.25f;
        title = "Vulkan Example - Offscreen rendering";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        // Textures
        // textureTarget.destroy();
        textures.colorMap.destroy();

        device.destroy(pipelines.shaded);
        device.destroy(pipelines.mirror);
        device.destroy(pipelineLayouts.offscreen);
        device.destroy(pipelineLayouts.quad);

        device.destroy(descriptorSetLayout);

        // Meshes
        meshes.example.destroy();
        meshes.plane.destroy();

        // Uniform buffers
        uniformData.vsShared.destroy();
        uniformData.vsMirror.destroy();
        uniformData.vsOffScreen.destroy();
    }

    void prepareOffscreen() override {
        auto builder = vkx::offscreen::Builder(512);
        builder.appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eSampled).withDepthFormat(defaultDepthStencilFormat);
        offscreen.prepare(builder);
    }

    // The command buffer to copy for rendering
    // the offscreen scene and blitting it into
    // the texture target is only build once
    // and gets resubmitted
    void buildOffscreenCommandBuffer() override {
        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        offscreen.setLayout(offscreen.cmdBuffer, vks::util::ImageTransitionState::COLOR_ATTACHMENT);
        offscreen.cmdBuffer.beginRendering(offscreen.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.offscreen, nullptr);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.shaded);
        offscreen.cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRendering();
        offscreen.setLayout(offscreen.cmdBuffer, vks::util::ImageTransitionState::SAMPLED);
        offscreen.cmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::DeviceSize offsets = 0;
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        // Reflection plane
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.quad, 0, descriptorSets.mirror, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.mirror);
        cmdBuffer.bindVertexBuffers(0, meshes.plane.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.plane.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.plane.indexCount, 1, 0, 0, 0);

        // Model
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.quad, 0, descriptorSets.model, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.shaded);
        cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override {
        meshes.plane.loadFromFile(getAssetPath() + "models/plane.obj", vertexLayout, 0.4f);
        meshes.example.loadFromFile(getAssetPath() + "models/chinesedragon.dae", vertexLayout, 0.3f);
        std::string filename;
        vk::Format format;
        const auto& features = context.deviceInfo.features.core10;
        if (features.textureCompressionBC) {
            filename = "textures/darkmetal_bc3_unorm.ktx";
            format = vk::Format::eBc3UnormBlock;
        } else if (features.textureCompressionASTC_LDR) {
            filename = "textures/darkmetal_astc_8x8_unorm.ktx";
            format = vk::Format::eAstc8x8UnormBlock;
        } else if (features.textureCompressionETC2) {
            filename = "textures/darkmetal_etc2_unorm.ktx";
            format = vk::Format::eEtc2R8G8B8UnormBlock;
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }
        textures.colorMap.loadFromFile(getAssetPath() + filename, format);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 6 },
            { vk::DescriptorType::eCombinedImageSampler, 8 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 5, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Textured quad pipeline layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 2 : Fragment shader image sampler
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayouts.quad = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        // Offscreen pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        // Mirror plane descriptor set
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        descriptorSets.mirror = device.allocateDescriptorSets(allocInfo)[0];

        // vk::Image descriptor for the offscreen mirror texture
        vk::DescriptorImageInfo texDescriptorMirror{ offscreen.sampler, offscreen.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal };
        vk::DescriptorImageInfo modelColorMapDesc = textures.colorMap.makeDescriptor(defaultSampler);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.mirror, 0, 0, vk::DescriptorType::eUniformBuffer, nullptr, uniformData.vsMirror.descriptor },
            // Binding 1 : Fragment shader texture sampler
            { descriptorSets.mirror, 1, 0, vk::DescriptorType::eCombinedImageSampler, texDescriptorMirror },
            // Binding 2 : Fragment shader texture sampler
            { descriptorSets.mirror, 2, 0, vk::DescriptorType::eCombinedImageSampler, modelColorMapDesc },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});

        // Model
        // No texture
        descriptorSets.model = device.allocateDescriptorSets(allocInfo)[0];
        std::vector<vk::WriteDescriptorSet> modelWriteDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.model, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsShared.descriptor },
        };
        device.updateDescriptorSets(modelWriteDescriptorSets, {});

        // Offscreen
        descriptorSets.offscreen = device.allocateDescriptorSets(allocInfo)[0];
        std::vector<vk::WriteDescriptorSet> offscreenWriteDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.offscreen, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsOffScreen.descriptor },
        };
        device.updateDescriptorSets(offscreenWriteDescriptorSets, {});
    }

    void preparePipelines() {
        // Solid rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.quad };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelineBuilder.loadShader(vkx::shaders::offscreen::mirror::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::offscreen::mirror::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.mirror = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Solid shading pipeline
        pipelineBuilder.loadShader(vkx::shaders::offscreen::offscreen::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::offscreen::offscreen::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.layout = pipelineLayouts.offscreen;
        pipelines.shaded = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Mesh vertex shader uniform buffer block
        uniformData.vsShared = loader.createUniformBuffer(ubos.vsShared);
        // Mirror plane vertex shader uniform buffer block
        uniformData.vsMirror = loader.createUniformBuffer(ubos.vsShared);
        // Offscreen vertex shader uniform buffer block
        uniformData.vsOffScreen = loader.createUniformBuffer(ubos.vsShared);

        updateUniformBuffers();
        updateUniformBufferOffscreen();
    }

    void updateUniformBuffers() {
        // Mesh
        ubos.vsShared.projection = getProjection();
        ubos.vsShared.model = glm::translate(camera.matrices.view, meshPos);
        uniformData.vsShared.copy(ubos.vsShared);

        // Mirror
        ubos.vsShared.model = camera.matrices.view;
        uniformData.vsMirror.copy(ubos.vsShared);
    }

    void updateUniformBufferOffscreen() {
        ubos.vsShared.projection = getProjection();
        ubos.vsShared.model = camera.matrices.view;
        ubos.vsShared.model = glm::scale(ubos.vsShared.model, glm::vec3(1.0f, -1.0f, 1.0f));
        ubos.vsShared.model = glm::translate(ubos.vsShared.model, meshPos);
        uniformData.vsOffScreen.copy(ubos.vsShared);
    }

    void prepare() override {
        OffscreenExampleBase::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildOffscreenCommandBuffer();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            updateUniformBuffers();
            updateUniformBufferOffscreen();
        }
    }

    void viewChanged() override {
        updateUniformBuffers();
        updateUniformBufferOffscreen();
    }
};

RUN_EXAMPLE(VulkanExample)
