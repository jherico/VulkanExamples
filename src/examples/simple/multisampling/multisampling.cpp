/*
 * Vulkan Example - Multisampling using resolve attachments
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <examples/offscreen.hpp>
#include <shaders/mesh/mesh.frag.inl>
#include <shaders/mesh/mesh.vert.inl>

#define SAMPLE_COUNT vk::SampleCountFlagBits::e4

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::OffscreenExampleBase {
    using Parent = vkx::OffscreenExampleBase;

public:
    struct {
        vks::texture::Texture2D colorMap;
    } textures;

    struct {
        vks::model::Model example;
    } meshes;

    struct {
        vks::Buffer vsScene;
    } uniformData;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, 5.0f, 1.0f);
    } uboVS;

    struct {
        vk::Pipeline solid;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::RenderPass uiRenderPass;

    VulkanExample() {
        zoomSpeed = 2.5f;
        camera.setRotation({ 0.0f, -90.0f, 0.0f });
        camera.setTranslation({ 2.5f, 2.5f, -7.5 });
        title = "Vulkan Example - Multisampling";
        settings.overlay = false;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.solid);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        meshes.example.destroy();

        textures.colorMap.destroy();
        uniformData.vsScene.destroy();
    }

    // Creates a multi sample render target (image and view) that is used to resolve
    // into the visible frame buffer target in the render pass
    void prepareOffscreen() {
        // Check if device supports requested sample count for color and depth frame buffer
        auto& limits = context.deviceInfo.properties.core10.limits;
        vk::SampleCountFlags colorSampleCount = limits.framebufferColorSampleCounts;
        vk::SampleCountFlags depthSampleCount = limits.framebufferDepthSampleCounts;
        vk::SampleCountFlags requiredSamples = SAMPLE_COUNT;
        assert((uint32_t)colorSampleCount >= (uint32_t)requiredSamples && (uint32_t)depthSampleCount >= (uint32_t)requiredSamples);

        vkx::offscreen::Builder builder{ size };
        builder.appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eTransferSrc);
        builder.withDepthFormat(defaultDepthStencilFormat);
        builder.withSampleCount(SAMPLE_COUNT);
        offscreen.prepare(builder);
    }

    void buildOffscreenCommandBuffer() {
        auto& cmdBuffer = offscreen.cmdBuffer;
        cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        offscreen.setLayout(cmdBuffer, vks::util::ImageTransitionState::COLOR_ATTACHMENT, vks::util::ImageTransitionState::DEPTH_ATTACHMENT);

        cmdBuffer.beginRendering(offscreen.renderingInfo);
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
        cmdBuffer.endRendering();
        offscreen.setLayout(cmdBuffer, vks::util::ImageTransitionState::TRANSFER_SRC);
        cmdBuffer.end();
    }

    void buildCommandBuffers() override {
        perImageData.resize(swapChain.imageCount);
        auto commandBuffers = graphicsQueue.allocateCommandBuffers(swapChain.imageCount);
        // Destroy and recreate command buffers if already present
        vk::CommandBufferBeginInfo cmdBufInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse };
        for (currentIndex = 0; currentIndex < swapChain.imageCount; ++currentIndex) {
            perImageData[currentIndex].commandBuffer = commandBuffers[currentIndex];
            const auto& swapchainImage = swapChain.images[currentIndex].image;
            const auto& cmdBuffer = perImageData[currentIndex].commandBuffer;
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(cmdBufInfo);
            using namespace vks::util;
            vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
            setImageLayout(cmdBuffer, swapchainImage, range, ImageTransitionState::UNDEFINED, ImageTransitionState::TRANSFER_DST);
            vk::ImageResolve resolve;
            resolve.srcSubresource = vk::ImageSubresourceLayers{ vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
            resolve.dstSubresource = vk::ImageSubresourceLayers{ vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
            resolve.extent = vk::Extent3D{ size.width, size.height, 1 };
            cmdBuffer.resolveImage(                                                           //
                offscreen.colorTargets[0].image.image, vk::ImageLayout::eTransferSrcOptimal,  //
                swapChain.images[currentIndex].image, vk::ImageLayout::eTransferDstOptimal,   //
                resolve);
            setImageLayout(cmdBuffer, swapchainImage, range, ImageTransitionState::TRANSFER_DST, ImageTransitionState::PRESENT);
            cmdBuffer.end();
        }
    }

    void loadAssets() override {
        textures.colorMap.loadFromFile(getAssetPath() + "models/voyager/voyager.ktx", vk::Format::eBc3UnormBlock);
        meshes.example.loadFromFile(getAssetPath() + "models/voyager/voyager.dae", vertexLayout);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one combined image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader combined sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        vk::DescriptorImageInfo texDescriptor{ defaultSampler, textures.colorMap.imageView, vk::ImageLayout::eReadOnlyOptimal };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsScene.descriptor },
            // Binding 1 : Color map
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Solid rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.multisampleState.rasterizationSamples = SAMPLE_COUNT;
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::mesh::mesh::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::mesh::mesh::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uniformData.vsScene.copy(uboVS);
    }

    void prepare() override {
        Parent::prepare();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }
};

RUN_EXAMPLE(VulkanExample)
