/*
 * Vulkan Example - Multiview sample with single pass stereo rendering using VK_KHR_multiview
 *
 * Copyright (C) 2018 by Bradley Austin Davis
 * Based on Viewport.cpp by Sascha Willems
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <examples/offscreen.hpp>
#include <shaders/multiview/scene.frag.inl>
#include <shaders/multiview/scene.vert.inl>
// Vertex layout for the models
static const vks::model::VertexLayout VERTEX_LAYOUT{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::OffscreenExampleBase {
    using Parent = OffscreenExampleBase;

public:
    vks::model::Model scene;

    struct UBOGS {
        glm::mat4 projection[2];
        glm::mat4 modelview[2];
        glm::vec4 lightPos = glm::vec4(-2.5f, -3.5f, 0.0f, 1.0f);
    } uboVS;

    vks::Buffer uniformBufferVS;

    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Camera and view properties
    float eyeSeparation = 0.08f;
    const float focalLength = 0.5f;
    const float fov = 90.0f;
    const float zNear = 0.1f;
    const float zFar = 256.0f;

    VulkanExample() {
        title = "Multiview";
        camera.type = Camera::CameraType::firstperson;
        camera.setRotation(glm::vec3(0.0f, 90.0f, 0.0f));
        camera.setTranslation(glm::vec3(7.0f, 3.2f, 0.0f));
        camera.movementSpeed = 5.0f;
        settings.overlay = true;
        context.requireDeviceExtensions({ VK_KHR_MULTIVIEW_EXTENSION_NAME });
    }

    ~VulkanExample() {
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        scene.destroy();
        uniformBufferVS.destroy();
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        if (!deviceInfo.features.core11.multiview) {
            throw std::runtime_error("Multiview unsupported");
        }

        context.enabledFeatures.core11.multiview = deviceInfo.features.core11.multiview;
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildOffscreenCommandBuffer() override {
        using namespace vks::util;
        vk::DeviceSize offsets = { 0 };
        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        auto& color = offscreen.colorTargets[0].image;
        setImageLayout(offscreen.cmdBuffer, color, ImageTransitionState::UNDEFINED, ImageTransitionState::COLOR_ATTACHMENT);
        offscreen.cmdBuffer.beginRendering(offscreen.renderingInfo);
        offscreen.cmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        offscreen.cmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        offscreen.cmdBuffer.bindVertexBuffers(0, scene.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(scene.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        offscreen.cmdBuffer.drawIndexed(scene.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRendering();
        setImageLayout(offscreen.cmdBuffer, color, ImageTransitionState::COLOR_ATTACHMENT, ImageTransitionState::TRANSFER_SRC);
        offscreen.cmdBuffer.end();
    }

    void buildCommandBuffers() override {
        std::array<vk::ImageBlit2, 2> compositeBlits;
        for (auto& blit : compositeBlits) {
            blit.dstSubresource.aspectMask = blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.layerCount = blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[1] = vk::Offset3D{ (int32_t)offscreen.size.width, (int32_t)offscreen.size.height, 1 };
            blit.dstOffsets[1] = vk::Offset3D{ (int32_t)offscreen.size.width, (int32_t)offscreen.size.height, 1 };
        }
        compositeBlits[1].srcSubresource.baseArrayLayer = 1;
        compositeBlits[1].dstOffsets[0].x += offscreen.size.width;
        compositeBlits[1].dstOffsets[1].x += offscreen.size.width;

        using namespace vks::util;
        for (size_t i = 0; i < swapChain.imageCount; ++i) {
            const auto& cmdBuffer = perImageData[i].commandBuffer;
            auto& swapchainImage = swapChain.images[i];
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(vk::CommandBufferBeginInfo{});
            setImageLayout(cmdBuffer, swapchainImage.image, swapchainImage.getWholeRange(), ImageTransitionState::UNDEFINED,
                           ImageTransitionState::TRANSFER_DST);
            cmdBuffer.blitImage2(vk::BlitImageInfo2{
                offscreen.colorTargets[0].image.image,
                vk::ImageLayout::eTransferSrcOptimal,
                swapChain.images[i].image,
                vk::ImageLayout::eTransferDstOptimal,
                compositeBlits,
                vk::Filter::eNearest,
            });
            setImageLayout(cmdBuffer, swapchainImage.image, swapchainImage.getWholeRange(), ImageTransitionState::TRANSFER_DST, ImageTransitionState::PRESENT);
            cmdBuffer.end();
        }
    }

    void loadAssets() override { scene.loadFromFile(getAssetPath() + "models/sampleroom.dae", VERTEX_LAYOUT, 0.25f); }

    void setupDescriptorPool() {
        // Example uses two ubos
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferVS.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.renderingCreateInfo.viewMask = 0x3;
        VERTEX_LAYOUT.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelineBuilder.loadShader(vkx::shaders::multiview::scene::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::multiview::scene::frag, vk::ShaderStageFlagBits::eFragment);
        pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Geometry shader uniform buffer block
        uniformBufferVS = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Matrices for the two viewports

        // Calculate some variables
        float aspectRatio = (float)(size.width * 0.5f) / (float)size.height;
        float wd2 = zNear * tan(glm::radians(fov / 2.0f));
        float ndfl = zNear / focalLength;
        float left, right;
        float top = wd2;
        float bottom = -wd2;

        glm::vec3 camFront = camera.getFront();
        glm::vec3 camRight = glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::mat4 rotM = glm::mat4(1.0f);
        glm::mat4 transM;

        rotM = glm::rotate(rotM, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

        // Left eye
        left = -aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;
        right = aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;

        transM = glm::translate(glm::mat4(1.0f), camera.position - camRight * (eyeSeparation / 2.0f));

        uboVS.projection[0] = glm::frustum(left, right, bottom, top, zNear, zFar);
        uboVS.modelview[0] = rotM * transM;

        // Right eye
        left = -aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;
        right = aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;

        transM = glm::translate(glm::mat4(1.0f), camera.position + camRight * (eyeSeparation / 2.0f));

        uboVS.projection[1] = glm::frustum(left, right, bottom, top, zNear, zFar);
        uboVS.modelview[1] = rotM * transM;
        uniformBufferVS.copy(uboVS);
    }

    void prepareOffscreen() override {
        vkx::offscreen::Builder builder({ size.width / 2, size.height });
        builder.withLayerCount(2)
            .withViewType(vk::ImageViewType::e2DArray)
            .appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eTransferSrc, vk::ClearColorValue{ 0.025f, 0.025f, 0.025f, 1.0f })
            .withDepthFormat(defaultDepthStencilFormat);

        offscreen.prepare(builder);
        // Because we're using multiview, we need to enable the view mask to target both array layers
        offscreen.renderingInfo.viewMask = 3;
    }

    void prepare() override {
        Parent::prepare();
        loadAssets();
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

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.sliderFloat("Eye separation", &eyeSeparation, -1.0f, 1.0f)) {
                updateUniformBuffers();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
