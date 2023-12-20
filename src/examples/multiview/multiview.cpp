/*
* Vulkan Example - Multiview sample with single pass stereo rendering using VK_KHR_multiview 
*
* Copyright (C) 2018 by Bradley Austin Davis
* Based on Viewport.cpp by Sascha Willems
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

// Vertex layout for the models
static const vks::model::VertexLayout VERTEX_LAYOUT{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::ExampleBase {
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

    struct Offscreen {
        vk::Extent2D size;
        vk::Device device;
        vk::Framebuffer frameBuffer;
        vk::RenderPass renderPass;
        vks::Image color, depth;
        vk::Semaphore semaphore;
        // FIXME not cleaned up on exit
        vk::CommandBuffer cmdBuffer;
        void destroy() {
            color.destroy();
            depth.destroy();
            device.destroy(frameBuffer);
            device.destroy(semaphore);
            device.destroy(renderPass);
        }
    } offscreen;

    struct Multiview {
        vk::PhysicalDeviceMultiviewFeatures features;
        vk::PhysicalDeviceMultiviewProperties properties;
    } multiview;

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
        offscreen.destroy();
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        scene.destroy();
        uniformBufferVS.destroy();
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        // FIXME support the KHR version of the extension?
        if (context.deviceProperties.apiVersion < VK_MAKE_VERSION(1, 1, 0)) {
            throw std::runtime_error("This example requires Vulkan 1.1");
        }

        multiview.features = physicalDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceMultiviewFeatures>(context.dynamicDispatch)
                                 .get<vk::PhysicalDeviceMultiviewFeatures>();
        if (!multiview.features.multiview) {
            throw std::runtime_error("Multiview unsupported");
        }

        multiview.properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMultiviewProperties>(context.dynamicDispatch)
                                   .get<vk::PhysicalDeviceMultiviewProperties>();
        context.enabledFeatures2.pNext = &multiview.features;
    }

    // Create a frame buffer attachment for rendering using multiview
    void createMultiviewAttachment(vk::Format format, const vk::ImageUsageFlags& usage, vks::Image& attachment, const vk::Extent2D& size) {
        vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;
        vk::ImageLayout imageLayout = vk::ImageLayout::eColorAttachmentOptimal;

        attachment.format = format;

        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        }

        vk::ImageCreateInfo image;
        image.imageType = vk::ImageType::e2D;
        image.format = format;
        image.extent.width = size.width;
        image.extent.height = size.height;
        image.extent.depth = 1;
        image.mipLevels = 1;
        // In order to support multiview, you must use array images in the framebuffer.  Each array layer will be
        // one of the target views
        image.arrayLayers = 2;
        image.usage = usage | vk::ImageUsageFlagBits::eSampled;

        attachment = context.createImage(image);

        // In order for the framebuffer to behave correctly, we must create the image view as an array view
        // Note the use of vk::ImageViewType::e2DArray and the layerCount of 2 in the subresourceRange
        vk::ImageViewCreateInfo imageView;
        imageView.viewType = vk::ImageViewType::e2DArray;
        imageView.format = format;
        imageView.subresourceRange = { aspectMask, 0, 1, 0, 2 };
        imageView.image = attachment.image;
        attachment.view = device.createImageView(imageView);
    }

    void prepareOffscreenFramebuffer() {
        offscreen.device = device;
        offscreen.size = size;
        // Each viewport is going to be half the width of the total window size
        offscreen.size.width /= 2;

        // G-Buffer
        createMultiviewAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, offscreen.color,
                                  offscreen.size);  // color
        createMultiviewAttachment(context.getSupportedDepthFormat(), vk::ImageUsageFlagBits::eDepthStencilAttachment, offscreen.depth,
                                  offscreen.size);  // Depth

        std::array<vk::SubpassDependency, 2> dependencies{
            vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                                   vk::DependencyFlagBits::eByRegion },
            vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead,
                                   vk::DependencyFlagBits::eByRegion },
        };

        // Render passes

        {
            std::array<vk::AttachmentDescription, 2> attachmentDescs;

            // Init attachment properties
            for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++) {
                attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
                attachmentDescs[i].storeOp = vk::AttachmentStoreOp::eStore;
                attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
                attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
                attachmentDescs[i].finalLayout = (i == 1) ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eTransferSrcOptimal;
            }

            // Formats
            attachmentDescs[0].format = offscreen.color.format;
            attachmentDescs[1].format = offscreen.depth.format;

            vk::AttachmentReference colorReference{ 0, vk::ImageLayout::eColorAttachmentOptimal };
            vk::AttachmentReference depthReference{ 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };
            vk::SubpassDescription subpass{ {}, vk::PipelineBindPoint::eGraphics };

            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorReference;
            subpass.pDepthStencilAttachment = &depthReference;

            uint32_t viewMask = 3;
            vk::RenderPassCreateInfo renderPassInfo;
            vk::RenderPassMultiviewCreateInfo renderPassMultiviewInfo;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
            renderPassInfo.pAttachments = attachmentDescs.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();
            renderPassInfo.pNext = &renderPassMultiviewInfo;
            renderPassMultiviewInfo.subpassCount = renderPassInfo.subpassCount;
            renderPassMultiviewInfo.pViewMasks = &viewMask;

            offscreen.renderPass = device.createRenderPass(renderPassInfo);
            std::array<vk::ImageView, 2> attachments{
                offscreen.color.view,
                offscreen.depth.view,
            };

            vk::FramebufferCreateInfo framebufferCreateInfo;
            framebufferCreateInfo.renderPass = offscreen.renderPass;
            framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferCreateInfo.pAttachments = attachments.data();
            framebufferCreateInfo.width = offscreen.size.width;
            framebufferCreateInfo.height = offscreen.size.height;
            framebufferCreateInfo.layers = 2;
            offscreen.frameBuffer = device.createFramebuffer(framebufferCreateInfo);
        }
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildOffscreenCommandBuffer() {
        vk::DeviceSize offsets = { 0 };

        if (!offscreen.cmdBuffer) {
            offscreen.cmdBuffer = context.allocateCommandBuffers(1, vk::CommandBufferLevel::ePrimary)[0];
        }

        // Create a semaphore used to synchronize offscreen rendering and usage
        if (!offscreen.semaphore) {
            offscreen.semaphore = device.createSemaphore({});
        }

        offscreen.cmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        std::vector<vk::ClearValue> clearValues(2);
        clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 1.0f });
        clearValues[1].depthStencil = defaultClearDepth;
        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = offscreen.renderPass;
        renderPassBeginInfo.framebuffer = offscreen.frameBuffer;
        renderPassBeginInfo.renderArea.extent = offscreen.size;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        vk::Viewport viewport{ 0, 0, (float)offscreen.size.width, (float)offscreen.size.height, 0, 1 };
        vk::Rect2D scissor{ vk::Offset2D{}, offscreen.size };
        offscreen.cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offscreen.cmdBuffer.setViewport(0, viewport);
        offscreen.cmdBuffer.setScissor(0, scissor);
        offscreen.cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        offscreen.cmdBuffer.bindVertexBuffers(0, scene.vertices.buffer, { 0 });
        offscreen.cmdBuffer.bindIndexBuffer(scene.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        offscreen.cmdBuffer.drawIndexed(scene.indexCount, 1, 0, 0, 0);
        offscreen.cmdBuffer.endRenderPass();
        offscreen.cmdBuffer.end();
    }

    void buildCommandBuffers() override {
        // Destroy and recreate command buffers if already present
        allocateCommandBuffers();

        std::array<vk::ImageBlit, 2> compositeBlits;
        for (auto& blit : compositeBlits) {
            blit.dstSubresource.aspectMask = blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.layerCount = blit.srcSubresource.layerCount = 1;
            blit.srcOffsets[1] = vk::Offset3D{ (int32_t)offscreen.size.width, (int32_t)offscreen.size.height, 1 };
            blit.dstOffsets[1] = vk::Offset3D{ (int32_t)offscreen.size.width, (int32_t)offscreen.size.height, 1 };
        }
        compositeBlits[1].srcSubresource.baseArrayLayer = 1;
        compositeBlits[1].dstOffsets[0].x += offscreen.size.width;
        compositeBlits[1].dstOffsets[1].x += offscreen.size.width;

        for (size_t i = 0; i < swapChain.imageCount; ++i) {
            const auto& cmdBuffer = commandBuffers[i];
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(vk::CommandBufferBeginInfo{});
            context.setImageLayout(cmdBuffer, swapChain.images[i].image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            cmdBuffer.blitImage(offscreen.color.image, vk::ImageLayout::eTransferSrcOptimal, swapChain.images[i].image, vk::ImageLayout::eTransferDstOptimal,
                                compositeBlits, vk::Filter::eNearest);
            context.setImageLayout(cmdBuffer, swapChain.images[i].image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
            cmdBuffer.end();
        }
    }

    void loadAssets() override { scene.loadFromFile(context, getAssetPath() + "models/sampleroom.dae", VERTEX_LAYOUT, 0.25f); }

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
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, offscreen.renderPass };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.vertexInputState.appendVertexLayout(VERTEX_LAYOUT);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/multiview/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/multiview/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Geometry shader uniform buffer block
        uniformBufferVS = context.createUniformBuffer(uboVS);
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

    void draw() override {
        prepareFrame();
        context.submit(offscreen.cmdBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreen.semaphore);
        renderWaitSemaphores = { offscreen.semaphore };
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        loadAssets();
        prepareOffscreenFramebuffer();
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
