/*
* Vulkan Example - Conservative rasterization
*
* Note: Requires a device that supports the VK_EXT_conservative_rasterization extension
*
* Uses an offscreen buffer with lower resolution to demonstrate the effect of conservative rasterization
*
* Copyright (C) 2018 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

#define FB_COLOR_FORMAT vk::Format::eR8G8B8A8Unorm
#define ZOOM_FACTOR 16

class VulkanExample : public vkx::ExampleBase {
public:
    // Fetch and store conservative rasterization state props for display purposes
    vk::PhysicalDeviceConservativeRasterizationPropertiesEXT conservativeRasterProps;

    bool conservativeRasterEnabled = true;

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct Triangle {
        vks::Buffer vertices;
        vks::Buffer indices;
        uint32_t indexCount;
    } triangle;

    vks::Buffer uniformBuffer;

    struct UniformBuffers {
        vks::Buffer scene;
    } uniformBuffers;

    struct UboScene {
        glm::mat4 projection;
        glm::mat4 model;
    } uboScene;

    struct PipelineLayouts {
        vk::PipelineLayout scene;
        vk::PipelineLayout fullscreen;
    } pipelineLayouts;

    struct Pipelines {
        vk::Pipeline triangle;
        vk::Pipeline triangleConservativeRaster;
        vk::Pipeline triangleOverlay;
        vk::Pipeline fullscreen;
    } pipelines;

    struct DescriptorSetLayouts {
        vk::DescriptorSetLayout scene;
        vk::DescriptorSetLayout fullscreen;
    } descriptorSetLayouts;

    struct DescriptorSets {
        vk::DescriptorSet scene;
        vk::DescriptorSet fullscreen;
    } descriptorSets;

    // Framebuffer for offscreen rendering
    struct OffscreenPass {
        vk::Extent2D extent;
        uint32_t& width = extent.width;
        uint32_t& height = extent.height;
        vk::Framebuffer frameBuffer;
        vks::Image color, depth;
        vk::RenderPass renderPass;
        vk::Sampler sampler;
        vk::DescriptorImageInfo descriptor;
        vk::CommandBuffer commandBuffer;
        vk::Semaphore semaphore;
    } offscreenPass;

    VulkanExample() {
        title = "Conservative rasterization";
        settings.overlay = true;

        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));

        // Enable extension required for conservative rasterization
        context.requireDeviceExtensions({ VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME });
        // Reading device properties of conservative rasterization requires VK_KHR_get_physical_device_properties2 to be enabled
        context.requireExtensions({ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
    }

    ~VulkanExample() {
        // Frame buffer
        offscreenPass.color.destroy();
        offscreenPass.depth.destroy();
        device.destroy(offscreenPass.renderPass);
        device.destroy(offscreenPass.sampler);
        device.destroy(offscreenPass.frameBuffer);
        device.destroy(pipelines.triangle);
        device.destroy(pipelines.triangleOverlay);
        device.destroy(pipelines.triangleConservativeRaster);
        device.destroy(pipelines.fullscreen);

        device.destroy(pipelineLayouts.scene);
        device.destroy(pipelineLayouts.fullscreen);

        device.destroy(descriptorSetLayouts.scene);
        device.destroy(descriptorSetLayouts.fullscreen);

        uniformBuffers.scene.destroy();
        triangle.vertices.destroy();
        triangle.indices.destroy();

        device.freeCommandBuffers(cmdPool, offscreenPass.commandBuffer);
        device.destroy(offscreenPass.semaphore);
    }

    void getEnabledFeatures() override {
        // Conservative rasterization setup
        conservativeRasterProps =
            physicalDevice
                .getProperties2KHR<vk::PhysicalDeviceProperties2KHR, vk::PhysicalDeviceConservativeRasterizationPropertiesEXT>(context.dynamicDispatch)
                .get<vk::PhysicalDeviceConservativeRasterizationPropertiesEXT>();

        if (context.deviceFeatures.fillModeNonSolid) {
            context.enabledFeatures.fillModeNonSolid = VK_TRUE;
        }
        if (context.deviceFeatures.wideLines) {
            context.enabledFeatures.wideLines = VK_TRUE;
        }
    }

    /* 
        Setup offscreen framebuffer, attachments and render passes for lower resolution rendering of the scene
    */
    void prepareOffscreen() {
        offscreenPass.width = width / ZOOM_FACTOR;
        offscreenPass.height = height / ZOOM_FACTOR;

        // Find a suitable depth format
        vk::Format fbDepthFormat = context.getSupportedDepthFormat();

        // Color attachment
        vk::ImageCreateInfo image;
        image.imageType = vk::ImageType::e2D;
        image.format = FB_COLOR_FORMAT;
        image.extent.width = offscreenPass.width;
        image.extent.height = offscreenPass.height;
        image.extent.depth = 1;
        image.mipLevels = 1;
        image.arrayLayers = 1;
        // We will sample directly from the color attachment
        image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

        offscreenPass.color = context.createImage(image);

        vk::ImageViewCreateInfo imageView;
        imageView.viewType = vk::ImageViewType::e2D;
        imageView.format = FB_COLOR_FORMAT;
        imageView.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        imageView.image = offscreenPass.color.image;
        offscreenPass.color.view = device.createImageView(imageView);

        // Create sampler to sample from the attachment in the fragment shader
        vk::SamplerCreateInfo samplerInfo;
        samplerInfo.magFilter = vk::Filter::eNearest;
        samplerInfo.minFilter = vk::Filter::eNearest;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeV = samplerInfo.addressModeU;
        samplerInfo.addressModeW = samplerInfo.addressModeU;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 1.0f;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        offscreenPass.sampler = device.createSampler(samplerInfo);

        // Depth stencil attachment
        image.format = fbDepthFormat;
        image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

        offscreenPass.depth = context.createImage(image);

        imageView.viewType = vk::ImageViewType::e2D;
        imageView.format = fbDepthFormat;
        imageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        imageView.image = offscreenPass.depth.image;
        offscreenPass.depth.view = device.createImageView(imageView);

        // Create a separate render pass for the offscreen rendering as it may differ from the one used for scene rendering

        std::array<vk::AttachmentDescription, 2> attchmentDescriptions = {};
        // Color attachment
        attchmentDescriptions[0].format = FB_COLOR_FORMAT;
        attchmentDescriptions[0].loadOp = vk::AttachmentLoadOp::eClear;
        attchmentDescriptions[0].storeOp = vk::AttachmentStoreOp::eStore;
        attchmentDescriptions[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attchmentDescriptions[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attchmentDescriptions[0].initialLayout = vk::ImageLayout::eUndefined;
        attchmentDescriptions[0].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        // Depth attachment
        attchmentDescriptions[1].format = fbDepthFormat;
        attchmentDescriptions[1].loadOp = vk::AttachmentLoadOp::eClear;
        attchmentDescriptions[1].storeOp = vk::AttachmentStoreOp::eDontCare;
        attchmentDescriptions[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attchmentDescriptions[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attchmentDescriptions[1].initialLayout = vk::ImageLayout::eUndefined;
        attchmentDescriptions[1].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };
        vk::AttachmentReference depthReference = { 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };

        vk::SubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;
        subpassDescription.pDepthStencilAttachment = &depthReference;

        // Use subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        // Create the actual renderpass
        vk::RenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
        renderPassInfo.pAttachments = attchmentDescriptions.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();
        offscreenPass.renderPass = device.createRenderPass(renderPassInfo);

        vk::ImageView attachments[2];
        attachments[0] = offscreenPass.color.view;
        attachments[1] = offscreenPass.depth.view;

        vk::FramebufferCreateInfo fbufCreateInfo;
        fbufCreateInfo.renderPass = offscreenPass.renderPass;
        fbufCreateInfo.attachmentCount = 2;
        fbufCreateInfo.pAttachments = attachments;
        fbufCreateInfo.width = offscreenPass.width;
        fbufCreateInfo.height = offscreenPass.height;
        fbufCreateInfo.layers = 1;

        offscreenPass.frameBuffer = device.createFramebuffer(fbufCreateInfo);

        // Fill a descriptor for later use in a descriptor set
        offscreenPass.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        offscreenPass.descriptor.imageView = offscreenPass.color.view;
        offscreenPass.descriptor.sampler = offscreenPass.sampler;
    }

    // Sets up the command buffer that renders the scene to the offscreen frame buffer
    void buildOffscreenCommandBuffer() {
        context.queue.waitIdle();
        context.device.waitIdle();
        if (!offscreenPass.commandBuffer) {
            offscreenPass.commandBuffer = context.allocateCommandBuffers(1, vk::CommandBufferLevel::ePrimary)[0];
        }
        if (!offscreenPass.semaphore) {
            offscreenPass.semaphore = device.createSemaphore(vk::SemaphoreCreateInfo{});
        }

        vk::CommandBufferBeginInfo cmdBufInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse };
        offscreenPass.commandBuffer.begin(cmdBufInfo);

        vk::ClearValue clearValues[2];
        clearValues[0].color = defaultClearColor;
        clearValues[1].depthStencil = defaultClearDepth;

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = offscreenPass.renderPass;
        renderPassBeginInfo.framebuffer = offscreenPass.frameBuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreenPass.width;
        renderPassBeginInfo.renderArea.extent.height = offscreenPass.height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        offscreenPass.commandBuffer.beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);
        vk::Viewport viewport = vk::Viewport(0.0f, 0.0f, (float)offscreenPass.width, (float)offscreenPass.height, 0.0f, 1.0f);
        offscreenPass.commandBuffer.setViewport(0, viewport);
        vk::Rect2D scissor{ vk::Offset2D{}, vk::Extent2D{ offscreenPass.width, offscreenPass.height } };
        offscreenPass.commandBuffer.setScissor(0, scissor);
        offscreenPass.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        offscreenPass.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                                 conservativeRasterEnabled ? pipelines.triangleConservativeRaster : pipelines.triangle);
        offscreenPass.commandBuffer.bindVertexBuffers(0, triangle.vertices.buffer, { 0 });
        offscreenPass.commandBuffer.bindIndexBuffer(triangle.indices.buffer, 0, vk::IndexType::eUint32);
        offscreenPass.commandBuffer.drawIndexed(triangle.indexCount, 1, 0, 0, 0);

        offscreenPass.commandBuffer.endRenderPass();
        offscreenPass.commandBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());

        // Low-res triangle from offscreen framebuffer
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.fullscreen);
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.fullscreen, 0, 1, &descriptorSets.fullscreen, 0, nullptr);
        drawCmdBuffer.draw(3, 1, 0, 0);

        // Overlay actual triangle
        VkDeviceSize offsets[1] = { 0 };
        drawCmdBuffer.bindVertexBuffers(0, triangle.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(triangle.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.triangleOverlay);
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        drawCmdBuffer.draw(3, 1, 0, 0);
    }

    void loadAssets() override {
        // Create a single triangle
        struct Vertex {
            float position[3];
            float color[3];
        };

        std::vector<Vertex> vertexBuffer = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
        uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
        triangle.indexCount = static_cast<uint32_t>(indexBuffer.size());
        triangle.vertices = context.stageToDeviceBuffer<Vertex>(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
        triangle.indices = context.stageToDeviceBuffer<uint32_t>(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 3 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 2 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Scene rendering
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.scene = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.scene = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.scene });

        // Fullscreen pass
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.fullscreen = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.fullscreen = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.fullscreen });
    }

    void setupDescriptorSet() {
        // Scene rendering
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.scene })[0];
        // Fullscreen pass
        descriptorSets.fullscreen = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.fullscreen })[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            vk::WriteDescriptorSet{ descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.scene.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.fullscreen, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &offscreenPass.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayouts.fullscreen, renderPass };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        builder.depthStencilState = false;

        // Conservative rasterization pipeline state
        vk::PipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterStateCI{};
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eOverestimate;
        conservativeRasterStateCI.extraPrimitiveOverestimationSize = conservativeRasterProps.maxExtraPrimitiveOverestimationSize;
        // Conservative rasterization state has to be chained into the pipeline rasterization state create info structure
        builder.rasterizationState.pNext = &conservativeRasterStateCI;

        // Full screen pass
        // Empty vertex input state (full screen triangle generated in vertex shader)
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eDisabled;
        builder.loadShader(getAssetPath() + "shaders/conservativeraster/fullscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/conservativeraster/fullscreen.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.fullscreen = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Vertex bindings and attributes
        builder.layout = pipelineLayouts.scene;
        builder.vertexInputState.bindingDescriptions = {
            { 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
        };
        builder.vertexInputState.attributeDescriptions = {
            { 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) },
            { 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) },
        };
        // Original triangle outline (no conservative rasterization)
        // TODO: Check support for lines
        builder.rasterizationState.lineWidth = 2.0f;
        builder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eDisabled;

        builder.loadShader(getAssetPath() + "shaders/conservativeraster/triangle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/conservativeraster/triangleoverlay.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.triangleOverlay = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        builder.renderPass = offscreenPass.renderPass;
        // Triangle rendering
        builder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eDisabled;
        builder.loadShader(getAssetPath() + "shaders/conservativeraster/triangle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/conservativeraster/triangle.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Default
        pipelines.triangle = builder.create(context.pipelineCache);

        // Conservative rasterization enabled
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eOverestimate;
        pipelines.triangleConservativeRaster = builder.create(context.pipelineCache);
        builder.destroyShaderModules();
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        uniformBuffers.scene = context.createUniformBuffer(uboScene);
        updateUniformBuffersScene();
    }

    void updateUniformBuffersScene() {
        uboScene.projection = camera.matrices.perspective;
        uboScene.model = camera.matrices.view;
        memcpy(uniformBuffers.scene.mapped, &uboScene, sizeof(uboScene));
    }

    void draw() override {
        ExampleBase::prepareFrame();

        // Offscreen rendering
        context.submit(offscreenPass.commandBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreenPass.semaphore);

        // Scene rendering
        renderWaitSemaphores = { offscreenPass.semaphore };
        drawCurrentCommandBuffer();

        ExampleBase::submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareOffscreen();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffersScene(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Conservative rasterization", &conservativeRasterEnabled)) {
                buildOffscreenCommandBuffer();
            }
        }
        if (ui.header("Device properties")) {
            ui.text("maxExtraPrimitiveOverestimationSize:         %f", conservativeRasterProps.maxExtraPrimitiveOverestimationSize);
            ui.text("extraPrimitiveOverestimationSizeGranularity: %f", conservativeRasterProps.extraPrimitiveOverestimationSizeGranularity);
            ui.text("primitiveUnderestimation:                    %s", conservativeRasterProps.primitiveUnderestimation ? "yes" : "no");
            ui.text("conservativePointAndLineRasterization:       %s", conservativeRasterProps.conservativePointAndLineRasterization ? "yes" : "no");
            ui.text("degenerateTrianglesRasterized:               %s", conservativeRasterProps.degenerateTrianglesRasterized ? "yes" : "no");
            ui.text("degenerateLinesRasterized:                   %s", conservativeRasterProps.degenerateLinesRasterized ? "yes" : "no");
            ui.text("fullyCoveredFragmentShaderInputVariable:     %s", conservativeRasterProps.fullyCoveredFragmentShaderInputVariable ? "yes" : "no");
            ui.text("conservativeRasterizationPostDepthCoverage:  %s", conservativeRasterProps.conservativeRasterizationPostDepthCoverage ? "yes" : "no");
        }
    }
};

VULKAN_EXAMPLE_MAIN()
