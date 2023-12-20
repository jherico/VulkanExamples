/*
* Vulkan Example - Multisampling using resolve attachments
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

#define SAMPLE_COUNT vk::SampleCountFlagBits::e4

struct {
    vks::Image color;
    vks::Image depth;
} multisampleTarget;

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::ExampleBase {
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

    // UI overlay configuration needs to be adjusted for this example (renderpass setup, attachment count, etc.)
    void OnSetupUIOverlay(vkx::ui::UIOverlayCreateInfo& createInfo) override {
        createInfo.renderPass = uiRenderPass;
        createInfo.framebuffers = framebuffers;
        createInfo.rasterizationSamples = SAMPLE_COUNT;
        createInfo.attachmentCount = 1;
        createInfo.clearValues = {
            vk::ClearValue{ vks::util::clearColor(glm::vec4(1.0f)) },
            vk::ClearValue{ vks::util::clearColor(glm::vec4(1.0f)) },
            vk::ClearValue{ vk::ClearDepthStencilValue{ 1.0f, 0 } },
        };
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.solid);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        meshes.example.destroy();

        // Destroy MSAA target
        device.destroyImage(multisampleTarget.color.image);
        device.destroyImageView(multisampleTarget.color.view);
        device.freeMemory(multisampleTarget.color.memory);
        device.destroyImage(multisampleTarget.depth.image);
        device.destroyImageView(multisampleTarget.depth.view);
        device.freeMemory(multisampleTarget.depth.memory);

        textures.colorMap.destroy();

        uniformData.vsScene.destroy();
    }

    // Creates a multi sample render target (image and view) that is used to resolve
    // into the visible frame buffer target in the render pass
    void setupMultisampleTarget() {
        // Check if device supports requested sample count for color and depth frame buffer
        vk::SampleCountFlags colorSampleCount = context.deviceProperties.limits.framebufferColorSampleCounts;
        vk::SampleCountFlags depthSampleCount = context.deviceProperties.limits.framebufferDepthSampleCounts;
        vk::SampleCountFlags requiredSamples = SAMPLE_COUNT;
        assert((uint32_t)colorSampleCount >= (uint32_t)requiredSamples && (uint32_t)depthSampleCount >= (uint32_t)requiredSamples);

        // Color target
        vk::ImageCreateInfo info;
        info.imageType = vk::ImageType::e2D;
        info.format = colorformat;
        info.extent.width = size.width;
        info.extent.height = size.height;
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.sharingMode = vk::SharingMode::eExclusive;
        info.tiling = vk::ImageTiling::eOptimal;
        info.samples = SAMPLE_COUNT;
        // vk::Image will only be used as a transient target
        info.usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment;
        info.initialLayout = vk::ImageLayout::eUndefined;
        multisampleTarget.color = context.createImage(info, vk::MemoryPropertyFlagBits::eDeviceLocal);

        //// We prefer a lazily allocated memory type
        //// This means that the memory get allocated when the implementation sees fit, e.g. when first using the images
        //vk::Bool32 lazyMemType = getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eLazilyAllocated, &memAlloc.memoryTypeIndex);
        //if (!lazyMemType) {
        //    // If this is not available, fall back to device local memory
        //    getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, &memAlloc.memoryTypeIndex);
        //}

        // Create image view for the MSAA target
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.image = multisampleTarget.color.image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = colorformat;
        viewInfo.components.r = vk::ComponentSwizzle::eR;
        viewInfo.components.g = vk::ComponentSwizzle::eG;
        viewInfo.components.b = vk::ComponentSwizzle::eB;
        viewInfo.components.a = vk::ComponentSwizzle::eA;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        multisampleTarget.color.view = device.createImageView(viewInfo);

        // Depth target
        info.imageType = vk::ImageType::e2D;
        info.format = depthFormat;
        info.extent.width = size.width;
        info.extent.height = size.height;
        info.extent.depth = 1;
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.sharingMode = vk::SharingMode::eExclusive;
        info.tiling = vk::ImageTiling::eOptimal;
        info.samples = SAMPLE_COUNT;
        // vk::Image will only be used as a transient target
        info.usage = vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment;
        info.initialLayout = vk::ImageLayout::eUndefined;
        multisampleTarget.depth = context.createImage(info, vk::MemoryPropertyFlagBits::eDeviceLocal);

        // Create image view for the MSAA target
        viewInfo.image = multisampleTarget.depth.image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = depthFormat;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;

        multisampleTarget.depth.view = device.createImageView(viewInfo);

        // Initial image layout transitions
        // We need to transform the MSAA target layouts before using them
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& setupCmdBuffer) {
            // Tansform MSAA color target
            context.setImageLayout(setupCmdBuffer, multisampleTarget.color.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eColorAttachmentOptimal);

            // Tansform MSAA depth target
            context.setImageLayout(setupCmdBuffer, multisampleTarget.depth.image, vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
                                   vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        });
    }

    // Setup a render pass for using a multi sampled attachment
    // and a resolve attachment that the msaa image is resolved
    // to at the end of the render pass
    void setupRenderPass() override {
        // Overrides the virtual function of the base class

        std::array<vk::AttachmentDescription, 4> attachments = {};

        // Multisampled attachment that we render to
        attachments[0].format = colorformat;
        attachments[0].samples = SAMPLE_COUNT;
        attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
        // No longer required after resolve, this may save some bandwidth on certain GPUs
        attachments[0].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachments[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachments[0].initialLayout = vk::ImageLayout::eColorAttachmentOptimal;
        attachments[0].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

        // This is the frame buffer attachment to where the multisampled image
        // will be resolved to and which will be presented to the swapchain
        attachments[1].format = colorformat;
        attachments[1].samples = vk::SampleCountFlagBits::e1;
        attachments[1].loadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[1].storeOp = vk::AttachmentStoreOp::eStore;
        attachments[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachments[1].initialLayout = vk::ImageLayout::eUndefined;
        attachments[1].finalLayout = vk::ImageLayout::ePresentSrcKHR;

        // Multisampled depth attachment we render to
        attachments[2].format = depthFormat;
        attachments[2].samples = SAMPLE_COUNT;
        attachments[2].loadOp = vk::AttachmentLoadOp::eClear;
        attachments[2].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachments[2].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[2].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachments[2].initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        attachments[2].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        // Depth resolve attachment
        attachments[3].format = depthFormat;
        attachments[3].samples = vk::SampleCountFlagBits::e1;
        attachments[3].loadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[3].storeOp = vk::AttachmentStoreOp::eStore;
        attachments[3].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachments[3].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachments[3].initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        attachments[3].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::AttachmentReference colorReference;
        colorReference.attachment = 0;
        colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::AttachmentReference depthReference;
        depthReference.attachment = 2;
        depthReference.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        // Two resolve attachment references for color and depth
        std::array<vk::AttachmentReference, 2> resolveReferences = {};
        resolveReferences[0].attachment = 1;
        resolveReferences[0].layout = vk::ImageLayout::eColorAttachmentOptimal;
        resolveReferences[1].attachment = 3;
        resolveReferences[1].layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        // Pass our resolve attachments to the sub pass
        subpass.pResolveAttachments = resolveReferences.data();
        subpass.pDepthStencilAttachment = &depthReference;

        std::vector<vk::SubpassDependency> dependencies{ { 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eBottomOfPipe,
                                                           vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite,
                                                           vk::AccessFlagBits::eColorAttachmentRead } };

        vk::RenderPassCreateInfo renderPassInfo;
        renderPassInfo.attachmentCount = (uint32_t)attachments.size();
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.dependencyCount = (uint32_t)dependencies.size();
        renderPassInfo.pDependencies = dependencies.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        renderPass = device.createRenderPass(renderPassInfo);
    }

    // Frame buffer attachments must match with render pass setup,
    // so we need to adjust frame buffer creation to cover our
    // multisample target
    void setupFrameBuffer() override {
        // Overrides the virtual function of the base class
        std::array<vk::ImageView, 4> attachments;

        setupMultisampleTarget();

        attachments[0] = multisampleTarget.color.view;
        // attachment[1] = swapchain image
        attachments[2] = multisampleTarget.depth.view;
        attachments[3] = depthStencil.view;

        vk::FramebufferCreateInfo framebufferCreateInfo;
        framebufferCreateInfo.renderPass = renderPass;
        framebufferCreateInfo.attachmentCount = (uint32_t)attachments.size();
        framebufferCreateInfo.pAttachments = attachments.data();
        framebufferCreateInfo.width = size.width;
        framebufferCreateInfo.height = size.height;
        framebufferCreateInfo.layers = 1;

        // Create frame buffers for every swap chain image
        framebuffers.resize(swapChain.imageCount);
        for (uint32_t i = 0; i < framebuffers.size(); i++) {
            attachments[1] = swapChain.images[i].view;
            framebuffers[i] = device.createFramebuffer(framebufferCreateInfo);
        }
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        //vk::CommandBufferBeginInfo cmdBufInfo;

        //vk::ClearValue clearValues[3];
        //// Clear to a white background for higher contrast
        //clearValues[0].color = vkx::clearColor({ 1.0f, 1.0f, 1.0f, 1.0f });
        //clearValues[1].color = vkx::clearColor({ 1.0f, 1.0f, 1.0f, 1.0f });
        //clearValues[2].depthStencil = { 1.0f, 0 };

        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);

        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, meshes.example.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.example.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(meshes.example.indexCount, 1, 0, 0, 0);
    }

    void setupRenderPassBeginInfo() override {
        clearValues.clear();
        clearValues.push_back(vks::util::clearColor(glm::vec4(1)));
        clearValues.push_back(vks::util::clearColor(glm::vec4(1)));
        clearValues.push_back(vk::ClearDepthStencilValue{ 1.0f, 0 });
        renderPassBeginInfo = vk::RenderPassBeginInfo{ renderPass, {}, { {}, size }, (uint32_t)clearValues.size(), clearValues.data() };
    }

    void loadAssets() override {
        textures.colorMap.loadFromFile(context, getAssetPath() + "models/voyager/voyager.ktx", vk::Format::eBc3UnormBlock);
        meshes.example.loadFromFile(context, getAssetPath() + "models/voyager/voyager.dae", vertexLayout);
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
        vk::DescriptorImageInfo texDescriptor{ textures.colorMap.sampler, textures.colorMap.view, vk::ImageLayout::eGeneral };
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
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.multisampleState.rasterizationSamples = SAMPLE_COUNT;
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        // Load shaders
        pipelineBuilder.loadShader(getAssetPath() + "shaders/mesh/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/mesh/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uniformData.vsScene.copy(uboVS);
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
};

RUN_EXAMPLE(VulkanExample)
