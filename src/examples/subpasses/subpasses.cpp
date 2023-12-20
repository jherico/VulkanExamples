/*
* Vulkan Example - Using subpasses for G-Buffer compositing
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
* Summary:
* Implements a deferred rendering setup with a forward transparency pass using sub passes
*
* Sub passes allow reading from the previous framebuffer (in the same render pass) at 
* the same pixel position.
* 
* This is a feature that was especially designed for tile-based-renderers 
* (mostly mobile GPUs) and is a new optomization feature in Vulkan for those GPU types.
*
*/

#include <vulkanExampleBase.h>

#define NUM_LIGHTS 64

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::texture::Texture2D glass;
    } textures;

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_COLOR,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_UV,
    });

    struct {
        vks::model::Model scene;
        vks::model::Model transparent;
    } models;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
    } uboGBuffer;

    struct Light {
        glm::vec4 position;
        glm::vec3 color;
        float radius;
    };

    struct {
        glm::vec4 viewPos;
        Light lights[NUM_LIGHTS];
    } uboLights;

    struct {
        vks::Buffer GBuffer;
        vks::Buffer lights;
    } uniformBuffers;

    struct {
        vk::Pipeline offscreen;
        vk::Pipeline composition;
        vk::Pipeline transparent;
    } pipelines;

    struct {
        vk::PipelineLayout offscreen;
        vk::PipelineLayout composition;
        vk::PipelineLayout transparent;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet scene;
        vk::DescriptorSet composition;
        vk::DescriptorSet transparent;
    } descriptorSets;

    struct {
        vk::DescriptorSetLayout scene;
        vk::DescriptorSetLayout composition;
        vk::DescriptorSetLayout transparent;
    } descriptorSetLayouts;

    // G-Buffer framebuffer attachments
    struct Attachments {
        vks::Image position, normal, albedo;
    } attachments;

    VkRenderPass uiRenderPass;

    VulkanExample() {
        title = "Subpasses";
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
        camera.rotationSpeed = 0.25f;
#endif
        camera.setPosition(glm::vec3(-3.2f, 1.0f, 5.9f));
        camera.setRotation(glm::vec3(0.5f, 210.05f, 0.0f));
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
        settings.overlay = true;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        attachments.position.destroy();
        attachments.normal.destroy();
        attachments.albedo.destroy();

        device.destroy(pipelines.offscreen);
        device.destroy(pipelines.composition);
        device.destroy(pipelines.transparent);

        device.destroy(pipelineLayouts.offscreen);
        device.destroy(pipelineLayouts.composition);
        device.destroy(pipelineLayouts.transparent);

        device.destroy(descriptorSetLayouts.scene);
        device.destroy(descriptorSetLayouts.composition);
        device.destroy(descriptorSetLayouts.transparent);
        device.destroy(uiRenderPass);

        textures.glass.destroy();
        models.scene.destroy();
        models.transparent.destroy();
        uniformBuffers.GBuffer.destroy();
        uniformBuffers.lights.destroy();
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        // Enable anisotropic filtering if supported
        if (deviceFeatures.samplerAnisotropy) {
            enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
        // Enable texture compression
        if (deviceFeatures.textureCompressionBC) {
            enabledFeatures.textureCompressionBC = VK_TRUE;
        } else if (deviceFeatures.textureCompressionASTC_LDR) {
            enabledFeatures.textureCompressionASTC_LDR = VK_TRUE;
        } else if (deviceFeatures.textureCompressionETC2) {
            enabledFeatures.textureCompressionETC2 = VK_TRUE;
        }
    };

    // Create a frame buffer attachment
    void createAttachment(vk::Format format, vk::ImageUsageFlags usage, vks::Image& attachment) {
        vk::ImageAspectFlags aspectMask;
        vk::ImageLayout imageLayout;

        attachment.format = format;

        if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eColor;
            imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }
        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        }

        assert(aspectMask.operator VkInstanceCreateFlags() > 0);

        vk::ImageCreateInfo image;
        image.imageType = vk::ImageType::e2D;
        image.format = format;
        image.extent.width = width;
        image.extent.height = height;
        image.extent.depth = 1;
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.usage = usage | vk::ImageUsageFlagBits::eInputAttachment;  // vk::ImageUsageFlagBits::eInputAttachment flag is required for input attachments;
        image.initialLayout = vk::ImageLayout::eUndefined;

        attachment = context.createImage(image);

        vk::ImageViewCreateInfo imageView;
        imageView.viewType = vk::ImageViewType::e2D;
        imageView.format = format;
        imageView.subresourceRange = { aspectMask, 0, 1, 0, 1 };
        imageView.image = attachment.image;
        attachment.view = device.createImageView(imageView);
    }

    // Create color attachments for the G-Buffer components
    void createGBufferAttachments() {
        createAttachment(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment, attachments.position);  // (World space) Positions
        createAttachment(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment, attachments.normal);    // (World space) Normals
        createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, attachments.albedo);         // Albedo (color)
    }

    // Override framebuffer setup from base class
    // Deferred components will be used as frame buffer attachments
    void setupFrameBuffer() override {
        vk::ImageView attachmentsDescs[5];

        vk::FramebufferCreateInfo frameBufferCreateInfo = {};
        frameBufferCreateInfo.renderPass = renderPass;
        frameBufferCreateInfo.attachmentCount = 5;
        frameBufferCreateInfo.pAttachments = attachmentsDescs;
        frameBufferCreateInfo.width = width;
        frameBufferCreateInfo.height = height;
        frameBufferCreateInfo.layers = 1;

        // Create frame buffers for every swap chain image
        framebuffers.resize(swapChain.imageCount);
        for (uint32_t i = 0; i < framebuffers.size(); i++) {
            attachmentsDescs[0] = swapChain.images[i].view;
            attachmentsDescs[1] = this->attachments.position.view;
            attachmentsDescs[2] = this->attachments.normal.view;
            attachmentsDescs[3] = this->attachments.albedo.view;
            attachmentsDescs[4] = depthStencil.view;
            framebuffers[i] = device.createFramebuffer(frameBufferCreateInfo);
        }
    }

    // Override render pass setup from base class
    void setupRenderPass() override {
        createGBufferAttachments();

        std::array<vk::AttachmentDescription, 5> attachmentsDescs{};
        // Color attachment
        attachmentsDescs[0].format = swapChain.colorFormat;
        attachmentsDescs[0].loadOp = vk::AttachmentLoadOp::eClear;
        attachmentsDescs[0].storeOp = vk::AttachmentStoreOp::eStore;
        attachmentsDescs[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachmentsDescs[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[0].initialLayout = vk::ImageLayout::eUndefined;
        attachmentsDescs[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

        // Deferred attachments
        // Position
        attachmentsDescs[1].format = this->attachments.position.format;
        attachmentsDescs[1].loadOp = vk::AttachmentLoadOp::eClear;
        attachmentsDescs[1].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachmentsDescs[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[1].initialLayout = vk::ImageLayout::eUndefined;
        attachmentsDescs[1].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        // Normals
        attachmentsDescs[2].format = this->attachments.normal.format;
        attachmentsDescs[2].loadOp = vk::AttachmentLoadOp::eClear;
        attachmentsDescs[2].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[2].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachmentsDescs[2].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[2].initialLayout = vk::ImageLayout::eUndefined;
        attachmentsDescs[2].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        // Albedo
        attachmentsDescs[3].format = this->attachments.albedo.format;
        attachmentsDescs[3].loadOp = vk::AttachmentLoadOp::eClear;
        attachmentsDescs[3].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[3].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachmentsDescs[3].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[3].initialLayout = vk::ImageLayout::eUndefined;
        attachmentsDescs[3].finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        // Depth attachment
        attachmentsDescs[4].format = depthFormat;
        attachmentsDescs[4].loadOp = vk::AttachmentLoadOp::eClear;
        attachmentsDescs[4].storeOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[4].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachmentsDescs[4].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachmentsDescs[4].initialLayout = vk::ImageLayout::eUndefined;
        attachmentsDescs[4].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

        // Three subpasses
        std::array<vk::SubpassDescription, 3> subpassDescriptions{};

        // First subpass: Fill G-Buffer components
        // ----------------------------------------------------------------------------------------

        vk::AttachmentReference colorReferences[4];
        colorReferences[0] = { 0, vk::ImageLayout::eColorAttachmentOptimal };
        colorReferences[1] = { 1, vk::ImageLayout::eColorAttachmentOptimal };
        colorReferences[2] = { 2, vk::ImageLayout::eColorAttachmentOptimal };
        colorReferences[3] = { 3, vk::ImageLayout::eColorAttachmentOptimal };
        vk::AttachmentReference depthReference{ 4, vk::ImageLayout::eDepthStencilAttachmentOptimal };

        subpassDescriptions[0].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescriptions[0].colorAttachmentCount = 4;
        subpassDescriptions[0].pColorAttachments = colorReferences;
        subpassDescriptions[0].pDepthStencilAttachment = &depthReference;

        // Second subpass: Final composition (using G-Buffer components)
        // ----------------------------------------------------------------------------------------

        vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

        vk::AttachmentReference inputReferences[3];
        inputReferences[0] = { 1, vk::ImageLayout::eShaderReadOnlyOptimal };
        inputReferences[1] = { 2, vk::ImageLayout::eShaderReadOnlyOptimal };
        inputReferences[2] = { 3, vk::ImageLayout::eShaderReadOnlyOptimal };

        uint32_t preserveAttachmentIndex = 1;

        subpassDescriptions[1].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescriptions[1].colorAttachmentCount = 1;
        subpassDescriptions[1].pColorAttachments = &colorReference;
        subpassDescriptions[1].pDepthStencilAttachment = &depthReference;
        // Use the color attachments filled in the first pass as input attachments
        subpassDescriptions[1].inputAttachmentCount = 3;
        subpassDescriptions[1].pInputAttachments = inputReferences;

        // Third subpass: Forward transparency
        // ----------------------------------------------------------------------------------------
        colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

        inputReferences[0] = { 1, vk::ImageLayout::eShaderReadOnlyOptimal };

        subpassDescriptions[2].pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescriptions[2].colorAttachmentCount = 1;
        subpassDescriptions[2].pColorAttachments = &colorReference;
        subpassDescriptions[2].pDepthStencilAttachment = &depthReference;
        // Use the color/depth attachments filled in the first pass as input attachments
        subpassDescriptions[2].inputAttachmentCount = 1;
        subpassDescriptions[2].pInputAttachments = inputReferences;

        // Subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 4> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        // This dependency transitions the input attachment from color attachment to shader read
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = 1;
        dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
        dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[1].dstAccessMask = vk::AccessFlagBits::eShaderRead;
        dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        dependencies[2].srcSubpass = 1;
        dependencies[2].dstSubpass = 2;
        dependencies[2].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[2].dstStageMask = vk::PipelineStageFlagBits::eFragmentShader;
        dependencies[2].srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[2].dstAccessMask = vk::AccessFlagBits::eShaderRead;
        dependencies[2].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        dependencies[3].srcSubpass = 0;
        dependencies[3].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[3].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[3].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[3].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[3].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[3].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        vk::RenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentsDescs.size());
        renderPassInfo.pAttachments = attachmentsDescs.data();
        renderPassInfo.subpassCount = static_cast<uint32_t>(subpassDescriptions.size());
        renderPassInfo.pSubpasses = subpassDescriptions.data();
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

        renderPass = device.createRenderPass(renderPassInfo);

        // Create custom overlay render pass
        attachmentsDescs[0].loadOp = vk::AttachmentLoadOp::eLoad;
        uiRenderPass = device.createRenderPass(renderPassInfo);
    }

    void setupRenderPassBeginInfo() {
        clearValues.clear();
        clearValues.push_back(vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }));
        clearValues.push_back(vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }));
        clearValues.push_back(vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }));
        clearValues.push_back(vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }));
        clearValues.push_back(vk::ClearDepthStencilValue{ 1.0f, 0 });

        renderPassBeginInfo = vk::RenderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent = size;
        renderPassBeginInfo.clearValueCount = (uint32_t)clearValues.size();
        renderPassBeginInfo.pClearValues = clearValues.data();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());

        // First sub pass
        // Renders the components of the scene to the G-Buffer atttachments
        {
            vks::debug::marker::beginRegion(drawCmdBuffer, "Subpass 0: Deferred G-Buffer creation", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
            drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, descriptorSets.scene, nullptr);
            drawCmdBuffer.bindVertexBuffers(0, models.scene.vertices.buffer, { 0 });
            drawCmdBuffer.bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
            drawCmdBuffer.drawIndexed(models.scene.indexCount, 1, 0, 0, 0);
            vks::debug::marker::endRegion(drawCmdBuffer);
        }

        // Second sub pass
        // This subpass will use the G-Buffer components that have been filled in the first subpass as input attachment for the final compositing
        {
            vks::debug::marker::beginRegion(drawCmdBuffer, "Subpass 1: Deferred composition", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            drawCmdBuffer.nextSubpass(vk::SubpassContents::eInline);
            drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
            drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, nullptr);
            drawCmdBuffer.draw(3, 1, 0, 0);
            vks::debug::marker::endRegion(drawCmdBuffer);
        }

        // Third subpass
        // Render transparent geometry using a forward pass that compares against depth generted during G-Buffer fill
        {
            vks::debug::marker::beginRegion(drawCmdBuffer, "Subpass 2: Forward transparency", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            drawCmdBuffer.nextSubpass(vk::SubpassContents::eInline);
            drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.transparent);
            drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.transparent, 0, descriptorSets.transparent, nullptr);
            drawCmdBuffer.bindVertexBuffers(0, models.transparent.vertices.buffer, { 0 });
            drawCmdBuffer.bindIndexBuffer(models.transparent.indices.buffer, 0, vk::IndexType::eUint32);
            drawCmdBuffer.drawIndexed(models.transparent.indexCount, 1, 0, 0, 0);
            vks::debug::marker::endRegion(drawCmdBuffer);
        }
    }

    void loadAssets() {
        models.scene.loadFromFile(context, getAssetPath() + "models/samplebuilding.dae", vertexLayout, 1.0f);
        models.transparent.loadFromFile(context, getAssetPath() + "models/samplebuilding_glass.dae", vertexLayout, 1.0f);
        // Textures
        if (deviceFeatures.textureCompressionBC) {
            textures.glass.loadFromFile(context, getAssetPath() + "textures/colored_glass_bc3_unorm.ktx", vk::Format::eBc3UnormBlock);
        } else if (deviceFeatures.textureCompressionASTC_LDR) {
            textures.glass.loadFromFile(context, getAssetPath() + "textures/colored_glass_astc_8x8_unorm.ktx", vk::Format::eAstc8x8UnormBlock);
        } else if (deviceFeatures.textureCompressionETC2) {
            textures.glass.loadFromFile(context, getAssetPath() + "textures/colored_glass_etc2_unorm.ktx", vk::Format::eEtc2R8G8B8A8UnormBlock);
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }
    }

/*
    void setupVertexDescriptions() {
        // Binding description
        vertices.bindingDescriptions = {
            vk::vertexInputBindingDescription(0, vertexLayout.stride(), VK_VERTEX_INPUT_RATE_VERTEX),
        };

        // Attribute descriptions
        vertices.attributeDescriptions = {
            // Location 0: Position
            vk::vertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),
            // Location 1: Color
            vk::vertexInputAttributeDescription(0, 1, vk::Format::eR32G32B32Sfloat, sizeof(float) * 3),
            // Location 2: Normal
            vk::vertexInputAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, sizeof(float) * 6),
            // Location 3: UV
            vk::vertexInputAttributeDescription(0, 3, vk::Format::eR32G32Sfloat, sizeof(float) * 9),
        };

        vertices.inputState = vk::pipelineVertexInputStateCreateInfo();
        vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
        vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
        vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
        vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
    }
    */

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 9 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 9 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eInputAttachment, 4 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 4, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Deferred shading layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayouts.scene = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.offscreen = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.scene });

        // Composition pass
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.composition = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.composition = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.composition });

        // Transparent (forward) pass
        // Descriptor set layout
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.transparent = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.transparent = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.transparent });
    }

    void setupDescriptorSet() {
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.scene })[0];
        descriptorSets.composition = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.composition })[0];
        descriptorSets.transparent = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.transparent })[0];

        vk::DescriptorImageInfo texDescriptorPosition{ nullptr, attachments.position.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorNormal{ nullptr, attachments.normal.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorAlbedo{ nullptr, attachments.albedo.view, vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            vk::WriteDescriptorSet{ descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.GBuffer.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.composition, 0, 0, 1, vk::DescriptorType::eInputAttachment, &texDescriptorPosition},
            vk::WriteDescriptorSet{ descriptorSets.composition, 1, 0, 1, vk::DescriptorType::eInputAttachment, &texDescriptorNormal},
            vk::WriteDescriptorSet{ descriptorSets.composition, 2, 0, 1, vk::DescriptorType::eInputAttachment, &texDescriptorAlbedo},
            vk::WriteDescriptorSet{ descriptorSets.composition, 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.lights.descriptor},
            vk::WriteDescriptorSet{ descriptorSets.transparent, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.GBuffer.descriptor},
            vk::WriteDescriptorSet{ descriptorSets.transparent, 1, 0, 1, vk::DescriptorType::eInputAttachment, &texDescriptorPosition},
            vk::WriteDescriptorSet{ descriptorSets.transparent, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.glass.descriptor},
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Deferred pass
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayouts.offscreen, renderPass };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // 4 color attachments, no blending
        builder.colorBlendState.blendAttachmentStates.resize(4);
        // Offscreen scene rendering pipeline
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.loadShader(getAssetPath() + "shaders/subpasses/gbuffer.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/subpasses/gbuffer.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.offscreen = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Composition pass
        builder.layout = pipelineLayouts.composition;
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        builder.colorBlendState.blendAttachmentStates.resize(1);
        builder.vertexInputState = vks::pipelines::PipelineVertexInputStateCreateInfo{};
        builder.loadShader(getAssetPath() + "shaders/subpasses/composition.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/subpasses/composition.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Use specialization constants to pass number of lights to the shader
        vk::SpecializationMapEntry specializationEntry{ 0, 0, sizeof(uint32_t) };
        uint32_t specializationData = NUM_LIGHTS;
        vk::SpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = 1;
        specializationInfo.pMapEntries = &specializationEntry;
        specializationInfo.dataSize = sizeof(specializationData);
        specializationInfo.pData = &specializationData;
        // Index of the subpass that this pipeline will be used in
        builder.subpass = 1;
        builder.depthStencilState.depthWriteEnable = VK_FALSE;
        builder.shaderStages[1].pSpecializationInfo = &specializationInfo;
        pipelines.composition = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Transparent (forward) pipeline
        {
            auto& blendAttachmentState = builder.colorBlendState.blendAttachmentStates[0];
            // Enable blending
            blendAttachmentState.blendEnable = VK_TRUE;
            blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
            blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
            blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
            blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        }
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.layout = pipelineLayouts.transparent;
        builder.subpass = 2;
        builder.loadShader(getAssetPath() + "shaders/subpasses/transparent.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/subpasses/transparent.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.transparent = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Deferred vertex shader
        uniformBuffers.GBuffer = context.createUniformBuffer(uboGBuffer);

        // Deferred fragment shader
        uniformBuffers.lights = context.createUniformBuffer(uboLights);

        // Update
        updateUniformBufferDeferredMatrices();
        updateUniformBufferDeferredLights();
    }

    void updateUniformBufferDeferredMatrices() {
        uboGBuffer.projection = camera.matrices.perspective;
        uboGBuffer.view = camera.matrices.view;
        uboGBuffer.model = glm::mat4(1.0f);

        memcpy(uniformBuffers.GBuffer.mapped, &uboGBuffer, sizeof(uboGBuffer));
    }

    void initLights() {
        std::vector<glm::vec3> colors = {
            glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 0.0f),
        };

        std::mt19937 rndGen((unsigned)time(NULL));
        std::uniform_real_distribution<float> rndDist(-1.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> rndCol(0, static_cast<uint32_t>(colors.size() - 1));

        for (auto& light : uboLights.lights) {
            light.position = glm::vec4(rndDist(rndGen) * 6.0f, 0.25f + std::abs(rndDist(rndGen)) * 4.0f, rndDist(rndGen) * 6.0f, 1.0f);
            light.color = colors[rndCol(rndGen)];
            light.radius = 1.0f + std::abs(rndDist(rndGen));
        }
    }

    // Update fragment shader light position uniform block
    void updateUniformBufferDeferredLights() {
        // Current view position
        uboLights.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

        memcpy(uniformBuffers.lights.mapped, &uboLights, sizeof(uboLights));
    }

    void draw() {
        ExampleBase::prepareFrame();
        drawCurrentCommandBuffer();
        ExampleBase::submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        initLights();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override {
        updateUniformBufferDeferredMatrices();
        updateUniformBufferDeferredLights();
    }

    // UI overlay configuration needs to be adjusted for this example (renderpass setup, attachment count, etc.)
    void OnSetupUIOverlay(vkx::ui::UIOverlayCreateInfo& createInfo) {
        createInfo.renderPass = uiRenderPass;
        createInfo.framebuffers = framebuffers;
        createInfo.subpassCount = 3;
        createInfo.attachmentCount = 4;
        createInfo.clearValues = {
            vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }), 
            vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }),
            vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }),
            vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f }), 
            vk::ClearDepthStencilValue{ 1.0f, 0 },
        };
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Subpasses")) {
            ui.text("0: Deferred G-Buffer creation");
            ui.text("1: Deferred composition");
            ui.text("2: Forward transparency");
        }
        if (ui.header("Settings")) {
            if (ui.button("Randomize lights")) {
                initLights();
                updateUniformBufferDeferredLights();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
