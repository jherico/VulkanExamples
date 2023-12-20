/*
* Vulkan Example - Multi sampling with explicit resolve for deferred shading example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"

// todo: check if hardware supports sample number (or select max. supported)
#define SAMPLE_COUNT vk::SampleCountFlagBits::e8;

class VulkanExample : public vkx::ExampleBase {
public:
    bool debugDisplay = false;
    bool useMSAA = true;
    bool useSampleShading = true;

    struct Material {
        vks::texture::Texture2D colorMap;
        vks::texture::Texture2D normalMap;
    };

    struct {
        Material model;
        Material floor;
    } textures;

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_TANGENT,
    });

    struct {
        vks::model::Model model;
        vks::model::Model floor;
        vks::model::Model quad;
    } models;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
        glm::vec4 instancePos[3];
    } uboVS, uboOffscreenVS;

    struct Light {
        glm::vec4 position;
        glm::vec3 color;
        float radius;
    };

    struct {
        Light lights[6];
        glm::vec4 viewPos;
        vk::Extent2D windowSize;
    } uboFragmentLights;

    struct {
        vks::Buffer vsFullScreen;
        vks::Buffer vsOffscreen;
        vks::Buffer fsLights;
    } uniformBuffers;

    struct {
        vk::Pipeline deferred;                // Deferred lighting calculation
        vk::Pipeline deferredNoMSAA;          // Deferred lighting calculation with explicit MSAA resolve
        vk::Pipeline offscreen;               // (Offscreen) scene rendering (fill G-Buffers)
        vk::Pipeline offscreenSampleShading;  // (Offscreen) scene rendering (fill G-Buffers) with sample shading rate enabled
        vk::Pipeline debug;                   // G-Buffers debug display
    } pipelines;

    struct {
        vk::PipelineLayout deferred;
        vk::PipelineLayout offscreen;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet model;
        vk::DescriptorSet floor;
    } descriptorSets;

    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Framebuffer for offscreen rendering
    using FrameBufferAttachment = vks::Image;

    struct Offscreen {
        vk::Extent2D size;
        vk::Framebuffer frameBuffer;
        FrameBufferAttachment position, normal, albedo;
        FrameBufferAttachment depth;
        vk::RenderPass renderPass;
        void destroy(const vk::Device& device) {
            position.destroy();
            normal.destroy();
            albedo.destroy();
            depth.destroy();
            device.destroy(frameBuffer);
            device.destroy(renderPass);
            device.destroy(semaphore);
            device.destroy(colorSampler);
        }
        vk::CommandBuffer commandBuffer;
        // Semaphore used to synchronize between offscreen and final scene rendering
        vk::Semaphore semaphore;
        // One sampler for the frame buffer color attachments
        vk::Sampler colorSampler;
    } offscreen;

    VulkanExample() {
        title = "Multi sampled deferred shading";
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
        camera.rotationSpeed = 0.25f;
#endif
        camera.position = { 2.15f, 0.3f, -8.75f };
        camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        paused = true;
        settings.overlay = true;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        device.destroy(pipelines.deferred);
        device.destroy(pipelines.deferredNoMSAA);
        device.destroy(pipelines.offscreen);
        device.destroy(pipelines.offscreenSampleShading);
        device.destroy(pipelines.debug);

        device.destroy(pipelineLayouts.deferred);
        device.destroy(pipelineLayouts.offscreen);

        device.destroy(descriptorSetLayout);

        // Meshes
        models.model.destroy();
        models.floor.destroy();

        // Uniform buffers
        uniformBuffers.vsOffscreen.destroy();
        uniformBuffers.vsFullScreen.destroy();
        uniformBuffers.fsLights.destroy();

        textures.model.colorMap.destroy();
        textures.model.normalMap.destroy();
        textures.floor.colorMap.destroy();
        textures.floor.normalMap.destroy();

        offscreen.destroy(device);
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        // Enable sample rate shading filtering if supported
        if (context.deviceFeatures.sampleRateShading) {
            context.enabledFeatures.sampleRateShading = VK_TRUE;
        }
        // Enable anisotropic filtering if supported
        if (context.deviceFeatures.samplerAnisotropy) {
            context.enabledFeatures.samplerAnisotropy = VK_TRUE;
        }
        // Enable texture compression
        if (context.deviceFeatures.textureCompressionBC) {
            context.enabledFeatures.textureCompressionBC = VK_TRUE;
        } else if (context.deviceFeatures.textureCompressionASTC_LDR) {
            context.enabledFeatures.textureCompressionASTC_LDR = VK_TRUE;
        } else if (context.deviceFeatures.textureCompressionETC2) {
            context.enabledFeatures.textureCompressionETC2 = VK_TRUE;
        }
    };

    // Create a frame buffer attachment
    void createAttachment(vk::Format format, const vk::ImageUsageFlags& usage, FrameBufferAttachment& attachment) {
        attachment.format = format;
        vk::ImageAspectFlags aspectMask = vk::ImageAspectFlagBits::eColor;
        vk::ImageLayout imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        }

        vk::ImageCreateInfo image;
        image.imageType = vk::ImageType::e2D;
        image.format = format;
        image.extent.width = offscreen.size.width;
        image.extent.height = offscreen.size.height;
        image.extent.depth = 1;
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = SAMPLE_COUNT;
        image.usage = usage | vk::ImageUsageFlagBits::eSampled;
        attachment = context.createImage(image);

        vk::ImageViewCreateInfo imageView;
        imageView.viewType = vk::ImageViewType::e2D;
        imageView.format = format;
        imageView.subresourceRange = { aspectMask, 0, 1, 0, 1 };
        imageView.image = attachment.image;
        attachment.view = device.createImageView(imageView);
    }

    // Prepare a new framebuffer for offscreen rendering
    // The contents of this framebuffer are then
    // blitted to our render target
    void prepareOffscreen() {
        offscreen.size = size;

        // Create a semaphore used to synchronize offscreen rendering and usage
        offscreen.semaphore = device.createSemaphore({});

        // Color attachments

        // (World space) Positions
        createAttachment(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment, offscreen.position);

        // (World space) Normals
        createAttachment(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment, offscreen.normal);

        // Albedo (color)
        createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, offscreen.albedo);

        // Depth attachment
        createAttachment(context.getSupportedDepthFormat(), vk::ImageUsageFlagBits::eDepthStencilAttachment, offscreen.depth);

        // Set up separate renderpass with references
        // to the color and depth attachments

        std::array<vk::AttachmentDescription, 4> attachmentDescs = {};

        // Init attachment properties
        for (uint32_t i = 0; i < 4; ++i) {
            attachmentDescs[i].samples = SAMPLE_COUNT;
            attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
            attachmentDescs[i].storeOp = vk::AttachmentStoreOp::eStore;
            attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attachmentDescs[i].initialLayout = vk::ImageLayout::eUndefined;
            if (i == 3) {
                attachmentDescs[i].finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
            } else {
                attachmentDescs[i].finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            }
        }

        // Formats
        attachmentDescs[0].format = offscreen.position.format;
        attachmentDescs[1].format = offscreen.normal.format;
        attachmentDescs[2].format = offscreen.albedo.format;
        attachmentDescs[3].format = offscreen.depth.format;

        std::vector<vk::AttachmentReference> colorReferences;
        colorReferences.push_back({ 0, vk::ImageLayout::eColorAttachmentOptimal });
        colorReferences.push_back({ 1, vk::ImageLayout::eColorAttachmentOptimal });
        colorReferences.push_back({ 2, vk::ImageLayout::eColorAttachmentOptimal });
        vk::AttachmentReference depthReference{ 3, vk::ImageLayout::eDepthStencilAttachmentOptimal };

        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.pColorAttachments = colorReferences.data();
        subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
        subpass.pDepthStencilAttachment = &depthReference;

        // Use subpass dependencies for attachment layput transitions
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

        offscreen.renderPass =
            device.createRenderPass({ {}, static_cast<uint32_t>(attachmentDescs.size()), attachmentDescs.data(), 1, &subpass, 2, dependencies.data() });

        std::array<vk::ImageView, 4> attachments{
            offscreen.position.view,
            offscreen.normal.view,
            offscreen.albedo.view,
            offscreen.depth.view,
        };

        offscreen.frameBuffer = device.createFramebuffer(
            { {}, offscreen.renderPass, static_cast<uint32_t>(attachments.size()), attachments.data(), offscreen.size.width, offscreen.size.height, 1 });

        // Create sampler to sample from the color attachments
        vk::SamplerCreateInfo sampler;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        sampler.addressModeV = sampler.addressModeU;
        sampler.addressModeW = sampler.addressModeU;
        sampler.mipLodBias = 0.0f;
        sampler.maxAnisotropy = 1.0f;
        sampler.minLod = 0.0f;
        sampler.maxLod = 1.0f;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        offscreen.colorSampler = device.createSampler(sampler);
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildDeferredCommandBuffer() {
        if (offscreen.commandBuffer) {
            context.trash(offscreen.commandBuffer);
        }
        offscreen.commandBuffer = context.allocateCommandBuffers(1)[0];

        // Clear values for all attachments written in the fragment sahder
        std::array<vk::ClearValue, 4> clearValues;
        clearValues[0].color = clearValues[1].color = clearValues[2].color = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 0.0f });
        clearValues[3].depthStencil = defaultClearDepth;

        offscreen.commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        vk::RenderPassBeginInfo renderPassBeginInfo{ offscreen.renderPass,
                                                     offscreen.frameBuffer,
                                                     { { 0, 0 }, offscreen.size },
                                                     static_cast<uint32_t>(clearValues.size()),
                                                     clearValues.data() };
        offscreen.commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        vk::Viewport viewport{ 0, 0, (float)offscreen.size.width, (float)offscreen.size.height, 0.0f, 1.0f };
        offscreen.commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{ { 0, 0 }, offscreen.size };
        offscreen.commandBuffer.setScissor(0, scissor);
        offscreen.commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, useSampleShading ? pipelines.offscreenSampleShading : pipelines.offscreen);

        // Background
        offscreen.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, 1, &descriptorSets.floor, 0, NULL);
        offscreen.commandBuffer.bindVertexBuffers(0, { models.floor.vertices.buffer }, { 0 });
        offscreen.commandBuffer.bindIndexBuffer(models.floor.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.commandBuffer.drawIndexed(models.floor.indexCount, 1, 0, 0, 0);

        // Object
        offscreen.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, 1, &descriptorSets.model, 0, NULL);
        offscreen.commandBuffer.bindVertexBuffers(0, { models.model.vertices.buffer }, { 0 });
        offscreen.commandBuffer.bindIndexBuffer(models.model.indices.buffer, 0, vk::IndexType::eUint32);
        offscreen.commandBuffer.drawIndexed(models.model.indexCount, 3, 0, 0, 0);

        offscreen.commandBuffer.endRenderPass();
        offscreen.commandBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        vk::Viewport viewport{ 0, 0, (float)size.width, (float)size.height, 0.0f, 1.0f };
        drawCmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{ { 0, 0 }, size };
        drawCmdBuffer.setScissor(0, scissor);

        vk::DeviceSize offsets[1] = { 0 };
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.deferred, 0, descriptorSet, nullptr);

        if (debugDisplay) {
            drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
            drawCmdBuffer.draw(3, 1, 0, 0);
            // Move viewport to display final composition in lower right corner
            viewport.x = viewport.width * 0.5f;
            viewport.y = viewport.height * 0.5f;
            viewport.width = (float)size.width * 0.5f;
            viewport.height = (float)size.height * 0.5f;
            drawCmdBuffer.setViewport(0, viewport);
        }

        camera.updateAspectRatio((float)viewport.width / (float)viewport.height);

        // Final composition as full screen quad
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, useMSAA ? pipelines.deferred : pipelines.deferredNoMSAA);
        drawCmdBuffer.draw(3, 1, 0, 0);
    }

    void loadAssets() override {
        models.model.loadFromFile(context, getAssetPath() + "models/armor/armor.dae", vertexLayout, 1.0f);

        vks::model::ModelCreateInfo modelCreateInfo;
        modelCreateInfo.scale = glm::vec3(15.0f);
        modelCreateInfo.uvscale = glm::vec2(8.0f, 8.0f);
        modelCreateInfo.center = glm::vec3(0.0f, 2.3f, 0.0f);
        models.floor.loadFromFile(context, getAssetPath() + "models/openbox.dae", vertexLayout, modelCreateInfo);

        // Textures
        std::string texFormatSuffix;
        vk::Format texFormat;
        // Get supported compressed texture format
        if (context.deviceFeatures.textureCompressionBC) {
            texFormatSuffix = "_bc3_unorm";
            texFormat = vk::Format::eBc3UnormBlock;
        } else if (context.deviceFeatures.textureCompressionASTC_LDR) {
            texFormatSuffix = "_astc_8x8_unorm";
            texFormat = vk::Format::eAstc8x8UnormBlock;
        } else if (context.deviceFeatures.textureCompressionETC2) {
            texFormatSuffix = "_etc2_unorm";
            texFormat = vk::Format::eEtc2R8G8B8A8UnormBlock;
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }

        textures.model.colorMap.loadFromFile(context, getAssetPath() + "models/armor/color" + texFormatSuffix + ".ktx", texFormat);
        textures.model.normalMap.loadFromFile(context, getAssetPath() + "models/armor/normal" + texFormatSuffix + ".ktx", texFormat);
        textures.floor.colorMap.loadFromFile(context, getAssetPath() + "textures/stonefloor02_color" + texFormatSuffix + ".ktx", texFormat);
        textures.floor.normalMap.loadFromFile(context, getAssetPath() + "textures/stonefloor02_normal" + texFormatSuffix + ".ktx", texFormat);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 8 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 9 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Deferred shading layout
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Position texture target / Scene colormap
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 2 : Normals texture target
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 3 : Albedo texture target
            vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 4 : Fragment shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 4, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.deferred = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        // Offscreen (scene) rendering pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayout })[0];
        // Model
        descriptorSets.model = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayout })[0];
        // Background
        descriptorSets.floor = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayout })[0];

        // Image descriptors for the offscreen color attachments
        vk::DescriptorImageInfo texDescriptorPosition{ offscreen.colorSampler, offscreen.position.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorNormal{ offscreen.colorSampler, offscreen.normal.view, vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo texDescriptorAlbedo{ offscreen.colorSampler, offscreen.albedo.view, vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.vsFullScreen.descriptor },
            // Binding 1 : Position texture target
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorPosition },
            // Binding 2 : Normals texture target
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorNormal },
            // Binding 3 : Albedo texture target
            { descriptorSet, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorAlbedo },
            // Binding 4 : Fragment shader uniform buffer
            { descriptorSet, 4, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.fsLights.descriptor },
            // Binding 0: Vertex shader uniform buffer
            { descriptorSets.model, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.vsOffscreen.descriptor },
            // Binding 1: Color map
            { descriptorSets.model, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.model.colorMap.descriptor },
            // Binding 2: Normal map
            { descriptorSets.model, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.model.normalMap.descriptor },
            // Binding 0: Vertex shader uniform buffer
            { descriptorSets.floor, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.vsOffscreen.descriptor },
            // Binding 1: Color map
            { descriptorSets.floor, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.floor.colorMap.descriptor },
            // Binding 2: Normal map
            { descriptorSets.floor, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.floor.normalMap.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayouts.deferred, renderPass };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

        // Final fullscreen pass pipeline
        // Deferred

        // Empty vertex input state, quads are generated by the vertex shader

        // Use specialization constants to pass number of samples to the shader (used for MSAA resolve)
        uint32_t specializationData = (uint32_t)SAMPLE_COUNT;
        vk::SpecializationMapEntry specializationEntry{ 0, 0, sizeof(uint32_t) };
        vk::SpecializationInfo specializationInfo{ 1, &specializationEntry, sizeof(uint32_t), &specializationData };
        // With MSAA
        builder.loadShader(getAssetPath() + "shaders/deferredmultisampling/deferred.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/deferredmultisampling/deferred.frag.spv", vk::ShaderStageFlagBits::eFragment);
        builder.shaderStages[1].pSpecializationInfo = &specializationInfo;
        pipelines.deferred = builder.create(context.pipelineCache);
        // No MSAA (1 sample)
        specializationData = 1;
        pipelines.deferredNoMSAA = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Debug display pipeline
        builder.loadShader(getAssetPath() + "shaders/deferredmultisampling/debug.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/deferredmultisampling/debug.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.debug = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Offscreen scene rendering pipeline
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.loadShader(getAssetPath() + "shaders/deferredmultisampling/mrt.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/deferredmultisampling/mrt.frag.spv", vk::ShaderStageFlagBits::eFragment);
        //builder.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
        //builder.rasterizationState.lineWidth = 2.0f;
        builder.multisampleState.rasterizationSamples = SAMPLE_COUNT;
        builder.multisampleState.alphaToCoverageEnable = VK_TRUE;

        // Separate render pass
        builder.renderPass = offscreen.renderPass;
        // Separate layout
        builder.layout = pipelineLayouts.offscreen;

        // Blend attachment states required for all color attachments
        // This is important, as color write mask will otherwise be 0x0 and you
        // won't see anything rendered to the attachment
        builder.colorBlendState.blendAttachmentStates.resize(3);
        pipelines.offscreen = builder.create(context.pipelineCache);

        builder.multisampleState.sampleShadingEnable = VK_TRUE;
        builder.multisampleState.minSampleShading = 0.25f;
        pipelines.offscreenSampleShading = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Fullscreen vertex shader
        uniformBuffers.vsFullScreen = context.createUniformBuffer(uboVS);
        // Deferred vertex shader
        uniformBuffers.vsOffscreen = context.createUniformBuffer(uboOffscreenVS);
        // Deferred fragment shader
        uniformBuffers.fsLights = context.createUniformBuffer(uboFragmentLights);

        // Init some values
        uboOffscreenVS.instancePos[0] = glm::vec4(0.0f);
        uboOffscreenVS.instancePos[1] = glm::vec4(-4.0f, 0.0, -4.0f, 0.0f);
        uboOffscreenVS.instancePos[2] = glm::vec4(4.0f, 0.0, -4.0f, 0.0f);

        // Update
        updateUniformBuffersScreen();
        updateUniformBufferDeferredMatrices();
        updateUniformBufferDeferredLights();
    }

    void updateUniformBuffersScreen() {
        if (debugDisplay) {
            uboVS.projection = glm::ortho(0.0f, 2.0f, 0.0f, 2.0f, -1.0f, 1.0f);
        } else {
            uboVS.projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        }
        uboVS.model = glm::mat4(1.0f);

        memcpy(uniformBuffers.vsFullScreen.mapped, &uboVS, sizeof(uboVS));
    }

    void updateUniformBufferDeferredMatrices() {
        uboOffscreenVS.projection = camera.matrices.perspective;
        uboOffscreenVS.view = camera.matrices.view;
        uboOffscreenVS.model = glm::mat4(1.0f);
        memcpy(uniformBuffers.vsOffscreen.mapped, &uboOffscreenVS, sizeof(uboOffscreenVS));
    }

    // Update fragment shader light position uniform block
    void updateUniformBufferDeferredLights() {
        // White
        uboFragmentLights.lights[0].position = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
        uboFragmentLights.lights[0].color = glm::vec3(1.5f);
        uboFragmentLights.lights[0].radius = 15.0f * 0.25f;
        // Red
        uboFragmentLights.lights[1].position = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[1].color = glm::vec3(1.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[1].radius = 15.0f;
        // Blue
        uboFragmentLights.lights[2].position = glm::vec4(2.0f, 1.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[2].color = glm::vec3(0.0f, 0.0f, 2.5f);
        uboFragmentLights.lights[2].radius = 5.0f;
        // Yellow
        uboFragmentLights.lights[3].position = glm::vec4(0.0f, 0.9f, 0.5f, 0.0f);
        uboFragmentLights.lights[3].color = glm::vec3(1.0f, 1.0f, 0.0f);
        uboFragmentLights.lights[3].radius = 2.0f;
        // Green
        uboFragmentLights.lights[4].position = glm::vec4(0.0f, 0.5f, 0.0f, 0.0f);
        uboFragmentLights.lights[4].color = glm::vec3(0.0f, 1.0f, 0.2f);
        uboFragmentLights.lights[4].radius = 5.0f;
        // Yellow
        uboFragmentLights.lights[5].position = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        uboFragmentLights.lights[5].color = glm::vec3(1.0f, 0.7f, 0.3f);
        uboFragmentLights.lights[5].radius = 25.0f;

        uboFragmentLights.lights[0].position.x = sin(glm::radians(360.0f * timer)) * 5.0f;
        uboFragmentLights.lights[0].position.z = cos(glm::radians(360.0f * timer)) * 5.0f;

        uboFragmentLights.lights[1].position.x = -4.0f + sin(glm::radians(360.0f * timer) + 45.0f) * 2.0f;
        uboFragmentLights.lights[1].position.z = 0.0f + cos(glm::radians(360.0f * timer) + 45.0f) * 2.0f;

        uboFragmentLights.lights[2].position.x = 4.0f + sin(glm::radians(360.0f * timer)) * 2.0f;
        uboFragmentLights.lights[2].position.z = 0.0f + cos(glm::radians(360.0f * timer)) * 2.0f;

        uboFragmentLights.lights[4].position.x = 0.0f + sin(glm::radians(360.0f * timer + 90.0f)) * 5.0f;
        uboFragmentLights.lights[4].position.z = 0.0f - cos(glm::radians(360.0f * timer + 45.0f)) * 5.0f;

        uboFragmentLights.lights[5].position.x = 0.0f + sin(glm::radians(-360.0f * timer + 135.0f)) * 10.0f;
        uboFragmentLights.lights[5].position.z = 0.0f - cos(glm::radians(-360.0f * timer - 45.0f)) * 10.0f;

        // Current view position
        uboFragmentLights.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

        memcpy(uniformBuffers.fsLights.mapped, &uboFragmentLights, sizeof(uboFragmentLights));
    }

    void draw() override {
        ExampleBase::prepareFrame();
        // Offscreen rendering
        context.submit(offscreen.commandBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreen.semaphore);
        // Scene rendering
        renderWaitSemaphores = { offscreen.semaphore };
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
        buildDeferredCommandBuffer();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        updateUniformBufferDeferredLights();
    }

    void viewChanged() override {
        updateUniformBufferDeferredMatrices();
        uboFragmentLights.windowSize = size;
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Display render targets", &debugDisplay)) {
                buildCommandBuffers();
                updateUniformBuffersScreen();
            }
            if (ui.checkBox("MSAA", &useMSAA)) {
                buildCommandBuffers();
            }
            if (context.deviceFeatures.sampleRateShading) {
                if (ui.checkBox("Sample rate shading", &useSampleShading)) {
                    buildDeferredCommandBuffer();
                }
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
