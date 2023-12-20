/*
* Vulkan Example - Screen space ambient occlusion example
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

#define SSAO_KERNEL_SIZE 32
#define SSAO_RADIUS 0.5f

#if defined(__ANDROID__)
#define SSAO_NOISE_DIM 8
#else
#define SSAO_NOISE_DIM 4
#endif
#define SSAO_NOISE_COUNT (SSAO_NOISE_DIM * SSAO_NOISE_DIM)

// Vertex layout for the models
static const vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_UV,
    vks::model::VERTEX_COMPONENT_COLOR,
    vks::model::VERTEX_COMPONENT_NORMAL,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::texture::Texture2D ssaoNoise;
    } textures;

    struct {
        vks::model::Model scene;
    } models;

    struct UBOSceneMatrices {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
    } uboSceneMatrices;

    struct UBOSSAOParams {
        glm::mat4 projection;
        int32_t ssao = true;
        int32_t ssaoOnly = false;
        int32_t ssaoBlur = true;
    } uboSSAOParams;

    struct {
        vk::Pipeline offscreen;
        vk::Pipeline composition;
        vk::Pipeline ssao;
        vk::Pipeline ssaoBlur;
    } pipelines;

    struct {
        vk::PipelineLayout gBuffer;
        vk::PipelineLayout ssao;
        vk::PipelineLayout ssaoBlur;
        vk::PipelineLayout composition;
    } pipelineLayouts;

    struct {
        const uint32_t count = 5;
        vk::DescriptorSet model;
        vk::DescriptorSet floor;
        vk::DescriptorSet ssao;
        vk::DescriptorSet ssaoBlur;
        vk::DescriptorSet composition;
    } descriptorSets;

    struct {
        vk::DescriptorSetLayout gBuffer;
        vk::DescriptorSetLayout ssao;
        vk::DescriptorSetLayout ssaoBlur;
        vk::DescriptorSetLayout composition;
    } descriptorSetLayouts;

    struct {
        vks::Buffer sceneMatrices;
        vks::Buffer ssaoKernel;
        vks::Buffer ssaoParams;
    } uniformBuffers;

    // Framebuffer for offscreen rendering
    using FrameBufferAttachment = vks::Image;

    struct FrameBuffer {
        vk::Extent2D size;
        vk::Device device;
        vk::Framebuffer frameBuffer;
        vk::RenderPass renderPass;

        virtual void destroy() {
            device.destroy(frameBuffer);
            device.destroy(renderPass);
        }
    };

    struct {
        struct Offscreen : public FrameBuffer {
            FrameBufferAttachment position, normal, albedo, depth;
            void destroy() override {
                position.destroy();
                normal.destroy();
                albedo.destroy();
                depth.destroy();
                FrameBuffer::destroy();
            }
        } offscreen;
        struct SSAO : public FrameBuffer {
            FrameBufferAttachment color;
            void destroy() override {
                color.destroy();
                FrameBuffer::destroy();
            }
        } ssao, ssaoBlur;
    } frameBuffers;

    // One sampler for the frame buffer color attachments
    vk::Sampler colorSampler;

    vk::CommandBuffer offScreenCmdBuffer;

    // Semaphore used to synchronize between offscreen and final scene rendering
    vk::Semaphore offscreenSemaphore;

    VulkanExample() {
        title = "Screen space ambient occlusion";
        settings.overlay = true;
        camera.type = Camera::CameraType::firstperson;
        camera.movementSpeed = 5.0f;
#ifndef __ANDROID__
        camera.rotationSpeed = 0.25f;
#endif
        camera.position = { 7.5f, -6.75f, 0.0f };
        camera.setRotation(glm::vec3(5.0f, 90.0f, 0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 64.0f);
    }

    ~VulkanExample() {
        device.destroy(colorSampler);

        // Framebuffers & Attachments
        frameBuffers.offscreen.destroy();
        frameBuffers.ssao.destroy();
        frameBuffers.ssaoBlur.destroy();

        device.destroy(pipelines.offscreen);
        device.destroy(pipelines.composition);
        device.destroy(pipelines.ssao);
        device.destroy(pipelines.ssaoBlur);

        device.destroy(pipelineLayouts.gBuffer);
        device.destroy(pipelineLayouts.ssao);
        device.destroy(pipelineLayouts.ssaoBlur);
        device.destroy(pipelineLayouts.composition);

        device.destroy(descriptorSetLayouts.gBuffer);
        device.destroy(descriptorSetLayouts.ssao);
        device.destroy(descriptorSetLayouts.ssaoBlur);
        device.destroy(descriptorSetLayouts.composition);

        // Meshes
        models.scene.destroy();

        // Uniform buffers
        uniformBuffers.sceneMatrices.destroy();
        uniformBuffers.ssaoKernel.destroy();
        uniformBuffers.ssaoParams.destroy();

        // Misc
        device.freeCommandBuffers(cmdPool, offScreenCmdBuffer);
        device.destroy(offscreenSemaphore);

        textures.ssaoNoise.destroy();
    }

    // Create a frame buffer attachment
    void createAttachment(vk::Format format, const vk::ImageUsageFlags& usage, FrameBufferAttachment& attachment, const vk::Extent2D& size) {
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
        image.arrayLayers = 1;
        image.usage = usage | vk::ImageUsageFlagBits::eSampled;

        attachment = context.createImage(image);

        vk::ImageViewCreateInfo imageView;
        imageView.viewType = vk::ImageViewType::e2D;
        imageView.format = format;
        imageView.subresourceRange = { aspectMask, 0, 1, 0, 1 };
        imageView.image = attachment.image;
        attachment.view = device.createImageView(imageView);
    }

    void prepareOffscreenFramebuffers() {
#if defined(__ANDROID__)
        const vk::Extent2D ssaoSize{ size.width / 2, size.height / 2 };
#else
        const vk::Extent2D& ssaoSize = size;
#endif

        frameBuffers.offscreen.device = device;
        frameBuffers.offscreen.size = size;
        frameBuffers.ssao.device = device;
        frameBuffers.ssao.size = ssaoSize;
        frameBuffers.ssaoBlur.device = device;
        frameBuffers.ssaoBlur.size = size;

        // G-Buffer
        createAttachment(vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eColorAttachment, frameBuffers.offscreen.position, size);  // Position + Depth
        createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, frameBuffers.offscreen.normal, size);         // Normals
        createAttachment(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment, frameBuffers.offscreen.albedo, size);         // Albedo (color)
        createAttachment(context.getSupportedDepthFormat(), vk::ImageUsageFlagBits::eDepthStencilAttachment, frameBuffers.offscreen.depth, size);  // Depth

        // SSAO
        createAttachment(vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eColorAttachment, frameBuffers.ssao.color, ssaoSize);  // Color

        // SSAO blur
        createAttachment(vk::Format::eR8Unorm, vk::ImageUsageFlagBits::eColorAttachment, frameBuffers.ssaoBlur.color, size);  // Color

        // All the renderpasses share the same subpass dependencies
        std::array<vk::SubpassDependency, 2> dependencies{
            vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                                   vk::DependencyFlagBits::eByRegion },
            vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead,
                                   vk::DependencyFlagBits::eByRegion },
        };

        // Render passes

        // G-Buffer creation
        {
            auto& fb = frameBuffers.offscreen;
            std::array<vk::AttachmentDescription, 4> attachmentDescs;

            // Init attachment properties
            for (uint32_t i = 0; i < static_cast<uint32_t>(attachmentDescs.size()); i++) {
                attachmentDescs[i].loadOp = vk::AttachmentLoadOp::eClear;
                attachmentDescs[i].storeOp = vk::AttachmentStoreOp::eStore;
                attachmentDescs[i].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
                attachmentDescs[i].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
                attachmentDescs[i].finalLayout = (i == 3) ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
            }

            // Formats
            attachmentDescs[0].format = fb.position.format;
            attachmentDescs[1].format = fb.normal.format;
            attachmentDescs[2].format = fb.albedo.format;
            attachmentDescs[3].format = fb.depth.format;

            std::vector<vk::AttachmentReference> colorReferences{
                { 0, vk::ImageLayout::eColorAttachmentOptimal },
                { 1, vk::ImageLayout::eColorAttachmentOptimal },
                { 2, vk::ImageLayout::eColorAttachmentOptimal },
            };

            vk::AttachmentReference depthReference{ 3, vk::ImageLayout::eDepthStencilAttachmentOptimal };

            vk::SubpassDescription subpass{ {}, vk::PipelineBindPoint::eGraphics };
            subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
            subpass.pColorAttachments = colorReferences.data();
            subpass.pDepthStencilAttachment = &depthReference;

            fb.renderPass =
                device.createRenderPass({ {}, static_cast<uint32_t>(attachmentDescs.size()), attachmentDescs.data(), 1, &subpass, 2, dependencies.data() });

            std::array<vk::ImageView, 4> attachments{
                fb.position.view,
                fb.normal.view,
                fb.albedo.view,
                fb.depth.view,
            };

            fb.frameBuffer = device.createFramebuffer(
                { {}, fb.renderPass, static_cast<uint32_t>(attachments.size()), attachments.data(), fb.size.width, fb.size.height, 1 });
        }

        // SSAO
        {
            auto& fb = frameBuffers.ssao;
            vk::AttachmentDescription attachmentDescription{};
            attachmentDescription.format = fb.color.format;
            attachmentDescription.loadOp = vk::AttachmentLoadOp::eClear;
            attachmentDescription.storeOp = vk::AttachmentStoreOp::eStore;
            attachmentDescription.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attachmentDescription.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attachmentDescription.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            vk::AttachmentReference colorReference{ 0, vk::ImageLayout::eColorAttachmentOptimal };

            vk::SubpassDescription subpass{ {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorReference };

            fb.renderPass = device.createRenderPass({ {}, 1, &attachmentDescription, 1, &subpass, 2, dependencies.data() });
            fb.frameBuffer = device.createFramebuffer({ {}, fb.renderPass, 1, &fb.color.view, fb.size.width, fb.size.height, 1 });
        }

        // SSAO Blur
        {
            auto& fb = frameBuffers.ssaoBlur;
            vk::AttachmentDescription attachmentDescription{};
            attachmentDescription.format = fb.color.format;
            attachmentDescription.loadOp = vk::AttachmentLoadOp::eClear;
            attachmentDescription.storeOp = vk::AttachmentStoreOp::eStore;
            attachmentDescription.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attachmentDescription.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attachmentDescription.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eColorAttachmentOptimal };

            vk::SubpassDescription subpass{ {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorReference };

            fb.renderPass = device.createRenderPass({ {}, 1, &attachmentDescription, 1, &subpass, 2, dependencies.data() });
            fb.frameBuffer = device.createFramebuffer({ {}, fb.renderPass, 1, &fb.color.view, fb.size.width, fb.size.height, 1 });
        }

        vk::SamplerCreateInfo sampler;
        sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        sampler.maxAnisotropy = 1.0f;
        sampler.maxLod = 1.0f;
        sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        colorSampler = device.createSampler(sampler);
    }

    // Build command buffer for rendering the scene to the offscreen frame buffer attachments
    void buildDeferredCommandBuffer() {
        vk::DeviceSize offsets = { 0 };

        if (!offScreenCmdBuffer) {
            offScreenCmdBuffer = context.allocateCommandBuffers(1, vk::CommandBufferLevel::ePrimary)[0];
        }

        // Create a semaphore used to synchronize offscreen rendering and usage
        offscreenSemaphore = device.createSemaphore({});

        offScreenCmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        // First pass: Fill G-Buffer components (positions+depth, normals, albedo) using MRT
        // -------------------------------------------------------------------------------------------------------
        std::vector<vk::ClearValue> clearValues(4);
        clearValues[0].color = clearValues[1].color = clearValues[2].color = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 1.0f });
        clearValues[3].depthStencil = defaultClearDepth;
        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = frameBuffers.offscreen.renderPass;
        renderPassBeginInfo.framebuffer = frameBuffers.offscreen.frameBuffer;
        renderPassBeginInfo.renderArea.extent = frameBuffers.offscreen.size;
        renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassBeginInfo.pClearValues = clearValues.data();
        vk::Viewport viewport{ 0, 0, (float)frameBuffers.offscreen.size.width, (float)frameBuffers.offscreen.size.height, 0, 1 };
        vk::Rect2D scissor{ vk::Offset2D{}, frameBuffers.offscreen.size };

        offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offScreenCmdBuffer.setViewport(0, viewport);
        offScreenCmdBuffer.setScissor(0, scissor);
        offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
        offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.gBuffer, 0, descriptorSets.floor, {});
        offScreenCmdBuffer.bindVertexBuffers(0, models.scene.vertices.buffer, { 0 });
        offScreenCmdBuffer.bindIndexBuffer(models.scene.indices.buffer, 0, vk::IndexType::eUint32);
        offScreenCmdBuffer.drawIndexed(models.scene.indexCount, 1, 0, 0, 0);
        offScreenCmdBuffer.endRenderPass();

        // Second pass: SSAO generation
        // -------------------------------------------------------------------------------------------------------
        clearValues[1].depthStencil = defaultClearDepth;
        renderPassBeginInfo.framebuffer = frameBuffers.ssao.frameBuffer;
        renderPassBeginInfo.renderPass = frameBuffers.ssao.renderPass;
        renderPassBeginInfo.renderArea.extent = frameBuffers.ssao.size;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues.data();
        viewport.width = (float)frameBuffers.ssao.size.width;
        viewport.height = (float)frameBuffers.ssao.size.height;
        scissor.extent = frameBuffers.ssao.size;

        offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offScreenCmdBuffer.setViewport(0, viewport);
        offScreenCmdBuffer.setScissor(0, scissor);
        offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.ssao, 0, descriptorSets.ssao, {});
        offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ssao);
        offScreenCmdBuffer.draw(3, 1, 0, 0);
        offScreenCmdBuffer.endRenderPass();

        // Third pass: SSAO blur
        // -------------------------------------------------------------------------------------------------------
        renderPassBeginInfo.framebuffer = frameBuffers.ssaoBlur.frameBuffer;
        renderPassBeginInfo.renderPass = frameBuffers.ssaoBlur.renderPass;
        renderPassBeginInfo.renderArea.extent = frameBuffers.ssaoBlur.size;
        viewport.width = (float)frameBuffers.ssaoBlur.size.width;
        viewport.height = (float)frameBuffers.ssaoBlur.size.height;
        scissor.extent = frameBuffers.ssaoBlur.size;

        offScreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        offScreenCmdBuffer.setViewport(0, viewport);
        offScreenCmdBuffer.setScissor(0, scissor);
        offScreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.ssaoBlur, 0, descriptorSets.ssaoBlur, {});
        offScreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.ssaoBlur);
        offScreenCmdBuffer.draw(3, 1, 0, 0);
        offScreenCmdBuffer.endRenderPass();

        offScreenCmdBuffer.end();
    }

    void loadAssets() override {
        vks::model::ModelCreateInfo modelCreateInfo;
        modelCreateInfo.scale = glm::vec3(0.5f);
        modelCreateInfo.uvscale = glm::vec2(1.0f);
        modelCreateInfo.center = glm::vec3(0.0f, 0.0f, 0.0f);
        models.scene.loadFromFile(context, getAssetPath() + "models/sibenik/sibenik.dae", vertexLayout, modelCreateInfo);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCommandBuffer) override {
        vk::Viewport viewport;
        viewport.width = (float)size.width;
        viewport.height = (float)size.height;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;
        drawCommandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor;
        scissor.extent = size;
        drawCommandBuffer.setScissor(0, scissor);
        // Final composition pass
        drawCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.composition, 0, descriptorSets.composition, {});
        drawCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.composition);
        drawCommandBuffer.draw(3, 1, 0, 0);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 10 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 12 },
        };
        descriptorPool =
            device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, descriptorSets.count, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupLayoutsAndDescriptors() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
        vk::DescriptorSetLayoutCreateInfo setLayoutCreateInfo;
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
        vk::DescriptorSetAllocateInfo descriptorAllocInfo{ descriptorPool, 1 };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

        // G-Buffer creation (offscreen scene rendering)
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayouts.gBuffer = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.gBuffer = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.gBuffer });
        descriptorSets.floor = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayouts.gBuffer })[0];
        writeDescriptorSets = { { descriptorSets.floor, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.sceneMatrices.descriptor } };
        device.updateDescriptorSets(writeDescriptorSets, {});
        // SSAO Generation
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Position+Depth
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Normals
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS SSAO Noise
            { 3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },         // FS SSAO Kernel UBO
            { 4, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },         // FS Params UBO
        };

        descriptorSetLayouts.ssao = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.ssao = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.ssao });
        descriptorSets.ssao = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayouts.ssao })[0];

        std::vector<vk::DescriptorImageInfo> imageDescriptors{
            { colorSampler, frameBuffers.offscreen.position.view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { colorSampler, frameBuffers.offscreen.normal.view, vk::ImageLayout::eShaderReadOnlyOptimal },
        };
        writeDescriptorSets = {
            { descriptorSets.ssao, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[0] },                     // FS Position+Depth
            { descriptorSets.ssao, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[1] },                     // FS Normals
            { descriptorSets.ssao, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &textures.ssaoNoise.descriptor },           // FS SSAO Noise
            { descriptorSets.ssao, 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.ssaoKernel.descriptor },  // FS SSAO Kernel UBO
            { descriptorSets.ssao, 4, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.ssaoParams.descriptor },  // FS SSAO Params UBO
        };
        device.updateDescriptorSets(writeDescriptorSets, {});

        // SSAO Blur
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Sampler SSAO
        };
        descriptorSetLayouts.ssaoBlur = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.ssaoBlur = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.ssaoBlur });
        descriptorSets.ssaoBlur = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.ssaoBlur })[0];

        imageDescriptors = {
            { colorSampler, frameBuffers.ssao.color.view, vk::ImageLayout::eShaderReadOnlyOptimal },
        };
        writeDescriptorSets = {
            { descriptorSets.ssaoBlur, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[0] },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});

        // Composition
        setLayoutBindings = {
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Position+Depth
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Normals
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS Albedo
            { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS SSAO
            { 4, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },  // FS SSAO blurred
            { 5, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },         // FS Lights UBO
        };

        descriptorSetLayouts.composition = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.composition = device.createPipelineLayout({ {}, 1, &descriptorSetLayouts.composition });
        descriptorSets.composition = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.composition })[0];

        imageDescriptors = {
            { colorSampler, frameBuffers.offscreen.position.view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { colorSampler, frameBuffers.offscreen.normal.view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { colorSampler, frameBuffers.offscreen.albedo.view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { colorSampler, frameBuffers.ssao.color.view, vk::ImageLayout::eShaderReadOnlyOptimal },
            { colorSampler, frameBuffers.ssaoBlur.color.view, vk::ImageLayout::eShaderReadOnlyOptimal },
        };

        writeDescriptorSets = {
            { descriptorSets.composition, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[0] },  // FS Sampler Position+Depth
            { descriptorSets.composition, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[1] },  // FS Sampler Normals
            { descriptorSets.composition, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[2] },  // FS Sampler Albedo
            { descriptorSets.composition, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[3] },  // FS Sampler SSAO
            { descriptorSets.composition, 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageDescriptors[4] },  // FS Sampler SSAO blurred
            { descriptorSets.composition, 5, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.ssaoParams.descriptor },  // FS SSAO Params UBO
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayouts.composition, renderPass };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        // Final composition pass pipeline
        {
            builder.loadShader(getAssetPath() + "shaders/ssao/fullscreen.vert.spv", vk::ShaderStageFlagBits::eVertex);
            builder.loadShader(getAssetPath() + "shaders/ssao/composition.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipelines.composition = builder.create(context.pipelineCache);
        }

        // SSAO Pass
        {
            builder.renderPass = frameBuffers.ssao.renderPass;
            builder.layout = pipelineLayouts.ssao;
            // Destroy the fragment shader, but not the vertex shader
            device.destroy(builder.shaderStages[1].module);
            builder.shaderStages.resize(1);
            builder.loadShader(getAssetPath() + "shaders/ssao/ssao.frag.spv", vk::ShaderStageFlagBits::eFragment);

            struct SpecializationData {
                uint32_t kernelSize = SSAO_KERNEL_SIZE;
                float radius = SSAO_RADIUS;
            } specializationData;

            // Set constant parameters via specialization constants
            std::array<vk::SpecializationMapEntry, 2> specializationMapEntries{
                vk::SpecializationMapEntry{ 0, offsetof(SpecializationData, kernelSize), sizeof(uint32_t) },  // SSAO Kernel size
                vk::SpecializationMapEntry{ 1, offsetof(SpecializationData, radius), sizeof(float) },         // SSAO radius
            };

            vk::SpecializationInfo specializationInfo{ 2, specializationMapEntries.data(), sizeof(SpecializationData), &specializationData };
            builder.shaderStages[1].pSpecializationInfo = &specializationInfo;
            pipelines.ssao = builder.create(context.pipelineCache);
        }

        // SSAO blur pass
        {
            builder.renderPass = frameBuffers.ssaoBlur.renderPass;
            builder.layout = pipelineLayouts.ssaoBlur;
            // Destroy the fragment shader, but not the vertex shader
            device.destroy(builder.shaderStages[1].module);
            builder.shaderStages.resize(1);
            builder.loadShader(getAssetPath() + "shaders/ssao/blur.frag.spv", vk::ShaderStageFlagBits::eFragment);
            pipelines.ssaoBlur = builder.create(context.pipelineCache);
        }

        // Fill G-Buffer
        {
            builder.destroyShaderModules();
            builder.renderPass = frameBuffers.offscreen.renderPass;
            builder.layout = pipelineLayouts.gBuffer;
            builder.vertexInputState.appendVertexLayout(vertexLayout);
            builder.loadShader(getAssetPath() + "shaders/ssao/gbuffer.vert.spv", vk::ShaderStageFlagBits::eVertex);
            builder.loadShader(getAssetPath() + "shaders/ssao/gbuffer.frag.spv", vk::ShaderStageFlagBits::eFragment);
            // Blend attachment states required for all color attachments
            // This is important, as color write mask will otherwise be 0x0 and you
            // won't see anything rendered to the attachment
            builder.colorBlendState.blendAttachmentStates.resize(3);
            pipelines.offscreen = builder.create(context.pipelineCache);
        }
    }

    float lerp(float a, float b, float f) { return a + f * (b - a); }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Scene matrices
        uniformBuffers.sceneMatrices = context.createUniformBuffer(uboSceneMatrices);
        // SSAO parameters
        uniformBuffers.ssaoParams = context.createUniformBuffer(uboSSAOParams);

        // Update
        updateUniformBufferMatrices();
        updateUniformBufferSSAOParams();

        // SSAO
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
        std::random_device rndDev;
        std::default_random_engine rndGen;

        // Sample kernel
        std::array<glm::vec4, SSAO_KERNEL_SIZE> ssaoKernel;
        for (uint32_t i = 0; i < SSAO_KERNEL_SIZE; ++i) {
            glm::vec3 sample(rndDist(rndGen) * 2.0 - 1.0, rndDist(rndGen) * 2.0 - 1.0, rndDist(rndGen));
            sample = glm::normalize(sample);
            sample *= rndDist(rndGen);
            float scale = float(i) / float(SSAO_KERNEL_SIZE);
            scale = lerp(0.1f, 1.0f, scale * scale);
            ssaoKernel[i] = glm::vec4(sample * scale, 0.0f);
        }

        // Upload as UBO
        uniformBuffers.ssaoKernel = context.createUniformBuffer(ssaoKernel);

        // Random noise
        std::vector<glm::vec4> ssaoNoise(SSAO_NOISE_DIM * SSAO_NOISE_DIM);
        for (uint32_t i = 0; i < static_cast<uint32_t>(ssaoNoise.size()); i++) {
            ssaoNoise[i] = glm::vec4(rndDist(rndGen) * 2.0f - 1.0f, rndDist(rndGen) * 2.0f - 1.0f, 0.0f, 0.0f);
        }
        // Upload as texture
        textures.ssaoNoise.fromBuffer(context, ssaoNoise.data(), ssaoNoise.size() * sizeof(glm::vec4), vk::Format::eR32G32B32A32Sfloat,
                                      { SSAO_NOISE_DIM, SSAO_NOISE_DIM }, vk::Filter::eLinear);
    }

    void updateUniformBufferMatrices() {
        uboSceneMatrices.projection = camera.matrices.perspective;
        uboSceneMatrices.view = camera.matrices.view;
        uboSceneMatrices.model = glm::mat4(1.0f);
        uniformBuffers.sceneMatrices.copyTo(&uboSceneMatrices, sizeof(uboSceneMatrices));
    }

    void updateUniformBufferSSAOParams() {
        uboSSAOParams.projection = camera.matrices.perspective;
        uniformBuffers.ssaoParams.copyTo(&uboSSAOParams, sizeof(uboSSAOParams));
    }

    void draw() override {
        prepareFrame();
        context.submit(offScreenCmdBuffer, { { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe } }, offscreenSemaphore);
        renderWaitSemaphores = { offscreenSemaphore };
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        loadAssets();
        prepareOffscreenFramebuffers();
        prepareUniformBuffers();
        setupDescriptorPool();
        setupLayoutsAndDescriptors();
        preparePipelines();
        buildCommandBuffers();
        buildDeferredCommandBuffer();
        prepared = true;
    }

    void viewChanged() override {
        updateUniformBufferMatrices();
        updateUniformBufferSSAOParams();
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Enable SSAO", &uboSSAOParams.ssao)) {
                updateUniformBufferSSAOParams();
            }
            if (ui.checkBox("SSAO blur", &uboSSAOParams.ssaoBlur)) {
                updateUniformBufferSSAOParams();
            }
            if (ui.checkBox("SSAO pass only", &uboSSAOParams.ssaoOnly)) {
                updateUniformBufferSSAOParams();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
