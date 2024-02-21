/*
 * Vulkan Example - Minimal headless rendering example
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <common/common.hpp>
#include <common/utils.hpp>
#include <rendering/context.hpp>
#include <rendering/loader.hpp>
#include <vks/buffer.hpp>
#include <vks/debug.hpp>
#include <vks/helpers.hpp>
#include <vks/image.hpp>
#include <vks/pipelines.hpp>

#include <shaders/renderheadless/triangle.frag.inl>
#include <shaders/renderheadless/triangle.vert.inl>

#define BUFFER_ELEMENTS 32

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#define LOG(...) ((void)__android_log_print(ANDROID_LOG_INFO, "vulkanExample", __VA_ARGS__))
#else
#define LOG(...) printf(__VA_ARGS__)
#endif

class VulkanExample {
public:
    vks::Context& context{ vks::Context::get() };
    vks::Loader& loader{ vks::Loader::get() };
    const vk::Device& device{ context.device };

    vk::Extent2D size;
    uint32_t& width{ size.width };
    uint32_t& height{ size.height };

    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vks::Buffer vertexBuffer, indexBuffer;

    struct Attachment {
        vks::Image image;
        vk::ImageView imageView;
        vk::Sampler sampler;

        void destroy() {
            const auto& device{ vks::Context::get().device };
            if (imageView) {
                device.destroy(imageView);
                imageView = nullptr;
            }
            if (sampler) {
                device.destroy(sampler);
                sampler = nullptr;
            }
            image.destroy();
        }
    };
    Attachment colorAttachment, depthAttachment;
    vks::QueueManager graphicsQueue;

    VulkanExample() {
        LOG("Running headless rendering example\n");

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
        LOG("loading vulkan lib");
        vks::android::loadVulkanLibrary();
#endif

        vk::ApplicationInfo appInfo;
        appInfo.pApplicationName = "Vulkan headless example";
        appInfo.pEngineName = "VulkanExample";
        appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

        /*
		    Vulkan instance creation (without surface extensions)
		*/
        vk::InstanceCreateInfo instanceCreateInfo;
        instanceCreateInfo.pApplicationInfo = &appInfo;

#if DEBUG
        context.setValidationEnabled(true);
#endif
        context.createInstance();
        context.pickDevice();
        context.createDevice();

        graphicsQueue = vks::QueueManager(context.device, context.queuesInfo.graphics);

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
        vks::android::loadVulkanFunctions(instance);
#endif

        /*
		    Prepare vertex and index buffers
		*/
        struct Vertex {
            float position[3];
            float color[3];
        };

        {
            std::vector<Vertex> vertices = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
            std::vector<uint32_t> indices = { 0, 1, 2 };

            // Vertices
            vertexBuffer = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertices);

            // Indices
            indexBuffer = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indices);
        }

        /*
		    Create framebuffer attachments
		*/
        width = 1024;
        height = 1024;
        static const vk::Format colorFormat = vk::Format::eR8G8B8A8Unorm;
        static const vk::Format depthFormat = context.deviceInfo.supportedDepthFormat;
        {
            // Color attachment
            vk::ImageCreateInfo image;
            image.imageType = vk::ImageType::e2D;
            image.format = colorFormat;
            image.extent.width = width;
            image.extent.height = height;
            image.extent.depth = 1;
            image.mipLevels = 1;
            image.arrayLayers = 1;
            image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
            colorAttachment.image.create(image);
            colorAttachment.imageView = colorAttachment.image.createView();

            // Depth stencil attachment
            image.format = depthFormat;
            image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
            depthAttachment.image.create(image);
            depthAttachment.imageView = depthAttachment.image.createView();
        }

#if 0
        /*
            Create renderpass
        */
        {
            std::array<vk::AttachmentDescription, 2> attchmentDescriptions = {};
            // Color attachment
            attchmentDescriptions[0].format = colorFormat;
            attchmentDescriptions[0].loadOp = vk::AttachmentLoadOp::eClear;
            attchmentDescriptions[0].storeOp = vk::AttachmentStoreOp::eStore;
            attchmentDescriptions[0].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attchmentDescriptions[0].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attchmentDescriptions[0].initialLayout = vk::ImageLayout::eUndefined;
            attchmentDescriptions[0].finalLayout = vk::ImageLayout::eTransferSrcOptimal;
            // Depth attachment
            attchmentDescriptions[1].format = depthFormat;
            attchmentDescriptions[1].loadOp = vk::AttachmentLoadOp::eClear;
            attchmentDescriptions[1].storeOp = vk::AttachmentStoreOp::eDontCare;
            attchmentDescriptions[1].stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
            attchmentDescriptions[1].stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
            attchmentDescriptions[1].initialLayout = vk::ImageLayout::eUndefined;
            attchmentDescriptions[1].finalLayout = vk::ImageLayout::eAttachmentOptimal;

            vk::AttachmentReference colorReference = { 0, vk::ImageLayout::eAttachmentOptimal };
            vk::AttachmentReference depthReference = { 1, vk::ImageLayout::eAttachmentOptimal };

            vk::SubpassDescription subpassDescription;
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
            vk::RenderPassCreateInfo renderPassInfo;
            renderPassInfo.attachmentCount = static_cast<uint32_t>(attchmentDescriptions.size());
            renderPassInfo.pAttachments = attchmentDescriptions.data();
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpassDescription;
            renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
            renderPassInfo.pDependencies = dependencies.data();
            renderPass = device.createRenderPass(renderPassInfo);

            vk::ImageView attachments[2];
            attachments[0] = colorAttachment.view;
            attachments[1] = depthAttachment.view;

            vk::FramebufferCreateInfo framebufferCreateInfo;
            framebufferCreateInfo.renderPass = renderPass;
            framebufferCreateInfo.attachmentCount = 2;
            framebufferCreateInfo.pAttachments = attachments;
            framebufferCreateInfo.width = width;
            framebufferCreateInfo.height = height;
            framebufferCreateInfo.layers = 1;
            framebuffer = device.createFramebuffer(framebufferCreateInfo);
        }
#endif

        /*
		    Prepare graphics pipeline
		*/
        {
            descriptorSetLayout = device.createDescriptorSetLayout({});

            vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
            // MVP via push constant block
            vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(glm::mat4) };
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
            pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);

            // Create pipeline
            vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout };
            builder.dynamicRendering(colorFormat, depthFormat);

            builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

            // Vertex bindings an attributes
            // Binding description
            builder.vertexInputState.bindingDescriptions = {
                vk::VertexInputBindingDescription{ 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
            };

            // Attribute descriptions
            builder.vertexInputState.attributeDescriptions = {
                vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) },  // Position
                vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) },     // Color
            };

            builder.loadShader(vkx::shaders::renderheadless::triangle::vert, vk::ShaderStageFlagBits::eVertex);
            builder.loadShader(vkx::shaders::renderheadless::triangle::frag, vk::ShaderStageFlagBits::eFragment);
            pipeline = builder.create(context.pipelineCache);
        }

        /*
		    Command buffer creation (for compute work submission)
		*/
        {
            vk::CommandBuffer commandBuffer = graphicsQueue.createCommandBuffer();
            commandBuffer.begin(vk::CommandBufferBeginInfo{});

            vk::RenderingAttachmentInfo colorAttachmentInfo;
            colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
            colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
            colorAttachmentInfo.clearValue = vks::util::clearColor({ 0.0f, 0.0f, 0.2f, 1.0f });
            colorAttachmentInfo.imageView = colorAttachment.imageView;

            vk::RenderingAttachmentInfo depthAttachmentInfo;
            depthAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
            depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
            depthAttachmentInfo.clearValue = vk::ClearDepthStencilValue{ 1.0, 0 };
            depthAttachmentInfo.imageView = depthAttachment.imageView;

            vk::RenderingInfo renderingInfo;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.layerCount = 1;
            renderingInfo.pColorAttachments = &colorAttachmentInfo;
            renderingInfo.pDepthAttachment = &depthAttachmentInfo;
            renderingInfo.pStencilAttachment = &depthAttachmentInfo;
            renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, size };
            using namespace vks::util;

            setImageLayout(commandBuffer, colorAttachment.image, ImageTransitionState::UNDEFINED, ImageTransitionState::RENDER);
            setImageLayout(commandBuffer, depthAttachment.image, ImageTransitionState::UNDEFINED, ImageTransitionState::RENDER);

            commandBuffer.beginRendering(renderingInfo);
            commandBuffer.setViewport(0, vk::Viewport{ 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f });
            commandBuffer.setScissor(0, vk::Rect2D{ vk::Offset2D{}, size });
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            vk::DeviceSize offset = 0;
            commandBuffer.bindVertexBuffers(0, vertexBuffer.buffer, offset);
            commandBuffer.bindIndexBuffer(indexBuffer.buffer, offset, vk::IndexType::eUint32);

            std::vector<glm::vec3> pos = {
                glm::vec3(-1.5f, 0.0f, -4.0f),
                glm::vec3(0.0f, 0.0f, -2.5f),
                glm::vec3(1.5f, 0.0f, -4.0f),
            };

            for (auto v : pos) {
                glm::mat4 mvpMatrix = glm::perspective(glm::radians(60.0f), (float)width / (float)height, 0.1f, 256.0f) * glm::translate(glm::mat4(1.0f), v);
                commandBuffer.pushConstants<glm::mat4>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, mvpMatrix);
                commandBuffer.drawIndexed(3, 1, 0, 0, 0);
            }

            commandBuffer.endRenderPass();
            commandBuffer.end();

            graphicsQueue.submitAndWait(commandBuffer);
        }

        /*
		    Copy framebuffer image to host visible image
		*/
        const char* imagedata;
        {
            // Create the linear tiled destination image to copy to and to read the memory from
            vk::ImageCreateInfo imgCreateInfo;
            imgCreateInfo.imageType = vk::ImageType::e2D;
            imgCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
            imgCreateInfo.extent.width = width;
            imgCreateInfo.extent.height = height;
            imgCreateInfo.extent.depth = 1;
            imgCreateInfo.arrayLayers = 1;
            imgCreateInfo.mipLevels = 1;
            imgCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
            imgCreateInfo.tiling = vk::ImageTiling::eLinear;
            imgCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst;

            // Create the image
            vks::Image dstImage;
            dstImage.create(imgCreateInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

            // Do the actual blit from the swapchain image to our host visible destination image
            using namespace vks::util;
            loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
                setImageLayout(copyCmd, dstImage, ImageTransitionState::UNDEFINED, ImageTransitionState::TRANSFER_DST);
                setImageLayout(copyCmd, colorAttachment.image, ImageTransitionState::RENDER, ImageTransitionState::TRANSFER_DST);

                // The source image is already in vk::ImageLayout::eTransferSrcOptimal due to the renderpass setup
                vk::ImageCopy imageCopyRegion;
                imageCopyRegion.srcSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
                imageCopyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                imageCopyRegion.dstSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
                imageCopyRegion.extent.width = width;
                imageCopyRegion.extent.height = height;
                imageCopyRegion.extent.depth = 1;

                copyCmd.copyImage(colorAttachment.image.image, vk::ImageLayout::eTransferSrcOptimal, dstImage.image, vk::ImageLayout::eTransferDstOptimal,
                                  imageCopyRegion);

                // Transition destination image to general layout, which is the required layout for mapping the image memory later on
                ImageTransitionState general{ vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eNone };

                setImageLayout(copyCmd, dstImage, ImageTransitionState::TRANSFER_DST, general);
                // The source image needs no transition because we're no longer using it for anything
            });

            // Get layout of the image (including row pitch)
            vk::SubresourceLayout subResourceLayout = device.getImageSubresourceLayout(dstImage.image, { vk::ImageAspectFlagBits::eColor });

            // Map image memory so we can start copying from it
            imagedata = (const char*)dstImage.map();
            imagedata += subResourceLayout.offset;

            /*
			Save host visible framebuffer image to disk (ppm format)
			*/

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
            const char* filename = strcat(getenv("EXTERNAL_STORAGE"), "/headless.ppm");
#else
            const char* filename = "headless.ppm";
#endif
            std::ofstream file(filename, std::ios::out | std::ios::binary);

            // ppm header
            file << "P6\n" << width << "\n" << height << "\n" << 255 << "\n";

            // ppm binary pixel data
            for (uint32_t y = 0; y < height; y++) {
                unsigned int* row = (unsigned int*)imagedata;
                for (uint32_t x = 0; x < width; x++) {
                    file.write((char*)row, 3);
                    row++;
                }
                imagedata += subResourceLayout.rowPitch;
            }
            file.close();
            LOG("Framebuffer image saved to %s\n", filename);

            // Clean up resources
            dstImage.unmap();
            dstImage.destroy();
        }
    }

    ~VulkanExample() {
        vertexBuffer.destroy();
        indexBuffer.destroy();
        colorAttachment.destroy();
        depthAttachment.destroy();
        // device.destroy(renderPass);
        // device.destroy(framebuffer);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(pipeline);

        context.destroy();
    }
};

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
void handleAppCommand(android_app* app, int32_t cmd) {
    if (cmd == APP_CMD_INIT_WINDOW) {
        VulkanExample* vulkanExample = new VulkanExample();
        delete (vulkanExample);
        ANativeActivity_finish(app->activity);
    }
}
void android_main(android_app* state) {
    app_dummy();
    androidapp = state;
    androidapp->onAppCmd = handleAppCommand;
    int ident, events;
    struct android_poll_source* source;
    while ((ident = ALooper_pollAll(-1, NULL, &events, (void**)&source)) >= 0) {
        if (source != NULL) {
            source->process(androidapp, source);
        }
        if (androidapp->destroyRequested != 0) {
            break;
        }
    }
}
#else
int main() {
    VulkanExample* vulkanExample = new VulkanExample();
    std::cout << "Finished. Press enter to terminate...";
    getchar();
    delete (vulkanExample);
    return 0;
}
#endif
