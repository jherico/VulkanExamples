/*
 * Vulkan Example - Taking screenshots
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <shaders/screenshot/mesh.frag.inl>
#include <shaders/screenshot/mesh.vert.inl>

class VulkanExample : public vkx::ExampleBase {
public:
    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_COLOR,
    });

    struct {
        vks::model::Model object;
    } models;

    vks::Buffer uniformBuffer;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
        int32_t texIndex = 0;
    } uboVS;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;

    bool screenshotSaved = false;

    VulkanExample() {
        title = "Saving framebuffer to screenshot";
        settings.overlay = true;

        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, size, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-25.0f, 23.75f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        models.object.destroy();

        uniformBuffer.destroy();
    }

    void loadAssets() override { models.object.loadFromFile(getAssetPath() + "models/chinesedragon.dae", vertexLayout, 0.1f); }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, { 0 });
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        static const vk::DeviceSize offset = 0;
        drawCmdBuffer.bindVertexBuffers(0, models.object.vertices.buffer, offset);
        drawCmdBuffer.bindIndexBuffer(models.object.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(models.object.indexCount, 1, 0, 0, 0);
    }

    void setupDescriptorPool() {
        // Example uses one ubo
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 1 },
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() };
        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Binding 0 : Vertex shader uniform buffer
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        // Vertex bindings and attributes
        vertexLayout.appendVertexLayout(pipelineBuilder.vertexInputState);
        // Mesh rendering pipeline
        pipelineBuilder.loadShader(vkx::shaders::screenshot::mesh::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::screenshot::mesh::frag, vk::ShaderStageFlagBits::eFragment);
        pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformBuffer = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.view = camera.matrices.view;
        uboVS.model = glm::mat4(1.0f);
        uniformBuffer.copy(uboVS);
    }

    // Take a screenshot for the curretn swapchain image
    // This is done using a blit from the swapchain image to a linear image whose memory content is then saved as a ppm image
    // Getting the image date directly from a swapchain image wouldn't work as they're usually stored in an implementation dependant optimal tiling format
    // Note: This requires the swapchain images to be created with the VK_IMAGE_USAGE_TRANSFER_SRC_BIT flag (see VulkanSwapChain::create)
    void saveScreenshot(const char* filename) {
        screenshotSaved = false;

        bool supportsBlit = true;

        // Check blit support for source and destination

        // Get format properties for the swapchain color format
        // Check if the device supports blitting from optimal images (the swapchain images are in optimal format)
        vk::FormatProperties formatProps;
        formatProps = context.physicalDevice.getFormatProperties(defaultColorFormat);
        if (!(formatProps.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc)) {
            std::cerr << "Device does not support blitting from optimal tiled images, using copy instead of blit!" << std::endl;
            supportsBlit = false;
        }

        // Check if the device supports blitting to linear images
        formatProps = context.physicalDevice.getFormatProperties(vk::Format::eR8G8B8A8Unorm);
        if (!(formatProps.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst)) {
            std::cerr << "Device does not support blitting to linear tiled images, using copy instead of blit!" << std::endl;
            supportsBlit = false;
        }

        // Source for the copy is the last rendered swapchain image
        auto& srcImage = swapChain.images[currentIndex];

        // Create the linear tiled destination image to copy to and to read the memory from
        vk::ImageCreateInfo imgCreateInfo;
        imgCreateInfo.imageType = vk::ImageType::e2D;
        // Note that vkCmdBlitImage (if supported) will also do format conversions if the swapchain color format would differ
        imgCreateInfo.format = vk::Format::eR8G8B8A8Unorm;
        imgCreateInfo.extent.width = size.width;
        imgCreateInfo.extent.height = size.height;
        imgCreateInfo.extent.depth = 1;
        imgCreateInfo.arrayLayers = 1;
        imgCreateInfo.mipLevels = 1;
        imgCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
        imgCreateInfo.tiling = vk::ImageTiling::eLinear;
        imgCreateInfo.usage = vk::ImageUsageFlagBits::eTransferDst;
        // Create the image
        // Memory must be host visible to copy from
        vks::Image dstImage;
        dstImage.create(imgCreateInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        // Do the actual blit from the swapchain image to our host visible destination image
        loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& commandBuffer) {
            using namespace vks::util;
            // Transition destination image to transfer destination layout
            setImageLayout(commandBuffer, dstImage, ImageTransitionState::UNDEFINED, ImageTransitionState::TRANSFER_DST);
            // Transition swapchain image from present to transfer source layout
            srcImage.setLayout(commandBuffer, ImageTransitionState::TRANSFER_SRC);

            auto srcLayout = vk::ImageLayout::eTransferSrcOptimal;
            auto dstLayout = vk::ImageLayout::eTransferDstOptimal;
            // If source and destination support blit we'll blit as this also does automatic format conversion (e.g. from BGR to RGB)
            if (supportsBlit) {
                // Define the region to blit (we will blit the whole swapchain image)
                vk::Offset3D blitSize;
                blitSize.x = size.width;
                blitSize.y = size.height;
                blitSize.z = 1;
                vk::ImageBlit imageBlitRegion;
                imageBlitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                imageBlitRegion.srcSubresource.layerCount = 1;
                imageBlitRegion.srcOffsets[1] = blitSize;
                imageBlitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                imageBlitRegion.dstSubresource.layerCount = 1;
                imageBlitRegion.dstOffsets[1] = blitSize;
                // Issue the blit command
                commandBuffer.blitImage(srcImage.image, srcLayout, dstImage.image, dstLayout, imageBlitRegion, vk::Filter::eNearest);
            } else {
                // Otherwise use image copy (requires us to manually flip components)
                vk::ImageCopy imageCopyRegion{};
                imageCopyRegion.srcSubresource = srcImage.getAllLayers();
                imageCopyRegion.dstSubresource = dstImage.getAllLayers();
                imageCopyRegion.extent.width = size.width;
                imageCopyRegion.extent.height = size.height;
                imageCopyRegion.extent.depth = 1;
                // Issue the copy command
                commandBuffer.copyImage(srcImage.image, srcLayout, dstImage.image, dstLayout, imageCopyRegion);
            }

            // Transition destination image to general layout, which is the required layout for mapping the image memory later on
            setImageLayout(commandBuffer, dstImage, ImageTransitionState::TRANSFER_DST, ImageTransitionState::GENERAL);
            // Transition back the swap chain image after the blit is done
            srcImage.setLayout(commandBuffer, ImageTransitionState::PRESENT);
        });

        // Get layout of the image (including row pitch)
        auto subResourceLayout = device.getImageSubresourceLayout(dstImage.image, { vk::ImageAspectFlagBits::eColor });

        // Map image memory so we can start copying from it
        const char* data = (const char*)dstImage.map();
        data += subResourceLayout.offset;

        std::ofstream file(filename, std::ios::out | std::ios::binary);

        // ppm header
        file << "P6\n" << size.width << "\n" << size.height << "\n" << 255 << "\n";

        // If source is BGR (destination is always RGB) and we can't use blit (which does automatic conversion), we'll have to manually swizzle color components
        bool colorSwizzle = false;
        // Check if source is BGR
        // Note: Not complete, only contains most common and basic BGR surface formats for demonstation purposes
        if (!supportsBlit) {
            std::vector<vk::Format> formatsBGR = { vk::Format::eB8G8R8A8Srgb, vk::Format::eB8G8R8A8Unorm, vk::Format::eB8G8R8A8Snorm };
            colorSwizzle = (std::find(formatsBGR.begin(), formatsBGR.end(), swapChain.surfaceFormat.format) != formatsBGR.end());
        }

        // ppm binary pixel data
        for (uint32_t y = 0; y < size.height; y++) {
            unsigned int* row = (unsigned int*)data;
            for (uint32_t x = 0; x < size.width; x++) {
                if (colorSwizzle) {
                    file.write((char*)row + 2, 1);
                    file.write((char*)row + 1, 1);
                    file.write((char*)row, 1);
                } else {
                    file.write((char*)row, 3);
                }
                row++;
            }
            data += subResourceLayout.rowPitch;
        }
        file.close();

        std::cout << "Screenshot saved to disk" << std::endl;

        // Clean up resources
        dstImage.unmap();
        dstImage.destroy();
        screenshotSaved = true;
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

    void OnUpdateUIOverlay() override {
        if (ui.header("Functions")) {
            if (ui.button("Take screenshot")) {
                saveScreenshot("screenshot.ppm");
            }
            if (screenshotSaved) {
                ui.text("Screenshot saved as screenshot.ppm");
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
