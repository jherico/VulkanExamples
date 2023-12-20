/*
* Vulkan texture loader
*
* Copyright(C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <string>
#include <fstream>
#include <vector>

#include <vulkan/vulkan.hpp>

#include <gli/gli.hpp>

#include "buffer.hpp"
#include "image.hpp"
#include "filesystem.hpp"

namespace vks { namespace texture {

/** @brief Vulkan texture base class */
class Texture : public vks::Image {
    using Parent = vks::Image;

public:
    vk::Device device;
    vk::ImageLayout imageLayout;
    uint32_t mipLevels;
    uint32_t layerCount{ 1 };
    vk::DescriptorImageInfo descriptor;

    Texture& operator=(const vks::Image& image) {
        destroy();
        static_cast<vks::Image&>(*this) = image;
        return *this;
    }

    /** @brief Update image descriptor from current sampler, view and image layout */
    void updateDescriptor() {
        descriptor.sampler = sampler;
        descriptor.imageView = view;
        descriptor.imageLayout = imageLayout;
    }

    /** @brief Release all Vulkan resources held by this texture */
    void destroy() override { Parent::destroy(); }
};

/** @brief 2D texture */
class Texture2D : public Texture {
    using Parent = Texture;

public:
    /**
        * Load a 2D texture including all mip levels
        *
        * @param filename File to load (supports .ktx and .dds)
        * @param format Vulkan format of the image data stored in the file
        * @param device Vulkan device to create the texture on
        * @param copyQueue Queue used for the texture staging copy commands (must support transfer)
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        * @param (Optional) forceLinear Force linear tiling (not advised, defaults to false)
        *
        */
    void loadFromFile(const vks::Context& context,
                      const std::string& filename,
                      vk::Format format = vk::Format::eR8G8B8A8Unorm,
                      vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
                      vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                      bool forceLinear = false) {
        this->imageLayout = imageLayout;
        descriptor.imageLayout = imageLayout;
        std::shared_ptr<gli::texture2d> tex2Dptr;
        vks::file::withBinaryFileContents(filename, [&](size_t size, const void* data) {
            tex2Dptr = std::make_shared<gli::texture2d>(gli::load((const char*)data, size));
        });
        const auto& tex2D = *tex2Dptr;
        assert(!tex2D.empty());

        device = context.device;
        extent.width = static_cast<uint32_t>(tex2D[0].extent().x);
        extent.height = static_cast<uint32_t>(tex2D[0].extent().y);
        extent.depth = 1;
        mipLevels = static_cast<uint32_t>(tex2D.levels());
        layerCount = 1;

        // Create optimal tiled target image
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.mipLevels = mipLevels;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.extent = extent;
        imageCreateInfo.usage = imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst;

        static_cast<vks::Image&>(*this) = context.stageToDeviceImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, tex2D, imageLayout);

        // Create sampler
        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.magFilter = vk::Filter::eLinear;
        samplerCreateInfo.minFilter = vk::Filter::eLinear;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        // Max level-of-detail should match mip level count
        samplerCreateInfo.maxLod = static_cast<float>(mipLevels);
        // Only enable anisotropic filtering if enabled on the devicec
        samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
        samplerCreateInfo.anisotropyEnable = context.deviceFeatures.samplerAnisotropy;
        samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        sampler = device.createSampler(samplerCreateInfo);

        // Create image view
        static const vk::ImageUsageFlags VIEW_USAGE_FLAGS =
            vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eStorage |
            vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eDepthStencilAttachment |
            vk::ImageUsageFlagBits::eInputAttachment;

        if (imageUsageFlags & VIEW_USAGE_FLAGS) {
            vk::ImageViewCreateInfo viewCreateInfo;
            viewCreateInfo.viewType = vk::ImageViewType::e2D;
            viewCreateInfo.image = image;
            viewCreateInfo.format = format;
            viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, layerCount };
            view = context.device.createImageView(viewCreateInfo);

            // Update descriptor image info member that can be used for setting up descriptor sets
            updateDescriptor();
        }
    }

    /**
        * Creates a 2D texture from a buffer
        *
        * @param buffer Buffer containing texture data to upload
        * @param bufferSize Size of the buffer in machine units
        * @param width Width of the texture to create
        * @param height Height of the texture to create
        * @param format Vulkan format of the image data stored in the file
        * @param device Vulkan device to create the texture on
        * @param copyQueue Queue used for the texture staging copy commands (must support transfer)
        * @param (Optional) filter Texture filtering for the sampler (defaults to VK_FILTER_LINEAR)
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        */
    void fromBuffer(const vks::Context& context,
                    void* buffer,
                    vk::DeviceSize bufferSize,
                    vk::Format format,
                    const vk::Extent2D& extent,
                    vk::Filter filter = vk::Filter::eLinear,
                    vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
                    vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal) {
        assert(buffer);

        device = context.device;
        this->format = format;
        this->imageLayout = imageLayout;
        this->extent.width = extent.width;
        this->extent.height = extent.height;
        this->extent.depth = 1;
        mipLevels = 1;

        // Create optimal tiled target image
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.mipLevels = mipLevels;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.extent = this->extent;
        // Ensure that the TRANSFER_DST bit is set for staging
        imageCreateInfo.usage = imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst;
        static_cast<vks::Image&>(*this) = context.createImage(imageCreateInfo);

        {
            // Create a host-visible staging buffer that contains the raw image data
            vks::Buffer stagingBuffer = context.createStagingBuffer(bufferSize, buffer);

            // Copy mip levels from staging buffer
            context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& commandBuffer) {
                context.setImageLayout(commandBuffer, this->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                                       vk::ImageLayout::eTransferDstOptimal);
                vk::BufferImageCopy bufferCopyRegion;
                bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                bufferCopyRegion.imageSubresource.layerCount = 1;
                bufferCopyRegion.imageExtent.width = extent.width;
                bufferCopyRegion.imageExtent.height = extent.height;
                bufferCopyRegion.imageExtent.depth = 1;

                commandBuffer.copyBufferToImage(stagingBuffer.buffer, image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);
                context.setImageLayout(commandBuffer, this->image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal, imageLayout);
            });
            stagingBuffer.destroy();
        }

        // Create sampler
        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.magFilter = filter;
        samplerCreateInfo.minFilter = filter;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        sampler = device.createSampler(samplerCreateInfo);

        // Create image view
        vk::ImageViewCreateInfo viewCreateInfo;
        viewCreateInfo.image = image;
        viewCreateInfo.viewType = vk::ImageViewType::e2D;
        viewCreateInfo.format = format;
        viewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        view = device.createImageView(viewCreateInfo);

        // Update descriptor image info member that can be used for setting up descriptor sets
        updateDescriptor();
    }
};

/** @brief 2D array texture */
class Texture2DArray : public Texture {
public:
    /**
        * Load a 2D texture array including all mip levels
        *
        * @param filename File to load (supports .ktx and .dds)
        * @param format Vulkan format of the image data stored in the file
        * @param device Vulkan device to create the texture on
        * @param copyQueue Queue used for the texture staging copy commands (must support transfer)
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        *
        */
    void loadFromFile(const vks::Context& context,
                      std::string filename,
                      vk::Format format,
                      vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
                      vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal) {
        this->device = device;
        this->imageLayout = imageLayout;
        descriptor.imageLayout = imageLayout;

        std::shared_ptr<gli::texture2d_array> texPtr;
        vks::file::withBinaryFileContents(filename, [&](size_t size, const void* data) {
            texPtr = std::make_shared<gli::texture2d_array>(gli::load((const char*)data, size));
        });

        const gli::texture2d_array& tex2DArray = *texPtr;

        extent.width = static_cast<uint32_t>(tex2DArray.extent().x);
        extent.height = static_cast<uint32_t>(tex2DArray.extent().y);
        extent.depth = 1;
        layerCount = static_cast<uint32_t>(tex2DArray.layers());
        mipLevels = static_cast<uint32_t>(tex2DArray.levels());

        auto stagingBuffer = context.createStagingBuffer(tex2DArray);

        // Setup buffer copy regions for each layer including all of it's miplevels
        std::vector<vk::BufferImageCopy> bufferCopyRegions;
        size_t offset = 0;
        vk::BufferImageCopy bufferCopyRegion;
        bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.depth = 1;
        for (uint32_t layer = 0; layer < layerCount; layer++) {
            for (uint32_t level = 0; level < mipLevels; level++) {
                auto image = tex2DArray[layer][level];
                auto imageExtent = image.extent();
                bufferCopyRegion.imageSubresource.mipLevel = level;
                bufferCopyRegion.imageSubresource.baseArrayLayer = layer;
                bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(imageExtent.x);
                bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(imageExtent.y);
                bufferCopyRegion.bufferOffset = offset;
                bufferCopyRegions.push_back(bufferCopyRegion);
                // Increase offset into staging buffer for next level / face
                offset += image.size();
            }
        }

        // Create optimal tiled target image
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent = extent;
        imageCreateInfo.usage = imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst;
        imageCreateInfo.arrayLayers = layerCount;
        imageCreateInfo.mipLevels = mipLevels;
        static_cast<vks::Image&>(*this) = context.createImage(imageCreateInfo);

        vk::ImageSubresourceRange subresourceRange;
        subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        subresourceRange.levelCount = mipLevels;
        subresourceRange.layerCount = layerCount;

        // Use a separate command buffer for texture loading
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
            // Image barrier for optimal image (target)
            // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
            context.setImageLayout(copyCmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, subresourceRange);
            // Copy the layers and mip levels from the staging buffer to the optimal tiled image
            copyCmd.copyBufferToImage(stagingBuffer.buffer, image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegions);
            // Change texture image layout to shader read after all faces have been copied
            context.setImageLayout(copyCmd, image, vk::ImageLayout::eTransferDstOptimal, imageLayout, subresourceRange);
        });

        // Clean up staging resources
        stagingBuffer.destroy();

        // Create sampler
        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.magFilter = vk::Filter::eLinear;
        samplerCreateInfo.minFilter = vk::Filter::eLinear;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
        samplerCreateInfo.maxLod = static_cast<float>(mipLevels);
        samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        sampler = context.device.createSampler(samplerCreateInfo);

        // Create image view
        vk::ImageViewCreateInfo viewCreateInfo;
        viewCreateInfo.viewType = vk::ImageViewType::e2DArray;
        viewCreateInfo.image = image;
        viewCreateInfo.format = format;
        viewCreateInfo.subresourceRange = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, layerCount };
        view = context.device.createImageView(viewCreateInfo);

        // Update descriptor image info member that can be used for setting up descriptor sets
        updateDescriptor();
    }
};

/** @brief Cube map texture */
class TextureCubeMap : public Texture {
public:
    /**
        * Load a cubemap texture including all mip levels from a single file
        *
        * @param filename File to load (supports .ktx and .dds)
        * @param format Vulkan format of the image data stored in the file
        * @param device Vulkan device to create the texture on
        * @param copyQueue Queue used for the texture staging copy commands (must support transfer)
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        *
        */
    void loadFromFile(const vks::Context& context,
                      const std::string& filename,
                      vk::Format format,
                      vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
                      vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal) {
        device = context.device;
        this->imageLayout = imageLayout;
        descriptor.imageLayout = imageLayout;

        std::shared_ptr<const gli::texture_cube> texPtr;
        vks::file::withBinaryFileContents(filename, [&](size_t size, const void* data) {
            texPtr = std::make_shared<const gli::texture_cube>(gli::load(static_cast<const char*>(data), size));
        });
        const auto& texCube = *texPtr;
        assert(!texCube.empty());

        extent.width = static_cast<uint32_t>(texCube.extent().x);
        extent.height = static_cast<uint32_t>(texCube.extent().y);
        extent.depth = 1;
        mipLevels = static_cast<uint32_t>(texCube.levels());
        auto stagingBuffer = context.createStagingBuffer(texCube);

        // Setup buffer copy regions for each face including all of it's miplevels
        std::vector<vk::BufferImageCopy> bufferCopyRegions;
        size_t offset = 0;
        vk::BufferImageCopy bufferImageCopy;
        bufferImageCopy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        bufferImageCopy.imageSubresource.layerCount = 1;
        bufferImageCopy.imageExtent.depth = 1;
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t level = 0; level < mipLevels; level++) {
                auto image = (texCube)[face][level];
                auto imageExtent = image.extent();
                bufferImageCopy.bufferOffset = offset;
                bufferImageCopy.imageSubresource.mipLevel = level;
                bufferImageCopy.imageSubresource.baseArrayLayer = face;
                bufferImageCopy.imageExtent.width = static_cast<uint32_t>(imageExtent.x);
                bufferImageCopy.imageExtent.height = static_cast<uint32_t>(imageExtent.y);
                bufferCopyRegions.push_back(bufferImageCopy);
                // Increase offset into staging buffer for next level / face
                offset += image.size();
            }
        }

        // Create optimal tiled target image
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.mipLevels = mipLevels;
        imageCreateInfo.extent = extent;
        // Cube faces count as array layers in Vulkan
        imageCreateInfo.arrayLayers = 6;
        // Ensure that the TRANSFER_DST bit is set for staging
        imageCreateInfo.usage = imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst;
        // This flag is required for cube map images
        imageCreateInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;
        static_cast<vks::Image&>(*this) = context.createImage(imageCreateInfo);

        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
            // Image barrier for optimal image (target)
            // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
            vk::ImageSubresourceRange subresourceRange{ vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6 };
            context.setImageLayout(copyCmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, subresourceRange);
            // Copy the cube map faces from the staging buffer to the optimal tiled image
            copyCmd.copyBufferToImage(stagingBuffer.buffer, image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegions);
            // Change texture image layout to shader read after all faces have been copied
            this->imageLayout = imageLayout;
            context.setImageLayout(copyCmd, image, vk::ImageLayout::eTransferDstOptimal, imageLayout, subresourceRange);
        });

        // Create sampler
        // Create a defaultsampler
        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.magFilter = vk::Filter::eLinear;
        samplerCreateInfo.minFilter = vk::Filter::eLinear;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        // Max level-of-detail should match mip level count
        samplerCreateInfo.maxLod = static_cast<float>(mipLevels);
        // Only enable anisotropic filtering if enabled on the devicec
        samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
        samplerCreateInfo.anisotropyEnable = context.deviceFeatures.samplerAnisotropy;
        samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        sampler = device.createSampler(samplerCreateInfo);

        // Create image view
        // Textures are not directly accessed by the shaders and
        // are abstracted by image views containing additional
        // information and sub resource ranges
        view = device.createImageView(vk::ImageViewCreateInfo{ {},
                                                               image,
                                                               vk::ImageViewType::eCube,
                                                               format,
                                                               {},
                                                               vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, mipLevels, 0, 6 } });
        stagingBuffer.destroy();

        // Update descriptor image info member that can be used for setting up descriptor sets
        updateDescriptor();
    }
};
}}  // namespace vks::texture
