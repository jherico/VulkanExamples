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

#include <vks/buffer.hpp>
#include <vks/image.hpp>
#include <common/filesystem.hpp>
#include <rendering/context.hpp>
#include <rendering/loader.hpp>

namespace vks { namespace texture {

/** @brief Vulkan texture base class */
class Texture {
public:
    using Layout = vks::util::ImageTransitionState;
    using Barrier = vk::ImageMemoryBarrier2;
    using DescriptorInfo = vk::DescriptorImageInfo;

    vks::Image image;
    vk::ImageView imageView;
    Layout layout{ Layout::UNDEFINED };

    void build(const vks::Image::Builder& builder, vk::ImageViewType viewType = static_cast<vk::ImageViewType>(-1));
    virtual void destroy();
    DescriptorInfo makeDescriptor(vk::Sampler sampler, vk::ImageLayout layout = vk::ImageLayout::eReadOnlyOptimal);
    Barrier buildBarrier(const Layout& newLayout);
    void setLayout(vk::CommandBuffer commandBuffer, const Layout& newLayout);
};

/** @brief 2D texture */
class Texture2D : public Texture {
    using Parent = Texture;

public:
    //void build(const vk::ImageCreateInfo& createInfo, vk::ImageLayout imageLayout = vk::ImageLayout::eReadOnlyOptimal);

    /**
        * Load a 2D texture including all mip levels
        *
        * @param filename File to load (supports .ktx and .dds)
        * @param format Vulkan format of the image data stored in the file
        * @param device Vulkan device to create the texture on
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        * @param (Optional) forceLinear Force linear tiling (not advised, defaults to false)
        *
        */
    void loadFromFile(const std::string& filename,
                      vk::Format format = vk::Format::eR8G8B8A8Unorm,
                      vk::ImageUsageFlags imageUsageFlags = vk::ImageUsageFlagBits::eSampled,
                      vk::ImageLayout imageLayout = vk::ImageLayout::eReadOnlyOptimal);

    /**
        * Creates a 2D texture from a buffer
        *
        * @param buffer Buffer containing texture data to upload
        * @param bufferSize Size of the buffer in machine units
        * @param width Width of the texture to create
        * @param height Height of the texture to create
        * @param format Vulkan format of the image data stored in the file
        * @param device Vulkan device to create the texture on
        * @param (Optional) filter Texture filtering for the sampler (defaults to VK_FILTER_LINEAR)
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        */
    void fromBuffer(const void* buffer, vk::DeviceSize bufferSize, vks::Image::Builder& imageBuilder);

    template <typename T>
    void fromBuffer(const std::vector<T>& buffer, vks::Image::Builder& imageBuilder) {
        fromBuffer(buffer.data(), buffer.size() * sizeof(T), imageBuilder);
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
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        *
        */
    void loadFromFile(const std::string& filename, vk::Format format, const vk::ImageUsageFlags& imageUsageFlags = vk::ImageUsageFlagBits::eSampled);
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
        * @param (Optional) imageUsageFlags Usage flags for the texture's image (defaults to VK_IMAGE_USAGE_SAMPLED_BIT)
        * @param (Optional) imageLayout Usage layout for the texture (defaults VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        *
        */
    void loadFromFile(const std::string& filename, vk::Format format, const vk::ImageUsageFlags& imageUsageFlags = vk::ImageUsageFlagBits::eSampled);
};

///** @brief Cube map texture */
//class Texture3D : public Texture {};

}}  // namespace vks::texture
