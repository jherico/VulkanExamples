#pragma once

#include "allocation.hpp"

namespace vks {
// Encaspulates an image, the memory for that image, a view of the image,
// as well as a sampler and the image format.
//
// The sampler is not populated by the allocation code, but is provided
// for convenience and easy cleanup if it is populated.
struct Image : public Allocation {
private:
    using Parent = Allocation;

public:
    static bool isDepthFormat(vk::Format format);
    static bool isStencilFormat(vk::Format format);
    static bool isColorFormat(vk::Format format);
    static vk::ImageAspectFlags getAspectFlags(vk::Format format);

    vk::Image image;
    vk::ImageCreateInfo createInfo;

    struct Builder : public Allocation::Builder<vks::Image::Builder> {
        vk::ImageCreateInfo imageCreateInfo;

        Builder() = delete;
        Builder(const Builder& other) = delete;

        Builder(uint32_t extent)
            : Builder(vk::Extent3D{ extent, extent, 1 }) {}
        Builder(uint32_t width, uint32_t height)
            : Builder(vk::Extent3D{ width, height, 1 }) {}

        Builder(const vk::Extent2D& extent)
            : Builder(vk::Extent3D{ extent.width, extent.height, 1 }) {}

        Builder(const vk::Extent3D& extent) {
            imageCreateInfo.extent = extent;
            imageCreateInfo.imageType = vk::ImageType::e2D;
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
        }

        Builder& withImageCreateInfo(const vk::ImageCreateInfo& createInfo) {
            imageCreateInfo = createInfo;
            return *this;
        }

        Builder& withFormat(vk::Format format) {
            imageCreateInfo.format = format;
            return *this;
        }

        //Builder& withExtent(const vk::Extent3D& extent) {
        //    imageCreateInfo.extent = extent;
        //    return *this;
        //}

        //Builder& withExtent(uint32_t width) { return withExtent(vk::Extent2D{ width, width }); }

        //Builder& withExtent(uint32_t width, uint32_t height) { return withExtent(vk::Extent2D{ width, height }); }

        //Builder& withExtent(const vk::Extent2D& extent) { return withExtent(vk::Extent3D{ extent.width, extent.height, 1 }); }

        Builder& withUsage(const vk::ImageUsageFlags& usageFlags) {
            imageCreateInfo.usage = usageFlags;
            return *this;
        }

        Builder& withFlags(const vk::ImageCreateFlags& createFlags) {
            imageCreateInfo.flags = createFlags;
            return *this;
        }

        Builder& withMipLevels(uint32_t mips) {
            imageCreateInfo.mipLevels = mips;
            return *this;
        }

        Builder& withArrayLayers(uint32_t arrayLayers) {
            imageCreateInfo.arrayLayers = arrayLayers;
            return *this;
        }

        Builder& withTiling(vk::ImageTiling tiling) {
            imageCreateInfo.tiling = tiling;
            return *this;
        }

        Builder& withInitialLayout(vk::ImageLayout initialLayout) {
            imageCreateInfo.initialLayout = initialLayout;
            return *this;
        }

        Builder& withSamples(vk::SampleCountFlagBits samples) {
            imageCreateInfo.samples = samples;
            return *this;
        }

        Builder& withSharingMode(vk::SharingMode sharingMode) {
            imageCreateInfo.sharingMode = sharingMode;
            return *this;
        }

        const vk::ImageCreateInfo& getCreateInfo() const { return imageCreateInfo; }

        static bool transientAllowed(const vk::ImageUsageFlags& usageFlags) {
            static vk::ImageUsageFlags ALLOWED_BITS =
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eInputAttachment;
            if (usageFlags == (usageFlags & ALLOWED_BITS)) {
                return true;
            }
            return false;
        }

        VULKAN_HPP_NODISCARD
        Image build() const {
            auto tempImageCreateInfo = imageCreateInfo;
            auto tempAllocationCreateInfo = allocationCreateInfo;
            if (imageCreateInfo.usage & vk::ImageUsageFlagBits::eColorAttachment || imageCreateInfo.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
                tempAllocationCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
                // If this is used ONLY as an attachment and never as a sampled image or a transfer src or dest, then we can add the transient attacment usage flag
                if (transientAllowed(imageCreateInfo.usage)) {
                    tempImageCreateInfo.usage |= vk::ImageUsageFlagBits::eTransientAttachment;
                }
            }

            Image result;
            result.create(tempImageCreateInfo, tempAllocationCreateInfo);
            return result;
        }
    };

    void create(const vk::ImageCreateInfo& createInfo, const VmaAllocationCreateInfo& allocCreateInfo);
    void create(const vk::ImageCreateInfo& createInfo, VmaAllocationCreateFlags allocationFlags = 0, VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO);
    operator bool() const { return image.operator bool(); }

    VULKAN_HPP_NODISCARD
    vk::ImageView createView(vk::ImageViewType type = static_cast<vk::ImageViewType>(-1)) const;

    vk::ImageSubresourceRange getWholeRange() const;
    vk::ImageSubresourceLayers getAllLayers() const;

    void free() override {
        vmaDestroyImage(allocator, image, allocation);
        allocation = VK_NULL_HANDLE;
        allocInfo = {};
    }

    void destroy() override {
        if (image) {
            free();
            image = nullptr;
        }
        Parent::destroy();
    }
};
}  // namespace vks
