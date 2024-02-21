#include "image.hpp"

// void vks::Image::create(const vks::Image::Builder& builder) {
//     create(builder.imageCreateInfo, builder.allocationCreateInfo);
// }

void vks::Image::create(const vk::ImageCreateInfo& createInfo, VmaAllocationCreateFlags allocationFlags, VmaMemoryUsage memoryUsage) {
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.flags = allocationFlags;
    allocCreateInfo.usage = memoryUsage;
    allocCreateInfo.pUserData = this;
    create(createInfo, allocCreateInfo);
}

void vks::Image::create(const vk::ImageCreateInfo& createInfo, const VmaAllocationCreateInfo& allocCreateInfo) {
    this->createInfo = createInfo;
#if USE_VMA
    VkImage rawImage;
    auto& ci = createInfo.operator const VkImageCreateInfo&();
    vmaCreateImage(allocator, &ci, &allocCreateInfo, &rawImage, &allocation, &allocInfo);
    image = rawImage;
#else
    image = device.createImage(createInfo);

    vk::MemoryDedicatedRequirements dedicatedAllocation;
    vk::MemoryRequirements2 memReqs{ {}, &dedicatedAllocation };
    vk::ImageMemoryRequirementsInfo2 imageMemReqs{ image, &dedicatedAllocation };
    device.getImageMemoryRequirements2(imageMemReqs, &memReqs);

    vk::MemoryAllocateInfo memAllocInfo;
    memAllocInfo.allocationSize = memReqs.memoryRequirements.size;
    memAllocInfo.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
    memory = device.allocateMemory(memAllocInfo);
    device.bindImageMemory(image, memory, 0);
#endif
}

bool vks::Image::isDepthFormat(vk::Format format) {
    switch (format) {
        case vk::Format::eD16Unorm:
        case vk::Format::eX8D24UnormPack32:
        case vk::Format::eD32Sfloat:
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return true;
        default:
            return false;
    }
}

bool vks::Image::isStencilFormat(vk::Format format) {
    switch (format) {
        case vk::Format::eS8Uint:
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return true;
        default:
            return false;
    }
}

bool vks::Image::isColorFormat(vk::Format format) {
    return !isDepthFormat(format) && !isStencilFormat(format);
}

vk::ImageAspectFlags vks::Image::getAspectFlags(vk::Format format) {
    vk::ImageAspectFlags result;
    if (isDepthFormat(format)) {
        result |= vk::ImageAspectFlagBits::eDepth;
    }
    if (isStencilFormat(format)) {
        result |= vk::ImageAspectFlagBits::eStencil;
    }
    if (!result) {
        result = vk::ImageAspectFlagBits::eColor;
    }
    return result;
}

vk::ImageSubresourceRange vks::Image::getWholeRange() const {
    vk::ImageSubresourceRange result;
    result.aspectMask = getAspectFlags(createInfo.format);
    result.baseMipLevel = 0;
    result.levelCount = createInfo.mipLevels;
    result.baseArrayLayer = 0;
    result.layerCount = createInfo.arrayLayers;
    return result;
}

vk::ImageSubresourceLayers vks::Image::getAllLayers() const {
    return vk::ImageSubresourceLayers{ getAspectFlags(createInfo.format), 0, 0, createInfo.arrayLayers };
}

VULKAN_HPP_NODISCARD
vk::ImageView vks::Image::createView(vk::ImageViewType type) const {
    // If a view type isn't explicitly set, force cast the image type.
    if (type == static_cast<vk::ImageViewType>(-1)) {
        type = static_cast<vk::ImageViewType>(createInfo.imageType);
    }
    vk::ImageViewCreateInfo imageView;
    imageView.viewType = type;
    imageView.format = createInfo.format;
    imageView.subresourceRange = getWholeRange();
    imageView.image = image;
    return Allocation::device.createImageView(imageView);
}