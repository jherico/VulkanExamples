#include "texture.hpp"
#include <gli/gli.hpp>

namespace vks {

// Template specialization for texture objects
template <>
inline Buffer Loader::createStagingBuffer(const gli::texture_cube& data) const {
    return createStagingBuffer(static_cast<vk::DeviceSize>(data.size()), data.data());
}

template <>
inline Buffer Loader::createStagingBuffer(const gli::texture2d_array& data) const {
    return createStagingBuffer(static_cast<vk::DeviceSize>(data.size()), data.data());
}

template <>
inline Buffer Loader::createStagingBuffer(const gli::texture2d& data) const {
    return createStagingBuffer(static_cast<vk::DeviceSize>(data.size()), data.data());
}

template <>
inline Buffer Loader::createStagingBuffer(const gli::texture& data) const {
    return createStagingBuffer(static_cast<vk::DeviceSize>(data.size()), data.data());
}

void stageToDeviceImage(vks::Image& image, const gli::texture2d& tex2D, vk::ImageLayout layout);

}  // namespace vks

using namespace vks;
using namespace vks::texture;

static const auto& loader = vks::Loader::get();
static const auto& device = vks::Context::get().device;

void vks::stageToDeviceImage(vks::Image& image, const gli::texture2d& tex2D, vk::ImageLayout layout) {
    std::vector<vks::MipData> mips;
    for (size_t i = 0; i < image.createInfo.mipLevels; ++i) {
        const auto& mip = tex2D[i];
        const auto dims = mip.extent();
        mips.push_back({ vk::Extent3D{ (uint32_t)dims.x, (uint32_t)dims.y, 1 }, (uint32_t)mip.size() });
    }
    vks::QueueManager queueManager{ device, vks::Context::get().queuesInfo.graphics };
    loader.stageToDeviceImage(queueManager, image, (vk::DeviceSize)tex2D.size(), tex2D.data(), mips, layout);
    queueManager.destroy();
}

vk::DescriptorImageInfo Texture::makeDescriptor(vk::Sampler sampler, vk::ImageLayout layout) {
    if (layout == vk::ImageLayout::eUndefined) {
        layout = this->layout.layout;
    }
    return vk::DescriptorImageInfo{ sampler, imageView, layout };
}

void Texture::destroy() {
    if (imageView) {
        device.destroy(imageView);
        imageView = nullptr;
    }
    image.destroy();
}

Texture::Barrier Texture::buildBarrier(const Layout& newLayout) {
    auto barrier = vks::util::buildImageBarrier(image, layout, newLayout);
    layout = newLayout;
    return barrier;
}

void Texture::setLayout(vk::CommandBuffer commandBuffer, const Layout& newLayout) {
    setImageLayout(commandBuffer, image, layout, newLayout);
    layout = newLayout;
}

void Texture::build(const vks::Image::Builder& builder, vk::ImageViewType viewType) {
    image = builder.build();
    imageView = image.createView(viewType);
}

void Texture2D::loadFromFile(const std::string& filename, vk::Format format, vk::ImageUsageFlags imageUsageFlags, vk::ImageLayout imageLayout) {
    std::shared_ptr<gli::texture2d> tex2Dptr;
    vks::file::withBinaryFileContents(filename, [&](auto span) {
        auto loaded = gli::load((const char*)span.data(), span.size());
        tex2Dptr = std::make_shared<gli::texture2d>(loaded);
    });

    const auto& tex2D = *tex2Dptr;
    auto size = vk::Extent2D{ static_cast<uint32_t>(tex2D.extent().x), static_cast<uint32_t>(tex2D.extent().y) };
    uint32_t levels = static_cast<uint32_t>(tex2D.levels());

    // Create optimal tiled target image
    vks::Image::Builder builder{ size };
    builder.withMipLevels(levels);
    builder.withFormat(format);
    builder.withUsage(imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst);
    build(builder, vk::ImageViewType::e2D);

    stageToDeviceImage(image, tex2D, imageLayout);
    vks::debug::marker::setObjectName(device, image.image, filename.c_str());
    vks::debug::marker::setObjectName(device, imageView, filename.c_str());
}

void Texture2D::fromBuffer(const void* buffer, vk::DeviceSize bufferSize, vks::Image::Builder& imageBuilder) {
    imageBuilder.withUsage(imageBuilder.imageCreateInfo.usage | vk::ImageUsageFlagBits::eTransferDst);
    build(imageBuilder, vk::ImageViewType::e2D);
    {
        // Create a host-visible staging buffer that contains the raw image data
        vks::Buffer stagingBuffer = loader.createStagingBuffer(bufferSize, buffer);

        vk::BufferImageCopy bufferCopyRegion;
        bufferCopyRegion.imageSubresource = image.getAllLayers();
        bufferCopyRegion.imageExtent = image.createInfo.extent;
        // Copy mip levels from staging buffer
        loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& commandBuffer) {
            // Setup buffer copy regions for each mip level
            vks::util::setImageLayout(commandBuffer, image, vks::util::ImageTransitionState::UNDEFINED, vks::util::ImageTransitionState::TRANSFER_DST);
            commandBuffer.copyBufferToImage(stagingBuffer.buffer, image.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegion);
            vks::util::setImageLayout(commandBuffer, image, vks::util::ImageTransitionState::TRANSFER_DST, vks::util::ImageTransitionState::SAMPLED);
        });
        stagingBuffer.destroy();
    }
}

void Texture2DArray::loadFromFile(const std::string& filename, vk::Format format, const vk::ImageUsageFlags& imageUsageFlags) {
    static const auto& loader = vks::Loader::get();
    std::shared_ptr<gli::texture2d_array> texPtr;
    vks::file::withBinaryFileContents(filename,
                                      [&](auto span) { texPtr = std::make_shared<gli::texture2d_array>(gli::load((const char*)span.data(), span.size())); });

    const gli::texture2d_array& tex2DArray = *texPtr;
    {
        vks::Image::Builder builder{ static_cast<uint32_t>(tex2DArray[0].extent().x), static_cast<uint32_t>(tex2DArray[0].extent().y) };
        builder.withMipLevels(static_cast<uint32_t>(tex2DArray.levels()));
        builder.withArrayLayers(static_cast<uint32_t>(tex2DArray.layers()));
        builder.withFormat(format);
        builder.withUsage(imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst);
        build(builder, vk::ImageViewType::e2DArray);
    }
    vks::debug::marker::setObjectName(device, image.image, filename.c_str());
    vks::debug::marker::setObjectName(device, imageView, filename.c_str());

    {
        auto stagingBuffer = loader.createStagingBuffer(tex2DArray);
        const auto& createInfo = image.createInfo;
        // Setup buffer copy regions for each layer including all of it's miplevels
        std::vector<vk::BufferImageCopy> bufferCopyRegions;
        size_t offset = 0;
        vk::BufferImageCopy bufferCopyRegion;
        bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.depth = 1;
        for (uint32_t layer = 0; layer < createInfo.arrayLayers; layer++) {
            for (uint32_t level = 0; level < createInfo.mipLevels; level++) {
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

        // Use a separate command buffer for texture loading
        loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
            // Image barrier for optimal image (target)
            // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
            setLayout(copyCmd, Layout::TRANSFER_DST);
            // Copy the layers and mip levels from the staging buffer to the optimal tiled image
            copyCmd.copyBufferToImage(stagingBuffer.buffer, image.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegions);
            // Change texture image layout to shader read after all faces have been copied
            setLayout(copyCmd, Layout::SAMPLED);
        });
        // Clean up staging resources
        stagingBuffer.destroy();
    }
}

void TextureCubeMap::loadFromFile(const std::string& filename, vk::Format format, const vk::ImageUsageFlags& imageUsageFlags) {
    std::shared_ptr<const gli::texture_cube> texPtr;
    vks::file::withBinaryFileContents(filename, [&](auto span) {
        texPtr = std::make_shared<const gli::texture_cube>(gli::load(reinterpret_cast<const char*>(span.data()), span.size()));
    });
    const auto& texCube = *texPtr;
    assert(!texCube.empty());

    {
        vks::Image::Builder builder{ static_cast<uint32_t>(texCube.extent().x), static_cast<uint32_t>(texCube.extent().y) };
        // This flag is required for cube map images
        builder.withFlags(vk::ImageCreateFlagBits::eCubeCompatible);
        // Cube faces count as array layers in Vulkan
        builder.withArrayLayers(6);
        builder.withMipLevels(static_cast<uint32_t>(texCube.levels()));
        builder.withFormat(format);
        // Ensure that the TRANSFER_DST bit is set for staging
        builder.withUsage(imageUsageFlags | vk::ImageUsageFlagBits::eTransferDst);
        build(builder, vk::ImageViewType::eCube);
        vks::debug::marker::setObjectName(device, image.image, filename.c_str());
        vks::debug::marker::setObjectName(device, imageView, filename.c_str());
    }

    {
        auto stagingBuffer = loader.createStagingBuffer(texCube);
        // Setup buffer copy regions for each face including all of it's miplevels
        std::vector<vk::BufferImageCopy> bufferCopyRegions;
        size_t offset = 0;
        vk::BufferImageCopy bufferImageCopy;
        bufferImageCopy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        bufferImageCopy.imageSubresource.layerCount = 1;
        bufferImageCopy.imageExtent.depth = 1;
        for (uint32_t face = 0; face < 6; face++) {
            for (uint32_t level = 0; level < image.createInfo.mipLevels; level++) {
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

        loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& copyCmd) {
            // Image barrier for optimal image (target)
            // Set initial layout for all array layers (faces) of the optimal (target) tiled texture
            setLayout(copyCmd, Layout::TRANSFER_DST);
            // Copy the cube map faces from the staging buffer to the optimal tiled image
            copyCmd.copyBufferToImage(stagingBuffer.buffer, image.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegions);
            // Change texture image layout to shader read after all faces have been copied
            setLayout(copyCmd, Layout::SAMPLED);
        });
        stagingBuffer.destroy();
    }
}

#if 0
    //// Create sampler
    //vk::SamplerCreateInfo samplerCreateInfo;
    //samplerCreateInfo.magFilter = filter;
    //samplerCreateInfo.minFilter = filter;
    //samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    //samplerCreateInfo.maxAnisotropy = 1.0f;
    //sampler = loader.device.createSampler(samplerCreateInfo);
    //descriptor = vk::DescriptorImageInfo{ sampler, imageView, imageLayout };

    //// Create sampler
    //{
    //    vk::SamplerCreateInfo samplerCreateInfo;
    //    samplerCreateInfo.magFilter = vk::Filter::eNearest;
    //    samplerCreateInfo.minFilter = vk::Filter::eNearest;
    //    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    //    samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    //    samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    //    samplerCreateInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    //    // samplerCreateInfo.maxAnisotropy = context.deviceFeatures.samplerAnisotropy ? context.deviceProperties.limits.maxSamplerAnisotropy : 1.0f;
    //    samplerCreateInfo.maxLod = static_cast<float>(image.createInfo.mipLevels - 1);
    //    samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
    //    sampler = loader.device.createSampler(samplerCreateInfo);
    //}
    // Sampler
    //vk::SamplerCreateInfo samplerCI;
    //samplerCI.magFilter = vk::Filter::eLinear;
    //samplerCI.minFilter = vk::Filter::eLinear;
    //samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
    //samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    //samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    //samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    //// Max level-of-detail should match mip level count
    //samplerCI.maxLod = static_cast<float>(image.createInfo.mipLevels);
    //samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;

    //sampler = device.createSampler(samplerCI);
    //descriptor = { sampler, imageView, imageLayout };
#endif
