#include "offscreen.hpp"
#include <sstream>

namespace vkx { namespace offscreen {

void Target::create(vk::ImageCreateInfo& createInfo, vk::ImageViewType viewType) {
    image.create(createInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    imageView = image.createView(viewType);
}

void Target::destroy() {
    auto& device = vks::Context::get().device;
    if (imageView) {
        device.destroy(imageView);
        imageView = nullptr;
    }
    image.destroy();
}

Target::operator bool() const {
    return image.operator bool();
}

vk::ImageMemoryBarrier2 Target::buildBarrier(const vks::util::ImageTransitionState& dstState) {
    auto result = buildImageBarrier(image, transitionState, dstState);
    transitionState = dstState;
    return result;
}

Builder::Builder(const vk::Extent2D& size)
    : size(size) {
    samplerCreateInfo.magFilter = vk::Filter::eLinear;
    samplerCreateInfo.minFilter = vk::Filter::eLinear;
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
    samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
    samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
}

Builder::Builder(uint32_t size)
    : Builder({ size, size }) {
}

Builder Builder::defaultBuilder(const vk::Extent2D& size) {
    static const auto& context = vks::Context::get();
    // context.deviceInfo.supportedDepthFormat
    Builder result(size);
    result.appendColorFormat(vk::Format::eB8G8R8Unorm, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc);
    return result;
}

Builder& Builder::withSamplerCreateInfo(const vk::SamplerCreateInfo& samplerCreateInfo) {
    this->samplerCreateInfo = samplerCreateInfo;
    return *this;
}

Builder& Builder::appendColorFormat(vk::Format format, const vk::ImageUsageFlags& usage, const vk::ClearValue& clearColor) {
    colorsInfo.push_back(TargetInfo{ format, usage, clearColor });
    return *this;
}

Builder& Builder::withViewType(vk::ImageViewType viewType) {
    this->viewType = viewType;
    return *this;
}

Builder& Builder::withLayerCount(uint32_t layerCount) {
    this->layerCount = layerCount;
    return *this;
}

Builder& Builder::withSampleCount(vk::SampleCountFlagBits sampleCount) {
    this->sampleCount = sampleCount;
    return *this;
}

Builder& Builder::withDepthFormat(vk::Format format, const vk::ImageUsageFlags& usage, const vk::ClearValue& clearValue) {
    depthInfo = { format, usage, clearValue };
    return *this;
}

Builder::operator bool() const {
    return colorsInfo.size() != 0 || depthInfo.format != vk::Format::eUndefined;
}

void Renderer::destroy() {
    for (auto& colorTarget : colorTargets) {
        colorTarget.destroy();
    }
    colorTargets.clear();
    depthTarget.destroy();
    depthTarget = {};
    if (sampler) {
        device.destroy(sampler);
        sampler = nullptr;
    }
    if (sampler) {
        device.destroy(sampler);
        sampler = nullptr;
    }
}

void Renderer::prepare(const Builder& builder) {
    assert(builder);

    size = builder.size;

    bool needsSampler = false;

    {
        colorTargets.reserve(builder.colorsInfo.size());
        colorAttachmentsInfo.reserve(builder.colorsInfo.size());

        // These parts of the create structure are common to depth and color
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.extent.width = size.width;
        imageCreateInfo.extent.height = size.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = builder.layerCount;
        imageCreateInfo.samples = builder.sampleCount;
        renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, size };
        renderingInfo.layerCount = builder.layerCount;

        int index = 0;
        for (const auto& colorInfo : builder.colorsInfo) {
            imageCreateInfo.format = colorInfo.format;
            imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | colorInfo.usage;
            if (imageCreateInfo.usage & vk::ImageUsageFlagBits::eSampled) {
                needsSampler = true;
            }
            auto& colorTarget = colorTargets.emplace_back();
            colorTarget.create(imageCreateInfo, builder.viewType);
            std::stringstream ss;
            ss << "Offscreen color target 0x" << vk::toHexString(index) << " (" << vk::to_string(colorInfo.format) << ") ";
            auto name = ss.str();
            vks::debug::marker::setObjectName(device, colorTarget.image.image, name);

            // Set up the rendering state
            vk::RenderingAttachmentInfo& colorAttachmentInfo = colorAttachmentsInfo.emplace_back();
            colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
            colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            // If we're not going to use the image as anything other than an output attachment then we don't need to store it later
            if (imageCreateInfo.usage == vk::ImageUsageFlagBits::eColorAttachment) {
                colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
            } else {
                colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
            }
            colorAttachmentInfo.clearValue = colorInfo.clearValue;
            colorAttachmentInfo.imageView = colorTarget.imageView;
            ++index;
        }
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentsInfo.size());
        renderingInfo.pColorAttachments = colorAttachmentsInfo.data();

        // Depth is optional
        if (builder.depthInfo.format != vk::Format::eUndefined) {
            imageCreateInfo.format = builder.depthInfo.format;
            imageCreateInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | builder.depthInfo.usage;
            if (imageCreateInfo.usage & vk::ImageUsageFlagBits::eSampled) {
                needsSampler = true;
            }
            depthTarget.create(imageCreateInfo, builder.viewType);
            std::stringstream ss;
            ss << "Offscreen depth/stencil target (" << vk::to_string(builder.depthInfo.format) << ") ";
            vks::debug::marker::setObjectName(device, depthTarget.image.image, ss.str());

            depthAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
            if (imageCreateInfo.usage == vk::ImageUsageFlagBits::eDepthStencilAttachment) {
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
            } else {
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
            }
            depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
            depthAttachmentInfo.clearValue = builder.depthInfo.clearValue;
            depthAttachmentInfo.imageView = depthTarget.imageView;
            renderingInfo.pDepthAttachment = &depthAttachmentInfo;
            if (vks::Image::isStencilFormat(builder.depthInfo.format)) {
                renderingInfo.pStencilAttachment = &depthAttachmentInfo;
            }
        }
    }

    // Create sampler
    if (needsSampler) {
        sampler = device.createSampler(builder.samplerCreateInfo);
    }
}

void Renderer::setLayout(vk::CommandBuffer commandBuffer,
                         const vks::util::ImageTransitionState& dstStateColor,
                         const vks::util::ImageTransitionState& dstStateDepthStencil) {
    using namespace vks::util;
    std::vector<vk::ImageMemoryBarrier2> imageBarriers;
    imageBarriers.reserve(colorTargets.size() + 1);
    for (auto& colorTarget : colorTargets) {
        if (dstStateColor != colorTarget.transitionState) {
            imageBarriers.push_back(colorTarget.buildBarrier(dstStateColor));
        }
    }
    if (dstStateDepthStencil.layout != vk::ImageLayout::eUndefined) {
        if (dstStateDepthStencil != depthTarget.transitionState) {
            imageBarriers.push_back(depthTarget.buildBarrier(dstStateDepthStencil));
        }
    }
    if (!imageBarriers.empty()) {
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, imageBarriers });
    }
}

void Renderer::setupDynamicRendering(vks::pipelines::GraphicsPipelineBuilder& pipelineBuilder) const {
    std::vector<vk::Format> colorFormats;
    for (const auto& target : colorTargets) {
        colorFormats.push_back(target.image.createInfo.format);
    }

    vk::Format depthFormat = depthTarget.image.createInfo.format;
    vk::Format stencilFormat = vk::Format::eUndefined;
    if (vks::Image::isStencilFormat(depthFormat)) {
        stencilFormat = depthFormat;
    }
    pipelineBuilder.dynamicRendering(colorFormats, depthFormat, stencilFormat);
}

}}  // namespace vkx::offscreen
