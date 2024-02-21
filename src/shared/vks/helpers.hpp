/*
* Assorted commonly used Vulkan helper functions
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>
#include <vks/forward.hpp>
#include <string>
#include <list>
#include <unordered_set>
#include <vector>

namespace vks {

using StringList = std::list<std::string>;
using StringBag = std::unordered_set<std::string>;
using CStringVector = std::vector<const char*>;

template <typename T, typename N>
bool containsNext(const T& t, const N& n) {
    const vk::BaseInStructure* tp = &reinterpret_cast<const vk::BaseInStructure&>(t);
    const vk::BaseInStructure* np = &reinterpret_cast<const vk::BaseInStructure&>(n);
    while (tp != nullptr) {
        if (tp == np) {
            return true;
        }
        tp = tp->pNext;
    }
    return false;
}

template <typename T, typename N>
void injectNext(T& t, N& n) {
    vk::BaseOutStructure* tp = &reinterpret_cast<vk::BaseOutStructure&>(t);
    vk::BaseOutStructure* np = &reinterpret_cast<vk::BaseOutStructure&>(n);
    np->pNext = tp->pNext;
    tp->pNext = np;
}

namespace util {

inline vk::Viewport viewport(float width, float height, float minDepth = 0, float maxDepth = 1) {
    vk::Viewport viewport;
    viewport.width = width;
    viewport.height = height;
    viewport.minDepth = minDepth;
    viewport.maxDepth = maxDepth;
    return viewport;
}

inline vk::Viewport viewport(const glm::uvec2& size, float minDepth = 0, float maxDepth = 1) {
    return viewport(static_cast<float>(size.x), static_cast<float>(size.y), minDepth, maxDepth);
}

inline vk::Viewport viewport(const vk::Extent2D& size, float minDepth = 0, float maxDepth = 1) {
    return viewport(static_cast<float>(size.width), static_cast<float>(size.height), minDepth, maxDepth);
}

inline vk::Rect2D rect2D(uint32_t width, uint32_t height, int32_t offsetX = 0, int32_t offsetY = 0) {
    vk::Rect2D rect2D;
    rect2D.extent.width = width;
    rect2D.extent.height = height;
    rect2D.offset.x = offsetX;
    rect2D.offset.y = offsetY;
    return rect2D;
}

inline vk::Rect2D rect2D(const glm::uvec2& size, const glm::ivec2& offset = glm::ivec2(0)) {
    return rect2D(size.x, size.y, offset.x, offset.y);
}

inline vk::Rect2D rect2D(const vk::Extent2D& size, const vk::Offset2D& offset = vk::Offset2D()) {
    return rect2D(size.width, size.height, offset.x, offset.y);
}

inline vk::AccessFlags2 accessFlagsForLayout(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::ePreinitialized:
            return vk::AccessFlagBits2::eHostWrite;
        case vk::ImageLayout::eTransferDstOptimal:
            return vk::AccessFlagBits2::eTransferWrite;
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::AccessFlagBits2::eTransferRead;
        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::AccessFlagBits2::eColorAttachmentWrite;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::AccessFlagBits2::eShaderRead;
        case vk::ImageLayout::eReadOnlyOptimal:
            return vk::AccessFlagBits2::eMemoryRead;
        case vk::ImageLayout::eAttachmentOptimal:
            return vk::AccessFlagBits2::eMemoryWrite;
        default:
            return vk::AccessFlags2();
    }
}

inline vk::PipelineStageFlags2 pipelineStageForLayout(vk::ImageLayout layout) {
    switch (layout) {
        case vk::ImageLayout::eTransferDstOptimal:
        case vk::ImageLayout::eTransferSrcOptimal:
            return vk::PipelineStageFlagBits2::eTransfer;

        case vk::ImageLayout::eColorAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eColorAttachmentOutput;

        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eEarlyFragmentTests;

        case vk::ImageLayout::eShaderReadOnlyOptimal:
            return vk::PipelineStageFlagBits2::eFragmentShader;

        case vk::ImageLayout::ePreinitialized:
            return vk::PipelineStageFlagBits2::eHost;

        case vk::ImageLayout::eUndefined:
            return vk::PipelineStageFlagBits2::eTopOfPipe;

        case vk::ImageLayout::eReadOnlyOptimal:
            return vk::PipelineStageFlagBits2::eAllGraphics;
        case vk::ImageLayout::eAttachmentOptimal:
            return vk::PipelineStageFlagBits2::eBottomOfPipe;

        default:
            return vk::PipelineStageFlagBits2::eBottomOfPipe;
    }
}

struct ImageTransitionState {
    static const ImageTransitionState UNDEFINED;
    static const ImageTransitionState TRANSFER_DST;
    static const ImageTransitionState TRANSFER_SRC;
    static const ImageTransitionState PRESENT;
    static const ImageTransitionState GENERAL;
    // output color attachment state
    static const ImageTransitionState& RENDER;
    static const ImageTransitionState COLOR_ATTACHMENT;
    static const ImageTransitionState DEPTH_ATTACHMENT;
    // input texture or uniform texel buffer
    static const ImageTransitionState SAMPLED;
    vk::ImageLayout layout{ vk::ImageLayout::eUndefined };
    vk::AccessFlags2 accessMask{ vk::AccessFlagBits2::eNone };
    vk::PipelineStageFlags2 stageMask{ vk::PipelineStageFlagBits2::eNone };
    uint32_t queueFamilyIndex{ VK_QUEUE_FAMILY_IGNORED };

    ImageTransitionState() = default;

    bool operator==(const ImageTransitionState& rhs) const {
        return (layout == rhs.layout) && (accessMask == rhs.accessMask) && (stageMask == rhs.stageMask) && (queueFamilyIndex == rhs.queueFamilyIndex);
    }

    bool operator!=(const ImageTransitionState& rhs) const { return !operator==(rhs); }

    explicit ImageTransitionState(vk::ImageLayout layout,
                                  vk::AccessFlags2 accessMask = { vk::AccessFlagBits2::eNone },
                                  vk::PipelineStageFlags2 stageMask = { vk::PipelineStageFlagBits2::eNone },
                                  uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED)
        : layout(layout)
        , accessMask(accessMask)
        , stageMask(stageMask)
        , queueFamilyIndex(queueFamilyIndex) {}
};

inline vk::ClearColorValue clearColor(const glm::vec4& v = glm::vec4(0)) {
    vk::ClearColorValue result;
    memcpy(&result.float32, &v, sizeof(result.float32));
    return result;
}

struct BarrierPair {
    vk::AccessFlags2 accessMask;
    vk::PipelineStageFlags2 stageMask;

    BarrierPair() = default;
    BarrierPair(const vk::AccessFlags2& accessFlags)
        : accessMask(accessFlags)
        , stageMask(pipelineStageForLayout(vk::ImageLayout::eUndefined)) {}
};

// Useful class to simplify layout changes
struct ImageBarrierBuilder {
    vk::ImageMemoryBarrier2 barrier;

    ImageBarrierBuilder() {
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    }

    ImageBarrierBuilder& withImage(vk::Image image) {
        barrier.image = image;
        return *this;
    }

    ImageBarrierBuilder& withSubresourceRange(const vk::ImageSubresourceRange& range) {
        barrier.subresourceRange = range;
        return *this;
    }

    ImageBarrierBuilder& withAspect(const vk::ImageAspectFlags& aspect) {
        barrier.subresourceRange.aspectMask = aspect;
        return *this;
    }

    ImageBarrierBuilder& withMipLevels(uint32_t count) {
        barrier.subresourceRange.levelCount = count;
        return *this;
    }
    ImageBarrierBuilder& withArrayLayers(uint32_t count) {
        barrier.subresourceRange.layerCount = count;
        return *this;
    }

    ImageBarrierBuilder& withBaseMipLevel(uint32_t index) {
        barrier.subresourceRange.baseMipLevel = index;
        return *this;
    }

    ImageBarrierBuilder& withBaseArrayLayer(uint32_t index) {
        barrier.subresourceRange.baseArrayLayer = index;
        return *this;
    }

    ImageBarrierBuilder& withSrc(vk::ImageLayout& layout,
                                 const vk::AccessFlags2& accessMask,
                                 const vk::PipelineStageFlags2& stageMask,
                                 uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED) {
        barrier.oldLayout = layout;
        barrier.srcAccessMask = accessMask;
        barrier.srcStageMask = stageMask;
        barrier.srcQueueFamilyIndex = queueFamilyIndex;
        return *this;
    }

    ImageBarrierBuilder& withDst(vk::ImageLayout& layout,
                                 const vk::AccessFlags2& accessMask,
                                 const vk::PipelineStageFlags2& stageMask,
                                 uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED) {
        barrier.newLayout = layout;
        barrier.dstAccessMask = accessMask;
        barrier.dstStageMask = stageMask;
        barrier.dstQueueFamilyIndex = queueFamilyIndex;
        return *this;
    }

    ImageBarrierBuilder& withSrcAccessMask(const vk::AccessFlags2& accessMask) {
        barrier.srcAccessMask = accessMask;
        return *this;
    }
    ImageBarrierBuilder& withSrcStageMask(const vk::PipelineStageFlags2& stageMask) {
        barrier.srcStageMask = stageMask;
        return *this;
    }
    ImageBarrierBuilder& withDstAccessMask(const vk::AccessFlags2& accessMask) {
        barrier.dstAccessMask = accessMask;
        return *this;
    }
    ImageBarrierBuilder& withDstStageMask(const vk::PipelineStageFlags2& stageMask) {
        barrier.dstStageMask = stageMask;
        return *this;
    }

    ImageBarrierBuilder& withOldLayout(vk::ImageLayout layout) {
        barrier.oldLayout = layout;
        if (barrier.srcAccessMask == vk::AccessFlags2{}) {
            barrier.srcAccessMask = vks::util::accessFlagsForLayout(layout);
        }
        if (barrier.srcStageMask == vk::PipelineStageFlags2{}) {
            barrier.srcStageMask = vks::util::pipelineStageForLayout(layout);
        }
        return *this;
    }
    ImageBarrierBuilder& withNewLayout(vk::ImageLayout layout) {
        barrier.newLayout = layout;
        if (barrier.dstAccessMask == vk::AccessFlags2{}) {
            barrier.dstAccessMask = vks::util::accessFlagsForLayout(layout);
        }
        if (barrier.dstStageMask == vk::PipelineStageFlags2{}) {
            barrier.dstStageMask = vks::util::pipelineStageForLayout(layout);
        }
        return *this;
    }

    vk::ImageMemoryBarrier2 build() { return barrier; }

    void buildAndSubmit(vk::CommandBuffer cmdbuffer, const vk::DependencyFlags& flags = vk::DependencyFlags{}) {
        vk::DependencyInfo dependencyInfo{ flags, nullptr, nullptr, barrier };
        cmdbuffer.pipelineBarrier2(dependencyInfo);
    }
};

template <typename T>
inline CStringVector toCStrings(const T& values) {
    CStringVector result;
    result.reserve(values.size());
    for (const auto& string : values) {
        result.push_back(string.c_str());
    }
    return result;
}

vk::ImageMemoryBarrier2 buildImageBarrier(const vks::Image& image, const ImageTransitionState& srcState, const ImageTransitionState& dstState);

vk::ImageMemoryBarrier2 buildImageBarrier(vk::Image image,
                                          const vk::ImageSubresourceRange& range,
                                          const ImageTransitionState& srcState,
                                          const ImageTransitionState& dstState);
void setImageLayout(vk::CommandBuffer cmdbuffer, const vks::Image& image, const ImageTransitionState& srcState, const ImageTransitionState& dstState);
void setImageLayout(vk::CommandBuffer cmdbuffer,
                    vk::Image image,
                    const vk::ImageSubresourceRange& range,
                    const ImageTransitionState& srcState,
                    const ImageTransitionState& dstState);

}  // namespace util
}  // namespace vks
