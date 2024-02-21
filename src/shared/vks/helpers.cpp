#include "helpers.hpp"
#include <vks/image.hpp>

namespace vks { namespace util {

vk::ImageMemoryBarrier2 buildImageBarrier(vk::Image image,
                                          const vk::ImageSubresourceRange& range,
                                          const ImageTransitionState& srcState,
                                          const ImageTransitionState& dstState) {
    vk::ImageMemoryBarrier2 barrier;
    barrier.oldLayout = srcState.layout;
    barrier.srcAccessMask = srcState.accessMask;
    barrier.srcStageMask = srcState.stageMask;
    barrier.srcQueueFamilyIndex = srcState.queueFamilyIndex;
    barrier.newLayout = dstState.layout;
    barrier.dstAccessMask = dstState.accessMask;
    barrier.dstStageMask = dstState.stageMask;
    barrier.dstQueueFamilyIndex = dstState.queueFamilyIndex;
    barrier.image = image;
    barrier.subresourceRange = range;
    return barrier;
}

vk::ImageMemoryBarrier2 buildImageBarrier(const vks::Image& image, const ImageTransitionState& srcState, const ImageTransitionState& dstState) {
    return buildImageBarrier(image.image, image.getWholeRange(), srcState, dstState);
}

void setImageLayout(vk::CommandBuffer cmdbuffer, const vks::Image& image, const ImageTransitionState& srcState, const ImageTransitionState& dstState) {
    setImageLayout(cmdbuffer, image.image, image.getWholeRange(), srcState, dstState);
}

void setImageLayout(vk::CommandBuffer cmdbuffer,
                    vk::Image image,
                    const vk::ImageSubresourceRange& range,
                    const ImageTransitionState& srcState,
                    const ImageTransitionState& dstState) {
    auto imageBarrier = buildImageBarrier(image, range, srcState, dstState);
    cmdbuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, imageBarrier });
}

using Access = vk::AccessFlagBits2;
using Pipeline = vk::PipelineStageFlagBits2;
using Layout = vk::ImageLayout;
using ITS = ImageTransitionState;

const ImageTransitionState ITS::UNDEFINED{ Layout::eUndefined };
const ImageTransitionState ITS::PRESENT{ Layout::ePresentSrcKHR };
const ImageTransitionState ITS::GENERAL{ Layout::eGeneral };
const ImageTransitionState ITS::SAMPLED{ Layout::eReadOnlyOptimal, Access::eShaderRead, Pipeline::eAllGraphics };
const ImageTransitionState ITS::TRANSFER_DST{ Layout::eTransferDstOptimal, Access::eTransferWrite, Pipeline::eTransfer };
const ImageTransitionState ITS::TRANSFER_SRC{ Layout::eTransferSrcOptimal, Access::eTransferRead, Pipeline::eTransfer };
const ImageTransitionState ITS::COLOR_ATTACHMENT{ Layout::eAttachmentOptimal, Access::eColorAttachmentWrite, Pipeline::eColorAttachmentOutput };
const ImageTransitionState ITS::DEPTH_ATTACHMENT{ Layout::eAttachmentOptimal, Access::eDepthStencilAttachmentWrite, Pipeline::eEarlyFragmentTests };
const ImageTransitionState& ITS::RENDER = ImageTransitionState::COLOR_ATTACHMENT;

}}  // namespace vks::util