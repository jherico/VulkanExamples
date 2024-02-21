#pragma once

#include <rendering/context.hpp>
#include <vks/image.hpp>
#include <vks/pipelines.hpp>

namespace vkx { namespace offscreen {

struct Target {
public:
    vks::Image image;
    vk::ImageView imageView;
    vks::util::ImageTransitionState transitionState{ vks::util::ImageTransitionState::UNDEFINED };

    void create(vk::ImageCreateInfo& createInfo, vk::ImageViewType viewType);
    void destroy();
    operator bool() const;

    vk::ImageMemoryBarrier2 buildBarrier(const vks::util::ImageTransitionState& dstState);
};

struct TargetInfo {
    vk::Format format;
    vk::ImageUsageFlags usage;
    vk::ClearValue clearValue;

    TargetInfo() = default;
    TargetInfo(vk::Format format, const vk::ImageUsageFlags& usage, const vk::ClearValue& clearValue)
        : format(format)
        , usage(usage)
        , clearValue(clearValue) {}
};

struct Builder {
    static Builder defaultBuilder(const vk::Extent2D& size);

    const vk::Extent2D size;
    vk::SamplerCreateInfo samplerCreateInfo;
    std::vector<TargetInfo> colorsInfo;
    TargetInfo depthInfo{ vk::Format::eUndefined, {}, {} };
    vk::ImageViewType viewType{ vk::ImageViewType::e2D };
    vk::SampleCountFlagBits sampleCount{ vk::SampleCountFlagBits::e1 };
    uint32_t layerCount{ 1 };

    Builder(const vk::Extent2D& size);

    Builder(uint32_t size);
    operator bool() const;

    Builder& withSamplerCreateInfo(const vk::SamplerCreateInfo& samplerCreateInfo);

    Builder& appendColorFormat(vk::Format format,
                               const vk::ImageUsageFlags& flags = {},
                               const vk::ClearValue& clearColor = vk::ClearColorValue{ 0.0f, 0.0f, 0.0f, 1.0f });

    Builder& withViewType(vk::ImageViewType viewType);

    Builder& withLayerCount(uint32_t layerCount);

    Builder& withSampleCount(vk::SampleCountFlagBits sampleCount);

    Builder& withDepthFormat(vk::Format format,
                             const vk::ImageUsageFlags& usage = {},
                             const vk::ClearValue& clearDepth = vk::ClearDepthStencilValue{ 1.0f, 0 });
};

struct Renderer {
    const vks::Context& context{ vks::Context::get() };
    const vk::Device& device{ context.device };
    vk::Extent2D size;

    std::vector<Target> colorTargets;
    Target depthTarget;

    vk::Sampler sampler;

    // Dynamic rendering information
    std::vector<vk::RenderingAttachmentInfo> colorAttachmentsInfo;
    vk::RenderingAttachmentInfo depthAttachmentInfo;
    vk::RenderingInfo renderingInfo;

    virtual void prepare(const Builder& builder);
    virtual void destroy();
    void setupDynamicRendering(vks::pipelines::GraphicsPipelineBuilder& pipelineBuilder) const;
    void setLayout(vk::CommandBuffer commandBuffer,
                   const vks::util::ImageTransitionState& dstStateColor,
                   const vks::util::ImageTransitionState& dstStateDepthStencil = vks::util::ImageTransitionState::UNDEFINED);
};

}}  // namespace vkx::offscreen
