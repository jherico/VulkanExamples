/*
* Vulkan framebuffer class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <algorithm>
#include <iterator>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "image.hpp"
#include "context.hpp"

namespace vks {
/**
 * @brief Encapsulates a single frame buffer attachment
 */
struct FramebufferAttachment : public vks::Image {
    vk::ImageSubresourceRange subresourceRange;
    vk::AttachmentDescription description;

    /**
     * @brief Returns true if the attachment has a depth component
     */
    bool hasDepth() const {
        static const std::vector<vk::Format> formats{ vk::Format::eD16Unorm,       vk::Format::eX8D24UnormPack32, vk::Format::eD32Sfloat,
                                                      vk::Format::eD16UnormS8Uint, vk::Format::eD24UnormS8Uint,   vk::Format::eD32SfloatS8Uint };
        return std::find(formats.begin(), formats.end(), format) != std::end(formats);
    }

    /**
     * @brief Returns true if the attachment has a stencil component
     */
    bool hasStencil() const {
        static const std::vector<vk::Format> formats{
            vk::Format::eS8Uint,
            vk::Format::eD16UnormS8Uint,
            vk::Format::eD24UnormS8Uint,
            vk::Format::eD32SfloatS8Uint,
        };
        return std::find(formats.begin(), formats.end(), format) != std::end(formats);
    }

    /**
     * @brief Returns true if the attachment is a depth and/or stencil attachment
     */
    bool isDepthStencil() const { return (hasDepth() || hasStencil()); }
};

/**
        * @brief Describes the attributes of an attachment to be created
        */
struct AttachmentCreateInfo {
    uint32_t layerCount{ 1 };
    vk::Format format{ vk::Format::eR8G8B8A8Unorm };
    vk::ImageUsageFlags usage{ vk::ImageUsageFlagBits::eColorAttachment };
};

/**
        * @brief Encapsulates a complete Vulkan framebuffer with an arbitrary number and combination of attachments
        */
struct Framebuffer {
public:
    const vks::Context& context;
    const vk::Device& device{ context.device };
    vk::Extent2D size;
    vk::Framebuffer framebuffer;
    vk::RenderPass renderPass;
    vk::Sampler sampler;
    std::vector<vks::FramebufferAttachment> attachments;

    /**
                * Default constructor
                *
                * @param vulkanDevice Pointer to a valid VulkanDevice
                */
    Framebuffer(const vks::Context& context)
        : context(context) {}

    /**
     * Destroy and free Vulkan resources used for the framebuffer and all of it's attachments
     */
    void destroy() {
        for (auto attachment : attachments) {
            attachment.destroy();
        }
        device.destroy(sampler);
        device.destroy(renderPass);
        device.destroy(framebuffer);
    }

    /**
     * Add a new attachment described by createinfo to the framebuffer's attachment list
     *
     * @param createinfo Structure that specifices the framebuffer to be constructed
     *
     * @return Index of the new attachment
     */
    uint32_t addAttachment(vks::AttachmentCreateInfo createinfo) {
        vks::FramebufferAttachment attachment;
        attachment.format = createinfo.format;

        vk::ImageAspectFlags aspectMask;

        // Select aspect mask and layout depending on usage

        // Color attachment
        if (createinfo.usage & vk::ImageUsageFlagBits::eColorAttachment) {
            aspectMask = vk::ImageAspectFlagBits::eColor;
        }

        // Depth (and/or stencil) attachment
        if (createinfo.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
            if (attachment.hasDepth()) {
                aspectMask = vk::ImageAspectFlagBits::eDepth;
            }
            if (attachment.hasStencil()) {
                aspectMask = aspectMask | vk::ImageAspectFlagBits::eStencil;
            }
        }

        vk::ImageCreateInfo image;
        image.imageType = vk::ImageType::e2D;
        image.format = createinfo.format;
        image.extent.width = size.width;
        image.extent.height = size.height;
        image.extent.depth = 1;
        image.mipLevels = 1;
        image.arrayLayers = createinfo.layerCount;
        image.usage = createinfo.usage;

        // Create image for this attachment
        (vks::Image&)attachment = context.createImage(image);

        attachment.subresourceRange = vk::ImageSubresourceRange{ aspectMask, 0, 1, 0, createinfo.layerCount };
        vk::ImageViewCreateInfo imageView;
        imageView.viewType = (createinfo.layerCount == 1) ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
        imageView.format = createinfo.format;
        imageView.subresourceRange = attachment.subresourceRange;
        //todo: workaround for depth+stencil attachments
        imageView.subresourceRange.aspectMask = (attachment.hasDepth()) ? vk::ImageAspectFlagBits::eDepth : aspectMask;
        imageView.image = attachment.image;
        attachment.view = device.createImageView(imageView);

        // Fill attachment description
        attachment.description.loadOp = vk::AttachmentLoadOp::eClear;
        attachment.description.storeOp =
            (createinfo.usage & vk::ImageUsageFlagBits::eSampled) ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare;
        attachment.description.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attachment.description.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attachment.description.format = createinfo.format;
        attachment.description.initialLayout = vk::ImageLayout::eUndefined;

        // Final layout
        // If not, final layout depends on attachment type
        if (attachment.hasDepth() || attachment.hasStencil()) {
            attachment.description.finalLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
        } else {
            attachment.description.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        }
        attachments.push_back(attachment);

        return static_cast<uint32_t>(attachments.size() - 1);
    }

    /**
     * Creates a default sampler for sampling from any of the framebuffer attachments
     * Applications are free to create their own samplers for different use cases
     *
     * @param magFilter Magnification filter for lookups
     * @param minFilter Minification filter for lookups
     * @param adressMode Adressing mode for the U,V and W coordinates
     *
     * @return VkResult for the sampler creation
     */
    void createSampler(vk::Filter magFilter, vk::Filter minFilter, vk::SamplerAddressMode adressMode) {
        vk::SamplerCreateInfo samplerInfo;
        samplerInfo.magFilter = magFilter;
        samplerInfo.minFilter = minFilter;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU = adressMode;
        samplerInfo.addressModeV = adressMode;
        samplerInfo.addressModeW = adressMode;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.maxAnisotropy = 1.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 1.0f;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        sampler = device.createSampler(samplerInfo);
    }

    /**
                * Creates a default render pass setup with one sub pass
                *
                * @return VK_SUCCESS if all resources have been created successfully
                */
    void createRenderPass() {
        std::vector<vk::AttachmentDescription> attachmentDescriptions;
        for (auto& attachment : attachments) {
            attachmentDescriptions.push_back(attachment.description);
        };

        // Collect attachment references
        std::vector<vk::AttachmentReference> colorReferences;
        vk::AttachmentReference depthReference{ 0, vk::ImageLayout::eDepthStencilAttachmentOptimal };
        bool hasDepth = false;
        bool hasColor = false;
        uint32_t attachmentIndex = 0;
        for (auto& attachment : attachments) {
            if (attachment.isDepthStencil()) {
                if (hasDepth) {
                    throw std::runtime_error("Only one depth/stencil attachment allowed");
                }
                depthReference.attachment = attachmentIndex;
                hasDepth = true;
            } else {
                colorReferences.emplace_back(attachmentIndex, vk::ImageLayout::eColorAttachmentOptimal);
                hasColor = true;
            }
            attachmentIndex++;
        };

        // Default render pass setup uses only one subpass
        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        if (hasColor) {
            subpass.pColorAttachments = colorReferences.data();
            subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
        }
        if (hasDepth) {
            subpass.pDepthStencilAttachment = &depthReference;
        }

        // Use subpass dependencies for attachment layout transitions
        std::array<vk::SubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[0].dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        dependencies[1].srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
        dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
        dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

        // Create render pass
        renderPass = device.createRenderPass(
            { {}, static_cast<uint32_t>(attachmentDescriptions.size()), attachmentDescriptions.data(), 1, &subpass, 2, dependencies.data() });

        // Find. max number of layers across attachments
        std::vector<vk::ImageView> attachmentViews;
        uint32_t maxLayers = 0;
        for (auto attachment : attachments) {
            maxLayers = std::max(maxLayers, attachment.subresourceRange.layerCount);
            attachmentViews.push_back(attachment.view);
        }

        vk::FramebufferCreateInfo framebufferInfo;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.pAttachments = attachmentViews.data();
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachmentViews.size());
        framebufferInfo.width = size.width;
        framebufferInfo.height = size.height;
        framebufferInfo.layers = maxLayers;
        framebuffer = device.createFramebuffer(framebufferInfo);
    }
};
}  // namespace vks
