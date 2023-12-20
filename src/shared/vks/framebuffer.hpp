//
//  Created by Bradley Austin Davis on 2016/03/19
//  Copyright 2013 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#pragma once

#include <vector>
#include <algorithm>
#include <vulkan/vulkan.hpp>

#include "context.hpp"

namespace vks {
using FramebufferAttachment = Image;
struct Framebuffer {
    using Attachment = FramebufferAttachment;
    vk::Device device;
    vk::Framebuffer framebuffer;
    Attachment depth;
    std::vector<Attachment> colors;

    void destroy() {
        for (auto& color : colors) {
            color.destroy();
        }
        if (depth.format != vk::Format::eUndefined) {
            depth.destroy();
        }
        if (framebuffer) {
            device.destroyFramebuffer(framebuffer);
            framebuffer = vk::Framebuffer();
        }
    }

    // Prepare a new framebuffer for offscreen rendering
    // The contents of this framebuffer are then
    // blitted to our render target
    void create(const vks::Context& context,
                const glm::uvec2& size,
                const std::vector<vk::Format>& colorFormats,
                vk::Format depthFormat,
                const vk::RenderPass& renderPass,
                vk::ImageUsageFlags colorUsage = vk::ImageUsageFlagBits::eSampled,
                vk::ImageUsageFlags depthUsage = vk::ImageUsageFlags()) {
        device = context.device;
        destroy();

        colors.resize(colorFormats.size());

        // Color attachment
        vk::ImageCreateInfo image;
        image.imageType = vk::ImageType::e2D;
        image.extent.width = size.x;
        image.extent.height = size.y;
        image.extent.depth = 1;
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = vk::SampleCountFlagBits::e1;
        image.tiling = vk::ImageTiling::eOptimal;
        // vk::Image of the framebuffer is blit source
        image.usage = vk::ImageUsageFlagBits::eColorAttachment | colorUsage;

        vk::ImageViewCreateInfo colorImageView;
        colorImageView.viewType = vk::ImageViewType::e2D;
        colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorImageView.subresourceRange.levelCount = 1;
        colorImageView.subresourceRange.layerCount = 1;

        for (size_t i = 0; i < colorFormats.size(); ++i) {
            image.format = colorFormats[i];
            colors[i] = context.createImage(image, vk::MemoryPropertyFlagBits::eDeviceLocal);
            colorImageView.format = colorFormats[i];
            colorImageView.image = colors[i].image;
            colors[i].view = device.createImageView(colorImageView);
        }

        bool useDepth = depthFormat != vk::Format::eUndefined;
        // Depth stencil attachment
        if (useDepth) {
            image.format = depthFormat;
            image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | depthUsage;
            depth = context.createImage(image, vk::MemoryPropertyFlagBits::eDeviceLocal);

            vk::ImageViewCreateInfo depthStencilView;
            depthStencilView.viewType = vk::ImageViewType::e2D;
            depthStencilView.format = depthFormat;
            depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            depthStencilView.subresourceRange.levelCount = 1;
            depthStencilView.subresourceRange.layerCount = 1;
            depthStencilView.image = depth.image;
            depth.view = device.createImageView(depthStencilView);
        }

        std::vector<vk::ImageView> attachments;
        attachments.resize(colors.size());
        for (size_t i = 0; i < colors.size(); ++i) {
            attachments[i] = colors[i].view;
        }
        if (useDepth) {
            attachments.push_back(depth.view);
        }

        vk::FramebufferCreateInfo fbufCreateInfo;
        fbufCreateInfo.renderPass = renderPass;
        fbufCreateInfo.attachmentCount = (uint32_t)attachments.size();
        fbufCreateInfo.pAttachments = attachments.data();
        fbufCreateInfo.width = size.x;
        fbufCreateInfo.height = size.y;
        fbufCreateInfo.layers = 1;
        framebuffer = context.device.createFramebuffer(fbufCreateInfo);
    }
};
}  // namespace vks
