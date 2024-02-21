/*
* Class wrapping access to the swap chain
*
* A swap chain is a collection of framebuffers used for rendering
* The swap chain images can then presented to the windowing system
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <vulkan/vulkan.hpp>
#include <vks/helpers.hpp>

namespace vks {
namespace swapchain {

struct Image {
    using State = vks::util::ImageTransitionState;

    uint32_t layers{ 1 };
    vk::Image image;
    vk::ImageView view;
    mutable State state{ State::UNDEFINED };

    vk::ImageSubresourceRange getWholeRange() const;
    vk::ImageSubresourceLayers getAllLayers() const;
    vk::ImageMemoryBarrier2 getBarrier(const State& newState) const;
    void setLayout(const vk::CommandBuffer& cmdBuffer, const State& newState) const;
};

using Images = std::vector<Image>;

struct Builder {
    Builder(const vk::Extent2D& size, vk::SurfaceKHR surface)
        : size(size)
        , surface(surface) {}

    vk::SurfaceKHR surface;
    uint32_t layers{ 1 };
    vk::ColorSpaceKHR colorSpace{ vk::ColorSpaceKHR::eSrgbNonlinear };
    vk::Extent2D size;
    bool vsync{ false };
};

struct Swapchain {
    vk::SwapchainKHR handle;
    vk::SurfaceFormatKHR surfaceFormat{ vk::Format::eUndefined, vk::ColorSpaceKHR::eSrgbNonlinear };
    // Builder builder;
    Images images;
    uint32_t imageCount{ 0 };

    // Prefer mailbox mode if present, it's the lowest latency non-tearing present  mode
    static vk::PresentModeKHR pickPresentMode(const vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR surface);

    // Creates an OS-specific surface
    // Tries to find a graphics and a present graphicsQueue
    void create(const Builder& builder);

    // Acquires the next image in the swap chain
    vk::ResultValue<uint32_t> acquireNextImage(const vk::Semaphore& presentCompleteSemaphore, const vk::Fence& fence = nullptr);
    std::vector<vk::ImageView> getViews();
    void destroyImageResources();
    // Free all Vulkan resources used by the swap chain
    void destroy();
};

}  // namespace swapchain

using SwapchainImage = swapchain::Image;
using Swapchain = swapchain::Swapchain;

}  // namespace vks
