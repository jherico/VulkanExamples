#include "swapchain.hpp"

#include <format>
#include <rendering/context.hpp>
#include <vks/debug.hpp>

namespace vks { namespace swapchain {

vk::ImageSubresourceRange Image::getWholeRange() const {
    return vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, layers };
}
vk::ImageSubresourceLayers Image::getAllLayers() const {
    return vk::ImageSubresourceLayers{ vk::ImageAspectFlagBits::eColor, 0, 0, layers };
}
vk::ImageMemoryBarrier2 Image::getBarrier(const State& newState) const {
    auto result = vks::util::buildImageBarrier(image, getWholeRange(), state, newState);
    state = newState;
    return result;
}

void Image::setLayout(const vk::CommandBuffer& cmdBuffer, const State& newState) const {
    auto barrier = getBarrier(newState);
    cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, barrier });
}

// Prefer mailbox mode if present, it's the lowest latency non-tearing present  mode
vk::PresentModeKHR Swapchain::pickPresentMode(const vk::PhysicalDevice& physicalDevice, vk::SurfaceKHR surface) {
    vk::PresentModeKHR result = vk::PresentModeKHR::eFifo;
    std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(surface);
    for (const auto& presentMode : presentModes) {
        if (presentMode == vk::PresentModeKHR::eMailbox) {
            return vk::PresentModeKHR::eMailbox;
        }
        if (presentMode == vk::PresentModeKHR::eImmediate) {
            result = vk::PresentModeKHR::eImmediate;
        }
    }
    return result;
}

// Creates an os specific surface
// Tries to find a graphics and a present graphicsQueue
void Swapchain::create(const Builder& builder) {
    const auto& context = vks::Context::get();
    const auto& physicalDevice = context.physicalDevice;
    const auto& device = context.device;

    // Get list of supported surface formats
    if (surfaceFormat.format == vk::Format::eUndefined) {
        std::vector<vk::SurfaceFormatKHR> surfaceFormats = physicalDevice.getSurfaceFormatsKHR(builder.surface);
        auto formatCount = surfaceFormats.size();
        // If the surface format list only includes one entry with  vk::Format::eUndefined,
        // there is no preferered format, so we assume  vk::Format::eB8G8R8A8Unorm
        if ((formatCount == 1) && (surfaceFormats[0].format == vk::Format::eUndefined)) {
            surfaceFormat.format = vk::Format::eB8G8R8A8Unorm;
            surfaceFormat.colorSpace = surfaceFormats[0].colorSpace;
        } else {
            for (const auto& surfaceFormat : surfaceFormats) {
                if (surfaceFormat.colorSpace == builder.colorSpace) {
                    this->surfaceFormat = surfaceFormat;
                    break;
                }
            }
        }
    }

    if (surfaceFormat.format == vk::Format::eUndefined) {
        throw std::runtime_error("Unable to find desired surface format");
    }

    // Get physical device surface properties and formats
    vk::SurfaceCapabilitiesKHR surfCaps = physicalDevice.getSurfaceCapabilitiesKHR(builder.surface);

    vk::Extent2D swapchainExtent;
    // width and height are either both -1, or both not -1.
    if (surfCaps.currentExtent.width == -1) {
        // If the surface size is undefined, the size is set to
        // the size of the images requested.
        swapchainExtent = builder.size;
    } else {
        // If the surface size is defined, the swap chain size must match
        swapchainExtent = surfCaps.currentExtent;
    }

    // Build the swapchain
    vk::SwapchainCreateInfoKHR swapchainCI{ {}, builder.surface };
    {
        // If we don't support identity transform, pick the current transform
        if (!(surfCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)) {
            swapchainCI.preTransform = surfCaps.currentTransform;
        }
        // Set the number of images
        swapchainCI.minImageCount = surfCaps.minImageCount + 1;
        // If a max is specified, make sure we don't exceed it
        if (surfCaps.maxImageCount > 0) {
            swapchainCI.minImageCount = std::min(surfCaps.maxImageCount, swapchainCI.minImageCount);
        }
        if (builder.vsync) {
            swapchainCI.presentMode = vk::PresentModeKHR::eFifo;
        } else {
            swapchainCI.presentMode = pickPresentMode(physicalDevice, builder.surface);
        }

        swapchainCI.imageFormat = surfaceFormat.format;
        swapchainCI.imageColorSpace = surfaceFormat.colorSpace;
        swapchainCI.imageArrayLayers = builder.layers;
        swapchainCI.imageExtent = vk::Extent2D{ swapchainExtent.width, swapchainExtent.height };
        swapchainCI.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
        swapchainCI.oldSwapchain = handle;
        swapchainCI.clipped = VK_TRUE;
        handle = device.createSwapchainKHR(swapchainCI);
    }

    // If an existing sawp chain is re-created, destroy the old swap chain
    // This also cleans up all the presentable images
    if (swapchainCI.oldSwapchain) {
        destroyImageResources();
        device.destroy(swapchainCI.oldSwapchain);
    }

    vk::ImageViewCreateInfo colorAttachmentView;
    colorAttachmentView.format = surfaceFormat.format;
    colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    colorAttachmentView.subresourceRange.levelCount = 1;
    colorAttachmentView.subresourceRange.layerCount = builder.layers;
    colorAttachmentView.viewType = builder.layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;

    // Get the swap chain images
    auto swapChainImages = device.getSwapchainImagesKHR(handle);
    imageCount = (uint32_t)swapChainImages.size();
    // Get the swap chain buffers containing the image and imageview
    images.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        auto& image = images[i];
        image.image = swapChainImages[i];
        image.layers = builder.layers;
        auto name = std::format("Swapchain image {:#0x}", i);
        vks::debug::marker::setObjectName(device, image.image, name);

        colorAttachmentView.image = swapChainImages[i];
        image.view = device.createImageView(colorAttachmentView);
    }
}

// Acquires the next image in the swap chain
vk::ResultValue<uint32_t> Swapchain::acquireNextImage(const vk::Semaphore& presentCompleteSemaphore, const vk::Fence& fence) {
    static const auto& device = vks::Context::get().device;
    auto resultValue = device.acquireNextImageKHR(handle, UINT64_MAX, presentCompleteSemaphore, fence);
    vk::Result result = resultValue.result;
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::error_code(result);
    }
    return resultValue;
}

std::vector<vk::ImageView> Swapchain::getViews() {
    std::vector<vk::ImageView> result;
    result.reserve(images.size());
    for (const auto& image : images) {
        result.push_back(image.view);
    }
    return result;
}

void Swapchain::destroyImageResources() {
    static const auto& device = vks::Context::get().device;
    for (auto& image : images) {
        if (image.view) {
            device.destroy(image.view);
            image.view = nullptr;
        }
    }
}

// Free all Vulkan resources used by the swap chain
void Swapchain::destroy() {
    static const auto& device = vks::Context::get().device;
    destroyImageResources();
    device.destroy(handle);
    handle = nullptr;
}

}}  // namespace vks::swapchain