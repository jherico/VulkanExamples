#include "queues.hpp"

namespace vks {

QueuesInfo::QueuesInfo(const vk::PhysicalDevice& physicalDevice) {
    queueFamilyProperties = physicalDevice.getQueueFamilyProperties2();
    {
        auto candidate = findQueue(physicalDevice, vk::QueueFlagBits::eTransfer);
        auto flags = candidate.properties.queueFamilyProperties.queueFlags;
        if (!(flags & vk::QueueFlagBits::eCompute) && !(flags & vk::QueueFlagBits::eGraphics)) {
            transfer = candidate;
        }
    }

    {
        auto candidate = findQueue(physicalDevice, vk::QueueFlagBits::eCompute);
        auto flags = candidate.properties.queueFamilyProperties.queueFlags;
        if (!(flags & vk::QueueFlagBits::eGraphics)) {
            compute = candidate;
        }
    }

    graphics = findQueue(physicalDevice, vk::QueueFlagBits::eGraphics);
}

QueueFamilyInfo QueuesInfo::findQueue(const vk::PhysicalDevice& physicalDevice,
                                      const vk::QueueFlags& desiredFlags,
                                      const vk::SurfaceKHR& presentSurface) const {
    uint32_t bestMatchIndex{ VK_QUEUE_FAMILY_IGNORED };
    vk::QueueFamilyProperties2 bestMatchProperties{};
    int bestExtraBits = std::popcount(static_cast<VkQueueFlags>(VK_QUEUE_FLAG_BITS_MAX_ENUM));
    size_t queueCount = queueFamilyProperties.size();
    for (uint32_t i = 0; i < queueCount; ++i) {
        const auto& properties = queueFamilyProperties[i];
        const auto& currentFlags = properties.queueFamilyProperties.queueFlags;

        // Doesn't contain the required flags, skip it
        if (!(currentFlags & desiredFlags)) {
            continue;
        }

        // If a surface has been provided and that surface isn't supported by the graphicsQueue, skip it
        if (presentSurface && VK_FALSE == physicalDevice.getSurfaceSupportKHR(i, presentSurface)) {
            continue;
        }

        int currentExtraBits = std::popcount((currentFlags & ~desiredFlags).operator VkQueueFlags());

        // If we find an exact match, return immediately
        // This is no longer really a likely case due to things like sparse bindings.
        if (0 == currentExtraBits) {
            return { i, properties };
        }

        // Use this graphicsQueue if it has fewer extra flags than the current best match
        if (bestMatchIndex == VK_QUEUE_FAMILY_IGNORED || (currentExtraBits < bestExtraBits)) {
            bestMatchIndex = i;
            bestExtraBits = currentExtraBits;
            bestMatchProperties = properties;
        }
    }

    return { bestMatchIndex, bestMatchProperties };
}

QueueManager::QueueManager(const vk::Device& device, const QueueFamilyInfo& familyInfo)
    : device(device)
    , familyInfo(familyInfo) {
    const auto& flags = familyInfo.properties.queueFamilyProperties.queueFlags;
    if (flags & vk::QueueFlagBits::eGraphics) {
        primaryFlag = vk::QueueFlagBits::eGraphics;
    } else if (flags & vk::QueueFlagBits::eCompute) {
        primaryFlag = vk::QueueFlagBits::eCompute;
    } else {
        primaryFlag = vk::QueueFlagBits::eTransfer;
    }
    handle = device.getQueue2({ {}, familyInfo.index, index });
    pool = device.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, familyInfo.index });
}

void QueueManager::destroy(bool flush) {
    if (flush) {
        this->flush();
    }
    if (pool) {
        device.destroy(pool);
        pool = vk::CommandPool{};
    }
    handle = vk::Queue{};
}

void QueueManager::flush() {
    if (handle) {
        handle.waitIdle();
    }
}

vk::CommandPool QueueManager::createCommandPool(const vk::CommandPoolCreateFlags& flags) const {
    return device.createCommandPool({ flags, familyInfo.index });
}

std::vector<vk::CommandBuffer> QueueManager::allocateCommandBuffers(uint32_t count, vk::CommandBufferLevel level) const {
    return device.allocateCommandBuffers({ pool, level, count });
}

vk::CommandBuffer QueueManager::createCommandBuffer(vk::CommandBufferLevel level) const {
    return allocateCommandBuffers(1, level)[0];
}

void QueueManager::freeCommandBuffers(const vk::ArrayProxy<vk::CommandBuffer>& commandBuffers) const {
    device.free(pool, commandBuffers);
}

void QueueManager::freeCommandBuffer(const vk::CommandBuffer& commandBuffer) const {
    freeCommandBuffers(vk::ArrayProxy<vk::CommandBuffer>{ commandBuffer });
}

void QueueManager::submit2(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers, const vk::Fence& fence) const {
    submit2(commandBuffers, nullptr, nullptr, fence);
}

void QueueManager::submit2(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers,
                           const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& wait,
                           const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& signal,
                           const vk::Fence& fence) const {
    std::vector<vk::CommandBufferSubmitInfo> commandBufferInfos;
    commandBufferInfos.reserve(commandBuffers.size());
    for (const auto& commandBuffer : commandBuffers) {
        commandBufferInfos.emplace_back(vk::CommandBufferSubmitInfo{ commandBuffer });
    }
    handle.submit2(vk::SubmitInfo2{ {}, wait, commandBufferInfos, signal }, fence);
}

vk::Result QueueManager::present(const vk::SwapchainKHR& swapchain, const uint32_t& index, const vk::Semaphore& waitSemaphore) {
    return handle.presentKHR(vk::PresentInfoKHR{ waitSemaphore, swapchain, index });
}

void QueueManager::submitAndWait(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers) const {
    vk::Fence fence = device.createFence({});
    submit2(commandBuffers, fence);
    auto waitResult = device.waitForFences(fence, true, UINT64_MAX);
    assert(waitResult != vk::Result::eTimeout);
    device.destroy(fence);
}

}  // namespace vks
