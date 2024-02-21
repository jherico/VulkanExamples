#pragma once
#include <vulkan/vulkan.hpp>

namespace vks {

struct QueueFamilyInfo {
    uint32_t index{ VK_QUEUE_FAMILY_IGNORED };
    vk::QueueFamilyProperties2 properties;

    QueueFamilyInfo() = default;
    QueueFamilyInfo(const uint32_t& index, const vk::QueueFamilyProperties2& properties)
        : index(index)
        , properties(properties) {}

    operator bool() const { return index != VK_QUEUE_FAMILY_IGNORED; };
};

struct QueuesInfo {
    using QueuesProperties = std::vector<vk::QueueFamilyProperties2>;
    QueuesProperties queueFamilyProperties;
    QueueFamilyInfo graphics;
    QueueFamilyInfo transfer;
    QueueFamilyInfo compute;

    QueuesInfo() = default;
    QueuesInfo(const vk::PhysicalDevice& physicalDevice);

    QueueFamilyInfo findQueue(const vk::PhysicalDevice& physicalDevice,
                              const vk::QueueFlags& desiredFlags,
                              const vk::SurfaceKHR& presentSurface = nullptr) const;
};

struct QueueManager {
private:
public:
    vk::Device device;
    QueueFamilyInfo familyInfo;
    vk::Queue handle;
    vk::CommandPool pool;
    vk::QueueFlagBits primaryFlag{};
    uint32_t index{ 0 };

    operator bool() const { return handle.operator bool(); }

    QueueManager() = default;
    QueueManager(const vk::Device& device, const QueueFamilyInfo& familyInfo);
    void destroy(bool flush = false);
    void flush();

    vk::CommandPool createCommandPool(const vk::CommandPoolCreateFlags& flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer) const;

    std::vector<vk::CommandBuffer> allocateCommandBuffers(uint32_t count, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;

    vk::CommandBuffer createCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;

    void freeCommandBuffers(const vk::ArrayProxy<vk::CommandBuffer>& commandBuffers) const;

    void freeCommandBuffer(const vk::CommandBuffer& commandBuffer) const;

    void submit2(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers, const vk::Fence& fence) const;

    void submit2(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers,
                 const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& wait = nullptr,
                 const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& signal = nullptr,
                 const vk::Fence& fence = vk::Fence{}) const;

    void submitAndWait(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers) const;

    VULKAN_HPP_NODISCARD
    vk::Result present(const vk::SwapchainKHR& swapchain, const uint32_t& index, const vk::Semaphore& waitSemaphore = {});
};

}  // namespace vks
