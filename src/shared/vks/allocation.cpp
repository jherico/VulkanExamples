#define VMA_IMPLEMENTATION
#include "allocation.hpp"
// #include "queues.hpp"
#include "device.hpp"

// #define VMA_DYNAMIC_VULKAN_FUNCTIONS

namespace vma {
class Allocator {
    using CType = VmaAllocator;
    using NativeType = VmaAllocator;

public:
    VULKAN_HPP_CONSTEXPR Allocator() = default;
    VULKAN_HPP_CONSTEXPR Allocator(std::nullptr_t) VULKAN_HPP_NOEXCEPT {}
    VULKAN_HPP_TYPESAFE_EXPLICIT Allocator(VmaAllocator allocator) VULKAN_HPP_NOEXCEPT : m_allocator(allocator) {}

#if defined(VULKAN_HPP_TYPESAFE_CONVERSION)
    Allocator& operator=(VmaAllocator allocator) VULKAN_HPP_NOEXCEPT {
        m_allocator = allocator;
        return *this;
    }
#endif

    Allocator& operator=(std::nullptr_t) VULKAN_HPP_NOEXCEPT {
        m_allocator = {};
        return *this;
    }

#if defined(VULKAN_HPP_HAS_SPACESHIP_OPERATOR)
    auto operator<=>(Allocator const&) const = default;
#else
    bool operator==(Allocator const& rhs) const VULKAN_HPP_NOEXCEPT { return m_allocator == rhs.m_allocator; }

    bool operator!=(Allocator const& rhs) const VULKAN_HPP_NOEXCEPT { return m_allocator != rhs.m_allocator; }

    bool operator<(Allocator const& rhs) const VULKAN_HPP_NOEXCEPT { return m_allocator < rhs.m_allocator; }
#endif

    VULKAN_HPP_TYPESAFE_EXPLICIT operator VmaAllocator() const VULKAN_HPP_NOEXCEPT { return m_allocator; }

    explicit operator bool() const VULKAN_HPP_NOEXCEPT { return m_allocator != VK_NULL_HANDLE; }

    bool operator!() const VULKAN_HPP_NOEXCEPT { return m_allocator == VK_NULL_HANDLE; }

private:
    VmaAllocator m_allocator = {};
    VmaAllocatorCreateInfo allocatorCreateInfo{};

public:
    void create(vk::Instance instance, const vks::DeviceInfo& deviceInfo, const vk::PhysicalDevice& physicalDevice, const vk::Device& device) {
        // These flags aren't technically needed because they've both been promoted to core by Vulkan 1.3, which is our minimum target
        allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT | VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
        if (deviceInfo.hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        }
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorCreateInfo.physicalDevice = physicalDevice;
        allocatorCreateInfo.device = device;
        allocatorCreateInfo.instance = instance;
        auto result = static_cast<vk::Result>(vmaCreateAllocator(&allocatorCreateInfo, &m_allocator));
        vk::resultCheck(result, "VMA initialization");
    }

    void destroy() {
        vmaDestroyAllocator(m_allocator);
        m_allocator = nullptr;
        allocatorCreateInfo = {};
    }
};

class Allocation {};

}  // namespace vma

VmaAllocator vks::Allocation::allocator{};
vk::Device vks::Allocation::device{};

static VmaAllocatorCreateInfo allocatorCreateInfo{};

VmaAllocation vks::Allocation::allocatePages(const vk::MemoryRequirements& memoryRequirements) {
    VmaAllocationCreateInfo createInfo{};

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.flags = 0;
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    VmaAllocation allocation;

    // allocCreateInfo.pUserData = this;

    vk::Result allocResult = static_cast<vk::Result>(
        vmaAllocateMemoryPages(allocator, &memoryRequirements.operator const VkMemoryRequirements&(), &allocCreateInfo, 1, &allocation, nullptr));
    vk::resultCheck(allocResult, "VMA sparse allocation");
    return allocation;
}

void vks::Allocation::freePages(VmaAllocation allocation) {
    vmaFreeMemoryPages(allocator, 1, &allocation);
}

VmaAllocationInfo vks::Allocation::getAllocationInfo(VmaAllocation allocation) {
    VmaAllocationInfo result;
    vmaGetAllocationInfo(allocator, allocation, &result);
    return result;
}

void vks::Allocation::init(vk::Instance instance, const vks::DeviceInfo& deviceInfo, const vk::PhysicalDevice& physicalDevice, const vk::Device& device) {
    vks::Allocation::device = device;

    // These flags aren't technically needed because they've both been promoted to core by Vulkan 1.3, which is our minimum target
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT | VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
    if (deviceInfo.hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorCreateInfo.physicalDevice = physicalDevice;
    allocatorCreateInfo.device = device;
    allocatorCreateInfo.instance = instance;
    vmaCreateAllocator(&allocatorCreateInfo, &allocator);
}

void vks::Allocation::shutdown() {
    vmaDestroyAllocator(allocator);
}

void vks::Allocation::unmap() {
#if USE_VMA
    if (allocInfo.pMappedData != nullptr && !allocation->IsPersistentMap()) {
        vmaUnmapMemory(allocator, allocation);
        allocInfo.pMappedData = nullptr;
    }
#else
    if (mapped != nullptr) {
        device.unmapMemory(memory);
    }
    mapped = nullptr;
#endif
}

void* vks::Allocation::mapRaw() {
#if USE_VMA
    if (allocInfo.pMappedData == nullptr) {
        auto result = vmaMapMemory(allocator, allocation, &allocInfo.pMappedData);
        vk::resultCheck(static_cast<VULKAN_HPP_NAMESPACE::Result>(result), "vmaMapMemory");
    }
    return allocInfo.pMappedData;
#else
    mapped = device.mapMemory(memory, offset, size, vk::MemoryMapFlags());
    return (T*)mapped;
#endif
}

void vks::Allocation::free() {
#if USE_VMA
    if (allocation) {
        vmaFreeMemory(allocator, allocation);
        allocation = VK_NULL_HANDLE;
        allocInfo = {};
    }
#else
    if (memory) {
        device.free(memory);
        memory = vk::DeviceMemory();
    }
#endif
}

void vks::Allocation::copy(size_t size, const void* data, VkDeviceSize offset) const {
#if USE_VMA
    const auto& mapped = allocInfo.pMappedData;
#endif
    memcpy(static_cast<uint8_t*>(mapped) + offset, data, size);
    return;
}

void vks::Allocation::copyOut(size_t size, void* output, VkDeviceSize offset) const {
#if USE_VMA
    const auto& mapped = allocInfo.pMappedData;
#endif
    memcpy(output, static_cast<uint8_t*>(mapped) + offset, size);
    return;
}

#if 0
void vks::Image::relocate(const vk::Device& device, const vk::CommandBuffer& cmdBuf, const VmaDefragmentationMove& move, vks::Image& image) {
    // Recreate and bind this buffer/image at: pass.pMoves[i].dstMemory, pass.pMoves[i].dstOffset.
    auto newImage = device.createImage(image.createInfo);
    auto res = vmaBindImageMemory(allocator, move.dstTmpAllocation, newImage);
    // FIXME Check res...

    // Issue a vkCmdCopyBuffer/vkCmdCopyImage to copy its content to the new place.
    cmdBuf.copyImage(image.image, ..., newImage, ...);
}

void vks::Buffer::relocate(const VmaDefragmentationMove& move, vks::Buffer& image) {
}

void relocate(VmaAllocator allocator, const VmaDefragmentationMove& move) {
    // Inspect pass.pMoves[i].srcAllocation, identify what buffer/image it represents.
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(allocator, move.srcAllocation, &allocInfo);
    vks::Allocation* resData = (vks::Allocation*)allocInfo.pUserData;
    if (resData->objectType == vk::DebugReportObjectTypeEXT::eBuffer) {
        vks::Buffer::relocate(allocator, move, *reinterpret_cast<vks::Buffer*>(resData));
    } else if (resData->objectType == vk::DebugReportObjectTypeEXT::eImage) {
        vks::Image::relocate(allocator, move, *reinterpret_cast<vks::Image*>(resData));
    }


}

void vks::Allocation::defragment(const vks::QueueManager& queueManager) {
    VmaDefragmentationInfo defragInfo = {};
    defragInfo.pool = pool;
    defragInfo.flags = VMA_DEFRAGMENTATION_FLAG_ALGORITHM_FAST_BIT;

    VmaDefragmentationContext defragCtx;
    VkResult res = vmaBeginDefragmentation(allocator, &defragInfo, &defragCtx);
    // Check res...

    for (;;) {
        VmaDefragmentationPassMoveInfo pass;
        res = vmaBeginDefragmentationPass(allocator, defragCtx, &pass);
        if (res == VK_SUCCESS)
            break;
        else if (res != VK_INCOMPLETE)
            // Handle error...
            for (uint32_t i = 0; i < pass.moveCount; ++i) {
                relocate(pass.pMoves[i]);
            }

        // Make sure the copy commands finished executing.
        vkWaitForFences(...);

        // Destroy old buffers/images bound with pass.pMoves[i].srcAllocation.
        for (uint32_t i = 0; i < pass.moveCount; ++i) {
            // ...
            vkDestroyImage(device, resData->image, nullptr);
        }

        // Update appropriate descriptors to point to the new places...

        res = vmaEndDefragmentationPass(allocator, defragCtx, &pass);
        if (res == VK_SUCCESS)
            break;
        else if (res != VK_INCOMPLETE)
        // Handle error...
    }

    vmaEndDefragmentation(allocator, defragCtx, nullptr);
}
#endif