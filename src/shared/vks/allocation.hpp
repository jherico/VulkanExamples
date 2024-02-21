#pragma once

#include <vulkan/vulkan.hpp>
#include "forward.hpp"
#ifndef USE_VMA
#error "VMA config not defined"
#endif

#if USE_VMA
#include <vk_mem_alloc.h>
#endif

namespace vks {

// A wrapper class for an allocation, either an Image or Buffer.  Not intended to be used used directly
// but only as a base class providing common functionality for the classes below.
//
// Provides easy to use mechanisms for mapping, unmapping and copying host data to the device memory
struct Allocation {
private:
protected:
    static VmaAllocator allocator;
    static vk::Device device;

    VmaAllocation allocation{ VK_NULL_HANDLE };
    VmaAllocationInfo allocInfo{};

public:
    template <typename BuilderType>
    struct Builder {
        VmaAllocationCreateInfo allocationCreateInfo{ .usage = VMA_MEMORY_USAGE_AUTO };
        BuilderType& withAllocCreateFlags(VmaAllocationCreateFlags flags) {
            allocationCreateInfo.flags = flags;
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocUsage(VmaMemoryUsage usage) {
            allocationCreateInfo.usage = usage;
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocRequiredFlags(const vk::MemoryPropertyFlags& requiredFlags) {
            allocationCreateInfo.requiredFlags = requiredFlags.operator VkMemoryPropertyFlags();
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocPreferredFlags(const vk::MemoryPropertyFlags& preferredFlags) {
            allocationCreateInfo.preferredFlags = preferredFlags.operator VkMemoryPropertyFlags();
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocMemoryTypeBits(uint32_t memoryTypeBits) {
            allocationCreateInfo.memoryTypeBits = memoryTypeBits;
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocPool(VmaPool pool) {
            allocationCreateInfo.pool = pool;
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocUserData(void* pUserData) {
            allocationCreateInfo.pUserData = pUserData;
            return static_cast<BuilderType&>(*this);
        }
        BuilderType& withAllocPriority(float priority) {
            allocationCreateInfo.priority = priority;
            return *this;
        }
    };

    static void init(vk::Instance instance, const vks::DeviceInfo& deviceInfo, const vk::PhysicalDevice& physicalDevice, const vk::Device& device);
    static void shutdown();
    static VmaAllocation allocatePages(const vk::MemoryRequirements& memoryRequirements);
    static void freePages(VmaAllocation allocation);
    static VmaAllocationInfo getAllocationInfo(VmaAllocation allocation);
    //static void defragment(const vk::CommandPool& pool);

    Allocation() = default;
    Allocation(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept
        : allocation{ std::exchange(other.allocation, VK_NULL_HANDLE) }
        , allocInfo{ std::exchange(other.allocInfo, {}) } {}

    Allocation& operator=(Allocation&& other) {
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        allocInfo = std::exchange(other.allocInfo, {});
        return *this;
    }

    virtual void free();

    void* mapRaw();
    void unmap();
    void copy(size_t size, const void* data, VkDeviceSize offset = 0) const;
    void copyOut(size_t size, void* data, VkDeviceSize offset) const;

    uint32_t getMemoryType() const { return allocInfo.memoryType; }

    template <typename T = void>
    inline T* map() {
        return (T*)mapRaw();
    }

    template <typename T>
    inline void copy(const T& data, VkDeviceSize offset = 0) const {
        copy(sizeof(T), &data, offset);
    }

    template <typename T>
    void copyOut(T& t, VkDeviceSize offset = 0) {
        copyOut(sizeof(T), &t, offset);
    }

    template <typename T>
    inline void copy(const std::vector<T>& data, VkDeviceSize offset = 0) const {
        copy(sizeof(T) * data.size(), data.data(), offset);
    }

    template <typename T>
    inline void copy(const vk::ArrayProxy<T>& data, VkDeviceSize offset = 0) const {
        copy(sizeof(T) * data.size(), data.data(), offset);
    }

    template <typename T>
    inline void copyWithStride(const vk::ArrayProxy<T>& data, vk::DeviceSize stride, VkDeviceSize offset = 0) const {
        for (size_t i = 0; i < data.size(); i++) {
            copy(sizeof(T), data.data() + i, offset + i * stride);
        }
    }

    /**
        * Flush a memory range of the buffer to make it visible to the device
        *
        * @note Only required for non-coherent memory
        *
        * @param size (Optional) Size of the memory range to flush. Pass VK_WHOLE_SIZE to flush the complete buffer range.
        * @param offset (Optional) Byte offset from beginning
        *
        * @return VkResult of the flush call
        */
    void flush(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
#if USE_VMA
        auto result = vmaFlushAllocation(allocator, allocation, offset, size);
        vk::resultCheck(static_cast<VULKAN_HPP_NAMESPACE::Result>(result), "vmaFlushAllocation");
#else
        device.flushMappedMemoryRanges(vk::MappedMemoryRange{ memory, offset, size });
#endif
    }

    /**
        * Invalidate a memory range of the buffer to make it visible to the host
        *
        * @note Only required for non-coherent memory
        *
        * @param size (Optional) Size of the memory range to invalidate. Pass VK_WHOLE_SIZE to invalidate the complete buffer range.
        * @param offset (Optional) Byte offset from beginning
        *
        * @return VkResult of the invalidate call
        */
    void invalidate(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0) {
#if USE_VMA
        auto result = vmaInvalidateAllocation(allocator, allocation, offset, size);
        vk::resultCheck(static_cast<VULKAN_HPP_NAMESPACE::Result>(result), "vmaInvalidateAllocation");
#else
        device.invalidateMappedMemoryRanges(vk::MappedMemoryRange{ memory, offset, size });
#endif
    }

    virtual void destroy() { free(); }
};

}  // namespace vks
