/*
* Vulkan buffer class
*
* Encapsulates a Vulkan buffer
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "allocation.hpp"

namespace vks {
/**
    * @brief Encapsulates access to a Vulkan buffer backed up by device memory
    * @note To be filled by an external source like the VulkanDevice
    */
struct Buffer : public Allocation {
    static void relocate(const VmaDefragmentationMove& move, Buffer& buffer);

private:
    using Parent = Allocation;

public:
    vk::BufferCreateInfo createInfo;
    vk::Buffer buffer;

    /** @brief Usage flags to be filled by external source at buffer creation (to query at some later point) */
    vk::DescriptorBufferInfo descriptor;

    struct Builder : public Parent::Builder<vks::Buffer::Builder> {
        vk::BufferCreateInfo createInfo;

        Builder(vk::DeviceSize size) { createInfo.size = size; }

        Builder& withBufferUsage(const vk::BufferUsageFlags& usage) {
            createInfo.usage = usage;
            return *this;
        }

        Buffer build() const {
            return Buffer{ createInfo, allocationCreateInfo };
        }
    };

    Buffer() = default;
    Buffer(const Buffer& other) = delete;
    Buffer(Buffer&& other);
    Buffer& operator=(Buffer&& other);

protected: 
    Buffer(const vk::BufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& allocationCreateInfo);

public:

    void create(const Builder& builder);
    vk::DescriptorBufferInfo getDescriptor(vk::DeviceSize offset = 0, vk::DeviceSize range = VK_WHOLE_SIZE) const;

    operator bool() const { return buffer.operator bool(); }

    void free() override {
        unmap();
        vmaDestroyBuffer(allocator, buffer, allocation);
        allocation = VK_NULL_HANDLE;
        allocInfo = {};
    }

    /**
        * Release all Vulkan resources held by this buffer
        */
    void destroy() override {
        if (buffer) {
            free();
            buffer = vk::Buffer{};
        }
        Parent::destroy();
    }
};
}  // namespace vks
