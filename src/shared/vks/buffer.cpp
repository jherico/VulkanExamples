#include "buffer.hpp"

vk::DescriptorBufferInfo vks::Buffer::getDescriptor(vk::DeviceSize offset, vk::DeviceSize range) const {
    return vk::DescriptorBufferInfo{ buffer, offset, range };
}

void vks::Buffer::create(const vks::Buffer::Builder& builder) {
    *this = builder.build();
}

vks::Buffer::Buffer(Buffer&& other)
    : Parent(static_cast<Parent&&>(other))
    , createInfo(other.createInfo)
    , buffer(std::exchange(other.buffer, nullptr))
    , descriptor(other.descriptor) {
}

vks::Buffer& vks::Buffer::operator=(vks::Buffer&& other) {
    Parent::operator=(static_cast<Parent&&>(other));
    buffer = std::exchange(other.buffer, nullptr);
    descriptor = std::exchange(other.descriptor, {});
    createInfo = std::exchange(other.createInfo, {});
    return *this;
}

vks::Buffer::Buffer(const vk::BufferCreateInfo& bufferCreateInfo, const VmaAllocationCreateInfo& allocationCreateInfo)
    : createInfo(bufferCreateInfo) {
    VkBuffer rawBuffer;
    vmaCreateBuffer(allocator, &createInfo.operator VkBufferCreateInfo&(), &allocationCreateInfo, &rawBuffer, &allocation, &allocInfo);
    buffer = rawBuffer;
    descriptor = getDescriptor();
}
