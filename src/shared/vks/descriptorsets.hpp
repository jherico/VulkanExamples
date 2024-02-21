#pragma once
#include <vulkan/vulkan.hpp>
#include <vector>
#include <unordered_map>

namespace vks { namespace descriptor {

class Writer {
public:
    std::vector<vk::WriteDescriptorSet> writes;
    std::unordered_map<uint32_t, size_t> writeSourceIndices;
    std::unordered_map<uint32_t, size_t> writeIndices;
    std::vector<vk::DescriptorImageInfo> imageInfos;
    std::vector<vk::DescriptorBufferInfo> bufferInfos;

    void parse(const std::vector<vk::DescriptorSetLayoutBinding>& layoutBindings);
    void setCombinedImageSampler(vk::DescriptorSet handle,
                                 uint32_t binding,
                                 vk::Sampler sampler,
                                 vk::ImageView view,
                                 vk::ImageLayout layout = vk::ImageLayout::eReadOnlyOptimal);
    void setUniformBuffer(vk::DescriptorSet handle, uint32_t binding, vk::Buffer buffer, vk::DeviceSize offset = 0, vk::DeviceSize range = VK_WHOLE_SIZE);
    bool valid() const;
};

}}  // namespace vks::descriptor