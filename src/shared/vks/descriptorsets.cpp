#include "descriptorsets.hpp"

void vks::descriptor::Writer::parse(const std::vector<vk::DescriptorSetLayoutBinding>& layoutBindings) {
    for (const auto& binding : layoutBindings) {
        writeIndices[binding.binding] = writes.size();
        writes.emplace_back();
        if (binding.descriptorType == vk::DescriptorType::eCombinedImageSampler) {
            writeSourceIndices[binding.binding] = imageInfos.size();
            imageInfos.emplace_back();
        } else if (binding.descriptorType == vk::DescriptorType::eUniformBuffer) {
            writeSourceIndices[binding.binding] = bufferInfos.size();
            bufferInfos.emplace_back();
        }
    }
}

void vks::descriptor::Writer::setCombinedImageSampler(vk::DescriptorSet handle,
                                                      uint32_t binding,
                                                      vk::Sampler sampler,
                                                      vk::ImageView view,
                                                      vk::ImageLayout layout) {
    const auto& writeIndex = writeIndices[binding];
    auto& write = writes[writeIndex];
    const auto& bindingIndex = writeSourceIndices[binding];
    auto& imageInfo = imageInfos[bindingIndex];
    imageInfo = vk::DescriptorImageInfo{ sampler, view, layout };
    write = vk::WriteDescriptorSet{ handle, binding, 0, vk::DescriptorType::eCombinedImageSampler, imageInfo };
}

void vks::descriptor::Writer::setUniformBuffer(vk::DescriptorSet handle, uint32_t binding, vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize range) {
    const auto& writeIndex = writeIndices[binding];
    auto& write = writes[writeIndex];
    const auto& bindingIndex = writeSourceIndices[binding];
    auto& bufferInfo = bufferInfos[bindingIndex];
    bufferInfo = vk::DescriptorBufferInfo{ buffer, offset, range };
    write = vk::WriteDescriptorSet{ handle, binding, 0, vk::DescriptorType::eUniformBuffer, nullptr, bufferInfo };
}

bool vks::descriptor::Writer::valid() const {
    for (const auto& write : writes) {
        if (!write.dstSet) {
            return false;
        }
    }
    return true;
}