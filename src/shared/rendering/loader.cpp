#include "loader.hpp"
#include "context.hpp"

namespace vks {

void Loader::init() {
}

void Loader::lock(bool lock) {
    this->locked = lock;
}

// Fixed sub resource on first mip level and layer
void Loader::setImageLayout(const vks::QueueManager& queue,
                            vk::Image image,
                            const vk::ImageSubresourceRange& range,
                            vk::ImageLayout oldImageLayout,
                            vk::ImageLayout newImageLayout) const {
    withPrimaryCommandBuffer(queue, [&](const vk::CommandBuffer& copyCmd) {
        vks::util::ImageTransitionState srcState{
            oldImageLayout,
            vks::util::accessFlagsForLayout(oldImageLayout),
            vks::util::pipelineStageForLayout(oldImageLayout),
        };
        vks::util::ImageTransitionState dstState{
            newImageLayout,
            vks::util::accessFlagsForLayout(newImageLayout),
            vks::util::pipelineStageForLayout(newImageLayout),
        };
        vks::util::setImageLayout(copyCmd, image, range, srcState, dstState);
    });
}

void Loader::forceToQueueFamily(const vks::QueueManager& queue, const vk::ArrayProxy<const vk::Buffer>& buffers) const {
    std::vector<vk::BufferMemoryBarrier2> barriers;
    for (const auto& buffer : buffers) {
        barriers.emplace_back(vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, buffer);
    }
    forceToQueueFamily(queue, vk::DependencyInfo{ {}, nullptr, barriers });
}

void Loader::forceToQueueFamily(const vks::QueueManager& queue, const vk::DependencyInfo& dependencyInfo) const {
    withPrimaryCommandBuffer(queue, [&](const vk::CommandBuffer& commandBuffer) {
        commandBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandBuffer.pipelineBarrier2(dependencyInfo);
        commandBuffer.end();
    });
}

// Fixed sub resource on first mip level and layer
void Loader::forceSignalSemaphore(const vks::QueueManager& queue, vk::Semaphore semaphore, uint64_t signalValue) const {
    vk::CommandBuffer commandBuffer = queue.createCommandBuffer();
    commandBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    commandBuffer.end();
    vk::SemaphoreSubmitInfo submitInfo{ semaphore, signalValue, vk::PipelineStageFlagBits2::eNone };
    queue.submit2(commandBuffer, nullptr, submitInfo);
    queue.handle.waitIdle();
    queue.device.waitIdle();
    queue.freeCommandBuffer(commandBuffer);
}

void Loader::forceImageLayout(const vks::QueueManager& queue, vk::Image image, const vk::ImageSubresourceRange& range, vk::ImageLayout newImageLayout) const {
    setImageLayout(queue, image, range, vk::ImageLayout::eUndefined, newImageLayout);
}

void Loader::forceImageLayout(const vks::QueueManager& queue, const vks::Image& image, vk::ImageLayout newImageLayout) const {
    forceImageLayout(queue, image.image, image.getWholeRange(), newImageLayout);
}

void Loader::forceImageLayout(const vks::QueueManager& queue, vk::Image image, const vk::ImageAspectFlags& aspect, vk::ImageLayout newImageLayout) const {
    forceImageLayout(queue, image, vk::ImageSubresourceRange{ aspect, 0, 1, 0, 1 }, newImageLayout);
}

// Image Loader::stageToDeviceImage(const vks::QueueManager& queue,
//                                  vks::Image::Builder buildInfo,
//                                  vk::DeviceSize size,
//                                  const void* data,
//                                  const std::vector<MipData>& mipData,
//                                  vk::ImageLayout layout) const {
//     buildInfo.imageCreateInfo.usage |= vk::ImageUsageFlagBits::eTransferDst;
//     Image result = buildInfo.build();
//     stageToDeviceImage(queue, result, size, data, mipData, layout);
//     return result;
// }

void Loader::stageToDeviceImage(const vks::QueueManager& queue,
                                vks::Image& image,
                                vk::DeviceSize size,
                                const void* data,
                                const std::vector<MipData>& mipData,
                                vk::ImageLayout layout) const {
    const auto& imageCreateInfo = image.createInfo;
    Buffer staging = createStagingBuffer(size, data);
    withPrimaryCommandBuffer(queue, [&](const vk::CommandBuffer& copyCmd) {
        vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, imageCreateInfo.mipLevels, 0, imageCreateInfo.arrayLayers);
        // Prepare for transfer
        vks::util::setImageLayout(copyCmd, image.image, image.getWholeRange(), vks::util::ImageTransitionState::UNDEFINED,
                                  vks::util::ImageTransitionState::TRANSFER_DST);
        // Prepare for transfer
        std::vector<vk::BufferImageCopy> bufferCopyRegions;
        {
            vk::BufferImageCopy bufferCopyRegion;
            bufferCopyRegion.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            bufferCopyRegion.imageSubresource.layerCount = 1;
            if (!mipData.empty()) {
                for (uint32_t i = 0; i < imageCreateInfo.mipLevels; i++) {
                    bufferCopyRegion.imageSubresource.mipLevel = i;
                    bufferCopyRegion.imageExtent = mipData[i].first;
                    bufferCopyRegions.push_back(bufferCopyRegion);
                    bufferCopyRegion.bufferOffset += mipData[i].second;
                }
            } else {
                bufferCopyRegion.imageExtent = imageCreateInfo.extent;
                bufferCopyRegions.push_back(bufferCopyRegion);
            }
        }
        copyCmd.copyBufferToImage(staging.buffer, image.image, vk::ImageLayout::eTransferDstOptimal, bufferCopyRegions);
        // Prepare for future use
        if (layout != vk::ImageLayout::eUndefined) {
            vks::util::setImageLayout(copyCmd, image.image, range, vks::util::ImageTransitionState::TRANSFER_DST, vks::util::ImageTransitionState{ layout });
        }
    });
    staging.destroy();
}

Buffer Loader::createStagingBuffer(vk::DeviceSize size, const void* data) const {
    Buffer result;
#ifdef USE_VMA
    Buffer::Builder builder{ size };
    builder.withAllocCreateFlags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
    builder.withBufferUsage(vk::BufferUsageFlagBits::eTransferSrc);
    result.create(builder);
#else
    result.create(size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
#endif
    if (data != nullptr) {
        result.copy(size, data, 0);
    }
    return result;
}

Buffer Loader::createDeviceBuffer(vk::DeviceSize size, const vk::BufferUsageFlags& usageFlags) const {
    return Buffer::Builder{ size }.withBufferUsage(usageFlags).build();
}

Buffer Loader::createSizedUniformBuffer(vk::DeviceSize size, const void* data) const {
    static const auto memoryFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    static const auto usageFlags = vk::BufferUsageFlagBits::eUniformBuffer;
    static const auto allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    auto result = Buffer::Builder{ size }.withBufferUsage(usageFlags).withAllocCreateFlags(allocationFlags).build();
    if (data != nullptr) {
        result.copy(size, data, 0);
    }
    return result;
}

Buffer Loader::createSizedUniformBuffer(vk::DeviceSize size, vk::DeviceSize stride, const void* data) const {
    static const auto memoryFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    static const auto usageFlags = vk::BufferUsageFlagBits::eUniformBuffer;
    static const auto allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    const auto deviceStride = context.deviceInfo.properties.getUniformAlignedSize(stride);
    const auto count = size / stride;
    const auto alignedSize = deviceStride * count;

    auto result = Buffer::Builder{ alignedSize }.withBufferUsage(usageFlags).withAllocCreateFlags(allocationFlags).build();
    if (data != nullptr) {
        const uint8_t* current = static_cast<const uint8_t*>(data);
        const uint8_t* end = current + size;
        vk::DeviceSize offset = 0;
        result.map();
        while (current + stride <= end) {
            result.copy(stride, current, offset);
            current += stride;
            offset += deviceStride;
        }
    }
    return result;
}

Buffer Loader::stageToDeviceBuffer(const vks::QueueManager& queue, const vk::BufferUsageFlags& usage, size_t size, const void* data) const {
    Buffer staging = createStagingBuffer(size, data);
    Buffer result = createDeviceBuffer(size, usage | vk::BufferUsageFlagBits::eTransferDst);
    withPrimaryCommandBuffer(queue, [&](vk::CommandBuffer copyCmd) { copyCmd.copyBuffer(staging.buffer, result.buffer, vk::BufferCopy(0, 0, size)); });
    staging.destroy();
    return result;
}

/*
 * Create a short lived command buffer which is immediately submitted and freed after it's commands have been executed.
 *
 *  This function is intended for initialization only.  It incurs a graphicsQueue and device flush and WILL impact performance if used in non-setup code
 */
void Loader::withPrimaryCommandBuffer(const QueueManager& queue, const std::function<void(const vk::CommandBuffer& commandBuffer)>& f) const {
    assert(!locked);
    if (locked) {
        throw std::runtime_error("Attempted to use loader in locked state. You must unlock the loader before using it again.");
    }
    vk::CommandBuffer commandBuffer = queue.createCommandBuffer();
    commandBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    f(commandBuffer);
    commandBuffer.end();
    queue.submit2(commandBuffer, vk::Fence{});
    queue.handle.waitIdle();
    queue.device.waitIdle();
    queue.freeCommandBuffer(commandBuffer);
}

void Loader::withPrimaryCommandBuffer(const std::function<void(const vk::CommandBuffer& commandBuffer)>& f) const {
    const auto& context = vks::Context::get();
    vks::QueueManager queue{ context.device, context.queuesInfo.graphics };
    withPrimaryCommandBuffer(queue, f);
    queue.destroy();
}
}  // namespace vks

#if 0
//const vks::QueueManager& Loader::getQueueForFamily(uint32_t queueFamily) const {
//    if (queueFamily == graphicsQueue.familyInfo.index) {
//        return graphicsQueue;
//    } else if (queueFamily == computeQueue.familyInfo.index) {
//        return computeQueue;
//    } else if (queueFamily == transferQueue.familyInfo.index) {
//        return transferQueue;
//    }
//    throw std::runtime_error("Unknown queue requested");
//}

//// Fixed sub resource on first mip level and layer
//void Loader::forceToQueueFamily(uint32_t queueFamily, const vk::ArrayProxy<const vk::Buffer>& buffers) const {
//    forceToQueueFamily(getQueueForFamily(queueFamily), buffers);
//}
//
void Loader::copyToMemory(Allocation& allocation, const void* data, vk::DeviceSize size, vk::DeviceSize offset) const {
    allocation.map();
    allocation.copy(size, data, offset);
    allocation.unmap();
}

void Loader::withPrimaryCommandBuffer(const std::function<void(const vk::CommandBuffer& commandBuffer)>& f) const {
    init::withPrimaryCommandBuffer(graphicsQueue, f);
}

void Loader::transferToQueue(const vks::QueueManager& destQueue, const vks::Image& image, vk::ImageLayout targetLayout) {
    transferToQueue(graphicsQueue, destQueue, image, targetLayout);
}

void Loader::transferToQueue(const vks::QueueManager& destQueue, const vks::Buffer& buffer) {
    transferToQueue(graphicsQueue, destQueue, buffer);
}

void Loader::injectBarriers(const vks::QueueManager& sourceQueue, const vks::QueueManager& destQueue, const vk::DependencyInfo& dependencyInfo) {
    const std::function<void(const vk::CommandBuffer&)> fn = [&](const vk::CommandBuffer& cmdBuffer) { cmdBuffer.pipelineBarrier2(dependencyInfo); };
    vks::init::withPrimaryCommandBuffer(sourceQueue, fn);
    vks::init::withPrimaryCommandBuffer(destQueue, fn);
}

void Loader::transferToQueue(const vks::QueueManager& sourceQueue, const vks::QueueManager& destQueue, const vks::Image& image, vk::ImageLayout targetLayout) {
    vk::ImageMemoryBarrier2 barrier{ // src stage and access
                                     vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                                     // dst stage and access
                                     vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                                     // old and new layout
                                     vk::ImageLayout::eUndefined, targetLayout,
                                     // src and dst graphicsQueue indices
                                     sourceQueue.familyInfo.index, destQueue.familyInfo.index,
                                     // buffer, offset and size
                                     image.image, image.getWholeRange()
    };

    injectBarriers(sourceQueue, destQueue, vk::DependencyInfo{ {}, nullptr, nullptr, barrier });
}

void Loader::transferToQueue(const vks::QueueManager& sourceQueue, const vks::QueueManager& destQueue, const vks::Buffer& buffer) {
    vk::BufferMemoryBarrier2 barrier{ // src stage and access
                                      vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                                      // dst stage and access
                                      vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                                      // src and dst graphicsQueue indices
                                      sourceQueue.familyInfo.index, destQueue.familyInfo.index,
                                      // buffer, offset and size
                                      buffer.buffer, 0, VK_WHOLE_SIZE
    };
    injectBarriers(sourceQueue, destQueue, vk::DependencyInfo{ {}, nullptr, barrier, nullptr });
}


    //vk::DeviceSize uniformAlignment{ 0 };
    //void init(const vks::QueueManager& graphics, vk::DeviceSize uniformAlignment);
    // vk::ImageLayout finalLayout = vk::ImageLayout::eReadOnlyOptimal) const;
    //Image stageToDeviceImage(vk::ImageCreateInfo imageCreateInfo,
    //                         const vk::MemoryPropertyFlags& memoryPropertyFlags,
    //                         vk::DeviceSize size,
    //                         const void* data,
    //                         const std::vector<MipData>& mipData = {}) const;
    //vk::DeviceSize getAlignedSize(vk::DeviceSize size, size_t count = 1) const;
    //void copyToMemory(Allocation& allocation, const void* data, vk::DeviceSize size, vk::DeviceSize offset = 0) const;
    //void transferToQueue(const vks::QueueManager& destQueue, const vks::Image& image, vk::ImageLayout targetLayout);
    //void transferToQueue(const vks::QueueManager& destQueue, const vks::Buffer& buffer);
    //static void transferToQueue(const vks::QueueManager& sourceQueue,
    //                            const vks::QueueManager& destQueue,
    //                            const vks::Image& image,
    //                            vk::ImageLayout targetLayout);
    //static void transferToQueue(const vks::QueueManager& sourceQueue, const vks::QueueManager& destQueue, const vks::Buffer& buffer);
    //static void injectBarriers(const vks::QueueManager& sourceQueue, const vks::QueueManager& destQueue, const vk::DependencyInfo& dependencyInfo);
    //void withPrimaryCommandBuffer(const std::function<void(const vk::CommandBuffer& commandBuffer)>& f) const;
    //vks::QueueManager graphicsQueue;
    //vks::QueueManager computeQueue;
    //vks::QueueManager transferQueue;

    //const vks::QueueManager& getQueueForFamily(uint32_t queueFamily) const;

#endif
