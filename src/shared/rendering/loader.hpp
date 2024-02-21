#pragma once
#include <vulkan/vulkan.hpp>
#include <functional>
#include <vks/image.hpp>
#include <vks/buffer.hpp>
#include "context.hpp"

namespace vks {

using MipData = ::std::pair<::vk::Extent3D, ::vk::DeviceSize>;

/*
* A brute-force loader intended only for example purposes.
*/
struct Loader {
private:
    Loader() = default;
    Loader(const Loader&) = delete;
    // Fixed sub resource on first mip level and layer
    void setImageLayout(const vks::QueueManager& queue,
                        vk::Image image,
                        const vk::ImageSubresourceRange& range,
                        vk::ImageLayout oldImageLayout,
                        vk::ImageLayout newImageLayout) const;
    const Context& context = vks::Context::get();
    const vk::Device& device = context.device;
    // The loader should be locked after initialization is done, so that any commands that would incur a waitIdle will trigger an assert;
    bool locked{ false };
    void init();

public:
    static Loader& get() {
        static Loader instance;
        static std::once_flag onceFlag;
        std::call_once(onceFlag, [&]() { instance.init(); });
        return instance;
    }

    void lock(bool lock = true);

    //void forceToQueueFamily(uint32_t queueFamily, const vk::ArrayProxy<const vk::Buffer>& buffers) const;
    void forceToQueueFamily(const vks::QueueManager& queue, const vk::ArrayProxy<const vk::Buffer>& buffers) const;
    void forceToQueueFamily(const vks::QueueManager& queue, const vk::DependencyInfo& dependencyInfo) const;
    void forceSignalSemaphore(const vks::QueueManager& queue, vk::Semaphore semaphore, uint64_t timelineValue = 0) const;
    void forceImageLayout(const vks::QueueManager& queue, vk::Image image, const vk::ImageAspectFlags& aspect, vk::ImageLayout newImageLayout) const;
    void forceImageLayout(const vks::QueueManager& queue, vk::Image image, const vk::ImageSubresourceRange& range, vk::ImageLayout newImageLayout) const;
    void forceImageLayout(const vks::QueueManager& queue, const vks::Image& image, vk::ImageLayout newImageLayout) const;

    void stageToDeviceImage(const vks::QueueManager& queue,
                            vks::Image& image,
                            vk::DeviceSize size,
                            const void* data,
                            const std::vector<MipData>& mipData,
                            vk::ImageLayout finalLayout) const;

    //Image stageToDeviceImage(const vks::QueueManager& queue,
    //                         vks::Image::Builder buildInfo,
    //                         vk::DeviceSize size,
    //                         const void* data,
    //                         const std::vector<MipData>& mipData,
    //                         vk::ImageLayout finalLayout) const;

    Buffer createStagingBuffer(vk::DeviceSize size, const void* data = nullptr) const;
    Buffer createDeviceBuffer(vk::DeviceSize size, const vk::BufferUsageFlags& usageFlags) const;
    Buffer createSizedUniformBuffer(vk::DeviceSize size, const void* data = nullptr) const;
    Buffer createSizedUniformBuffer(vk::DeviceSize size, vk::DeviceSize stride, const void* data = nullptr) const;

    /*
    * Create a short lived command buffer which is immediately submitted and freed after it's commands have been executed.
    *
    *  This function is intended for initialization only.  It incurs a graphicsQueue and device flush and WILL impact performance if used in non-setup code
    */
    void withPrimaryCommandBuffer(const QueueManager& queue, const std::function<void(const vk::CommandBuffer& commandBuffer)>& f) const;
    void withPrimaryCommandBuffer(const std::function<void(const vk::CommandBuffer& commandBuffer)>& f) const;

    template <typename T>
    Image stageToDeviceImage(const vks::QueueManager& queue,
                             const vk::ImageCreateInfo& imageCreateInfo,
                             const vk::MemoryPropertyFlags& memoryPropertyFlags,
                             const std::vector<T>& data) const {
        return stageToDeviceImage(imageCreateInfo, memoryPropertyFlags, data.size() * sizeof(T), (void*)data.data());
    }

    template <typename T>
    Image stageToDeviceImage(const vks::QueueManager& queue, const vk::ImageCreateInfo& imageCreateInfo, const std::vector<T>& data) const {
        return stageToDeviceImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal, data.size() * sizeof(T), (void*)data.data());
    }

    template <typename T>
    Buffer createStagingBuffer(const std::vector<T>& data) const {
        return createStagingBuffer(data.size() * sizeof(T), (void*)data.data());
    }

    template <typename T>
    Buffer createStagingBuffer(const T& data) const {
        return createStagingBuffer(sizeof(T), &data);
    }

    template <typename T>
    Buffer createUniformBuffer(const T& data) const {
        return createSizedUniformBuffer(sizeof(T), &data);
    }

    template <typename T>
    Buffer createUniformBuffer(const vk::ArrayProxy<T>& data) const {
        return createSizedUniformBuffer(sizeof(T) * data.size(), sizeof(T), &data);
    }

    //template <typename T>
    //void copyToMemory(Allocation& allocation, const T& data, size_t offset = 0) const {
    //    copyToMemory(allocation, &data, sizeof(T), offset);
    //}

    //template <typename T>
    //void copyToMemory(Allocation& allocation, const vk::ArrayProxy<T>& data, size_t offset = 0) const {
    //    copyToMemory(allocation, data.data(), data.size() * sizeof(T), offset);
    //}

    Buffer stageToDeviceBuffer(const vks::QueueManager& queue, const vk::BufferUsageFlags& usage, size_t size, const void* data) const;

    template <typename T>
    Buffer stageToDeviceBuffer(const vks::QueueManager& queue, const vk::BufferUsageFlags& usage, const std::vector<T>& data) const {
        return stageToDeviceBuffer(queue, usage, sizeof(T) * data.size(), data.data());
    }

    template <typename T>
    Buffer stageToDeviceBuffer(const vks::QueueManager& queue, const vk::BufferUsageFlags& usage, const T& data) const {
        return stageToDeviceBuffer(queue, usage, sizeof(T), (void*)&data);
    }
};

}  // namespace vks
