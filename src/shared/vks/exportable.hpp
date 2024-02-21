#pragma once
#include <vulkan/vulkan.hpp>
#include <vks/forward.hpp>
//#include <vks/helpers.hpp>

namespace vks { namespace exportable {

#if defined(WIN32)
constexpr vk::ExternalSemaphoreHandleTypeFlagBits SEMAPHORE_EXPORT_TYPE = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
constexpr vk::ExternalMemoryHandleTypeFlagBits MEMORY_EXPORT_TYPE = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
using ExportType = HANDLE;
#else
#error "FIXME"
#endif

struct Exportable {
    static const ExportType INVALID_VALUE;
    ExportType exportHandle{ INVALID_VALUE };
    static void setup(const vk::Device& device, const vk::PhysicalDeviceMemoryProperties& memoryProperties);
    static ExportType getSemaphoreExport(const vk::Semaphore& semaphore);
    static ExportType getMemoryExport(const vk::DeviceMemory& memory);
};

struct Semaphore : public Exportable {
public:
    vk::Semaphore semaphore;
    void create(vk::SemaphoreCreateInfo createInfo = {});
    //operator const vk::Semaphore&() const { return semaphore; }
    void destroy();
};

struct Texture : public Exportable {
private:
    using Parent = Exportable;

public:
    vk::ImageCreateInfo createInfo;
    vk::MemoryRequirements memoryRequirements;

    vk::Image image;
    vk::DeviceMemory memory;
    vk::ImageView view;
    ExportType exportHandle;
    vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

    void create(const vk::ImageCreateInfo& imageCreateInfo, vk::ImageViewType viewType = vk::ImageViewType::e2D);
    // operator const vk::Image&() const { return image; }
    void destroy();
};

}}  // namespace vks::exportable