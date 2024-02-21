#include "exportable.hpp"
#include <vks/device.hpp>
namespace vks { namespace exportable {

static vk::Device device;
static vk::PhysicalDeviceMemoryProperties memoryProperties;

static uint32_t getMemoryType(uint32_t typeBits, const vk::MemoryPropertyFlags& properties) {
    for (uint32_t i = 0; i < 32; i++) {
        if ((typeBits & 1) == 1) {
            if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        typeBits >>= 1;
    }
    throw std::runtime_error("Unable to find memory type " + vk::to_string(properties));
}

#ifdef WIN32
const ExportType Exportable::INVALID_VALUE = INVALID_HANDLE_VALUE;

ExportType Exportable::getSemaphoreExport(const vk::Semaphore& semaphore) {
    return device.getSemaphoreWin32HandleKHR(vk::SemaphoreGetWin32HandleInfoKHR{ semaphore, SEMAPHORE_EXPORT_TYPE });
}

ExportType Exportable::getMemoryExport(const vk::DeviceMemory& memory) {
    return device.getMemoryWin32HandleKHR(vk::MemoryGetWin32HandleInfoKHR{ memory, MEMORY_EXPORT_TYPE });
}
#else
const ExportType Exportable::INVALID_VALUE = -1;

ExportType Exportable::getSemaphoreExport(const vk::Semaphore& semaphore) {
    return device.getSemaphoreFdKHR(vk::SemaphoreGetFdInfoKHR{ semaphore, SEMAPHORE_EXPORT_TYPE });
}

ExportType Exportable::getMemoryExport(const vk::DeviceMemory& memory) {
    return device.getMemoryFdKHR(vk::MemoryGetFdInfoKHR{ memory, MEMORY_EXPORT_TYPE });
}
#endif

void Exportable::setup(const vk::Device& device, const vk::PhysicalDeviceMemoryProperties& memoryProperties) {
    vks::exportable::device = device;
    vks::exportable::memoryProperties = memoryProperties;
}

void Semaphore::create(vk::SemaphoreCreateInfo createInfo) {
    assert(device);
    vk::ExportSemaphoreCreateInfo esci{ SEMAPHORE_EXPORT_TYPE };
    injectNext(createInfo, esci);

    semaphore = device.createSemaphore(createInfo);
    exportHandle = getSemaphoreExport(semaphore);
}

void Semaphore::destroy() {
    if (semaphore) {
        device.destroy(semaphore);
        semaphore = nullptr;
    }
    exportHandle = INVALID_VALUE;
}

void Texture::create(const vk::ImageCreateInfo& imageCreateInfo, vk::ImageViewType viewType) {
    assert(device);
    createInfo = imageCreateInfo;
    range.layerCount = createInfo.arrayLayers;
    range.levelCount = createInfo.mipLevels;

    vk::ExternalMemoryImageCreateInfo externalImageCreateInfo{ MEMORY_EXPORT_TYPE };
    injectNext(createInfo, externalImageCreateInfo);
    image = device.createImage(createInfo);

    {
        vk::ExportMemoryAllocateInfo exportAllocInfo{ MEMORY_EXPORT_TYPE };
        memoryRequirements = device.getImageMemoryRequirements(image);
        auto memoryTypeIndex = getMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = device.allocateMemory(vk::MemoryAllocateInfo{ memoryRequirements.size, memoryTypeIndex, &exportAllocInfo });
        device.bindImageMemory(image, memory, 0);
        exportHandle = getMemoryExport(memory);
    }

    {
        // Create image view
        vk::ImageViewCreateInfo viewCreateInfo;
        viewCreateInfo.viewType = vk::ImageViewType::e2D;
        viewCreateInfo.image = image;
        viewCreateInfo.format = createInfo.format;
        viewCreateInfo.subresourceRange = range;
        view = device.createImageView(vk::ImageViewCreateInfo{ {}, image, viewType, createInfo.format, {}, range });
    }
}

void Texture::destroy() {
    if (view) {
        device.destroy(view);
        view = nullptr;
    }
    if (memory) {
        device.free(memory);
        memory = nullptr;
    }
    if (image) {
        device.destroy(image);
        image = nullptr;
    }
    exportHandle = INVALID_VALUE;
}

}}  // namespace vks::exportable