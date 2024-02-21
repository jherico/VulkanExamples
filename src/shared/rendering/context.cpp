#include "context.hpp"

#include <unordered_map>
#include <vks/allocation.hpp>

vk::PhysicalDevice vks::DeviceInfo::pickDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface, const Predicate& predicate) {
    // Physical device
    auto physicalDevices = instance.enumeratePhysicalDevices();

    // Filter on devices that can present
    if (surface) {
        std::unordered_set<VkPhysicalDevice> eligibleDevices;
        eligibleDevices.reserve(physicalDevices.size());
        for (const auto& physicalDevice : physicalDevices) {
            const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties2();
            for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
                if (queueFamilyProperties[i].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics) {
                    if (physicalDevice.getSurfaceSupportKHR(i, surface)) {
                        eligibleDevices.insert(physicalDevice);
                        break;
                    }
                }
            }
        }
        physicalDevices.clear();
        physicalDevices.insert(physicalDevices.end(), eligibleDevices.begin(), eligibleDevices.end());
    }

    if (predicate) {
        std::unordered_set<VkPhysicalDevice> eligibleDevices;
        eligibleDevices.reserve(physicalDevices.size());
        for (const auto& physicalDevice : physicalDevices) {
            if (predicate(physicalDevice)) {
                eligibleDevices.insert(physicalDevice);
            }
        }
        physicalDevices.clear();
        physicalDevices.insert(physicalDevices.end(), eligibleDevices.begin(), eligibleDevices.end());
    }

    // Pick the device with the most device local memory
    vk::PhysicalDevice bestDevice;
    vk::DeviceSize bestMemory = 0;
    for (const auto& physicalDevice : physicalDevices) {
        const auto memoryProperties = physicalDevice.getMemoryProperties();
        if (memoryProperties.memoryHeaps[0].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            if (!bestDevice || memoryProperties.memoryHeaps[0].size > bestMemory) {
                bestDevice = physicalDevice;
                bestMemory = memoryProperties.memoryHeaps[0].size;
            }
        }
    }
    return bestDevice;
}

vks::SimpleContext& vks::SimpleContext::get() {
    static SimpleContext SimpleContext;
    instanceInit();
    return SimpleContext;
}

vks::StringList vks::SimpleContext::filterLayers(const StringList& desiredLayers) {
    static auto validLayerNames = getAvailableLayers();
    StringList result;
    for (const auto& string : desiredLayers) {
        if (validLayerNames.count(string) != 0) {
            result.emplace_back(string);
        }
    }
    return result;
}

void vks::SimpleContext::instanceInit() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] { VULKAN_HPP_DEFAULT_DISPATCHER.init(&vkGetInstanceProcAddr); });
}

// Create application wide Vulkan instance
const vks::LayerMap& vks::SimpleContext::getAvailableLayers() {
    instanceInit();
    static std::once_flag onceFlag;
    static std::unordered_map<std::string, const vk::LayerProperties> layers;
    std::call_once(onceFlag, [&] {
        for (auto layer : vk::enumerateInstanceLayerProperties()) {
            std::string layerName = layer.layerName;
            layers.emplace(layerName, layer);
        }
    });
    return layers;
}

const vks::ExtensionMap& vks::SimpleContext::getExtensions() {
    instanceInit();
    static std::once_flag onceFlag;
    static std::unordered_map<std::string, const vk::ExtensionProperties> extensions;
    std::call_once(onceFlag, [&] {
        for (const auto extension : vk::enumerateInstanceExtensionProperties()) {
            std::string extName = extension.extensionName;
            extensions.emplace(extName, extension);
        }
    });
    return extensions;
}

bool vks::SimpleContext::isExtensionPresent(const std::string& extensionName) {
    return getExtensionNames().count(extensionName) != 0;
}

std::unordered_set<std::string> vks::SimpleContext::getExtensionNames() {
    std::unordered_set<std::string> extensionNames;
    for (auto& entry : getExtensions()) {
        extensionNames.insert(entry.first);
    }
    return extensionNames;
}

void vks::SimpleContext::createInstance(uint32_t version) {
    instanceInit();
    getExtensions();
    getAvailableLayers();

    if (enableValidation) {
        if (isExtensionPresent(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            requireExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            enableValidation = false;
        }
    }

    // Vulkan instance
    vk::ApplicationInfo appInfo;
    appInfo.pApplicationName = "VulkanExamples";
    appInfo.pEngineName = "VulkanExamples";
    appInfo.apiVersion = version;

    auto enabledExtensions = vks::util::toCStrings(requiredExtensions);

    vk::InstanceCreateInfo instanceCreateInfo{ {}, &appInfo };
    using vks::debug::MessageSeverityBits;
    if (enableValidation) {
        vks::debug::setupDebugging(instanceCreateInfo, MessageSeverityBits::eError | MessageSeverityBits::eWarning);
    }

    if (!enabledExtensions.empty()) {
        instanceCreateInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
    }

    StringList strLayers = filterLayers(debug::validationLayerNames);
    CStringVector layers = vks::util::toCStrings(strLayers);
    if (enableValidation) {
        instanceCreateInfo.enabledLayerCount = (uint32_t)layers.size();
        instanceCreateInfo.ppEnabledLayerNames = layers.data();
    }

    // vk::ValidationFeatureEnableEXT enables[] = { vk::ValidationFeatureEnableEXT::eSynchronizationValidation };
    // vk::ValidationFeatureDisableEXT disables[4] = { vk::ValidationFeatureDisableEXT::eThreadSafety, vk::ValidationFeatureDisableEXT::eApiParameters,
    //                                                 vk::ValidationFeatureDisableEXT::eObjectLifetimes, vk::ValidationFeatureDisableEXT::eCoreChecks };
    // vk::ValidationFeaturesEXT features{ 1, enables, 4, disables };
    // injectNext(instanceCreateInfo, features);

    instance = vk::createInstance(instanceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    if (enableValidation) {
        debug::setupDebugging(instance, MessageSeverityBits::eError | MessageSeverityBits::eWarning);
    }
}

void vks::SimpleContext::pickDevice(const vk::SurfaceKHR& surface, const vks::DeviceInfo::Predicate& predicate) {
    assert(instance);
    assert(!physicalDevice);
    // Pick the best device (one with support for the specified surface, and which matches any predicates necessary).
    physicalDevice = vks::DeviceInfo::pickDevice(instance, surface, predicate);
    deviceInfo = vks::DeviceInfo{ physicalDevice };
    queuesInfo = vks::QueuesInfo{ physicalDevice };
}

vk::Device vks::SimpleContext::createDevice() {
    assert(!device);
    // Always request the dedicated allocation if available.
    if (deviceInfo.hasExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME)) {
        requiredDeviceExtensions.insert(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
    // enable the debug marker extension if it is present (likely meaning a debugging tool is present)
    if (deviceInfo.hasExtension(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)) {
        requiredDeviceExtensions.insert(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
        enableDebugMarkers = true;
    }
    if (deviceInfo.hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        requiredDeviceExtensions.insert(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
    if (deviceInfo.hasExtension(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) {
        requiredDeviceExtensions.insert(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    }

    // Turn on the desired features (should be set before calling createDevice
    {
        vk::DeviceCreateInfo createInfo;
        vk::PhysicalDeviceFeatures2 enabledFeatures2 = enabledFeatures.getFeatures2(deviceInfo.extensions);
        createInfo.pNext = &enabledFeatures2;
        std::vector<vk::DeviceQueueCreateInfo> deviceQueues;
        float priority = 1.0f;

        // Create one graphicsQueue instance of each of the available graphicsQueue types
        deviceQueues.push_back({ {}, queuesInfo.graphics.index, 1, &priority });
        if (queuesInfo.compute) {
            deviceQueues.push_back({ {}, queuesInfo.compute.index, 1, &priority });
        }
        if (queuesInfo.transfer) {
            deviceQueues.push_back({ {}, queuesInfo.transfer.index, 1, &priority });
        }
        createInfo.pQueueCreateInfos = &(deviceQueues[0]);
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(deviceQueues.size());

        CStringVector enabledExtensions;
        if (!requiredDeviceExtensions.empty()) {
            enabledExtensions = vks::util::toCStrings(requiredDeviceExtensions);
            createInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
            createInfo.ppEnabledExtensionNames = enabledExtensions.data();
        }
        // Vulkan device
        device = physicalDevice.createDevice(createInfo);
    }

    // Update the default dispatcher with the device
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance, device);
    Allocation::init(instance, deviceInfo, physicalDevice, device);

    if (enableDebugMarkers) {
        debug::marker::setup(instance, device);
    }
    return device;
}

#if 0
//// Add all the requested extensions to the device creation info
//allDeviceExtensions.insert(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end());
//for (const auto& extension : desiredDeviceExtensions) {
//    if (hasExtension(extension)) {
//        allDeviceExtensions.insert(extension);
//    }
//}

//std::vector<const char*> enabledExtensions;
//for (const auto& extension : allDeviceExtensions) {
//    enabledExtensions.push_back(extension.c_str());
//}

// Create an image memory barrier for changing the layout of
// an image and put it into an active command buffer
// See chapter 11.4 "vk::Image Layout" for details

//vk::ImageMemoryBarrier createLayoutBarrier(vk::Image image,
//                                           vk::ImageLayout oldImageLayout,
//                                           vk::ImageLayout newImageLayout,
//                                           vk::ImageSubresourceRange subresourceRange) const {
//    // Create an image barrier object
//    vk::ImageMemoryBarrier imageMemoryBarrier;
//    imageMemoryBarrier.oldLayout = oldImageLayout;
//    imageMemoryBarrier.newLayout = newImageLayout;
//    imageMemoryBarrier.image = image;
//    imageMemoryBarrier.subresourceRange = subresourceRange;
//    imageMemoryBarrier.srcAccessMask = vks::util::accessFlagsForLayout(oldImageLayout);
//    imageMemoryBarrier.dstAccessMask = vks::util::accessFlagsForLayout(newImageLayout);
//    return imageMemoryBarrier;
//}

//// Fixed sub resource on first mip level and layer
//void setImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout, vk::ImageLayout newImageLayout) const {
//    setImageLayout(cmdbuffer, image, vk::ImageAspectFlagBits::eColor, oldImageLayout, newImageLayout);
//}

//// Fixed sub resource on first mip level and layer
//void setImageLayout(vk::CommandBuffer cmdbuffer,
//                    vk::Image image,
//                    vk::ImageAspectFlags aspectMask,
//                    vk::ImageLayout oldImageLayout,
//                    vk::ImageLayout newImageLayout) const {
//    vk::ImageSubresourceRange subresourceRange;
//    subresourceRange.aspectMask = aspectMask;
//    subresourceRange.levelCount = 1;
//    subresourceRange.layerCount = 1;
//    setImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange);
//}

//void setImageLayout(vk::Image image, vk::ImageLayout oldImageLayout, vk::ImageLayout newImageLayout, vk::ImageSubresourceRange subresourceRange) const {
//    withPrimaryCommandBuffer([&](const auto& commandBuffer) { setImageLayout(commandBuffer, image, oldImageLayout, newImageLayout, subresourceRange); });
//}

//// Fixed sub resource on first mip level and layer
//void setImageLayout(vk::Image image, vk::ImageAspectFlags aspectMask, vk::ImageLayout oldImageLayout, vk::ImageLayout newImageLayout) const {
//    withPrimaryCommandBuffer([&](const auto& commandBuffer) { setImageLayout(commandBuffer, image, aspectMask, oldImageLayout, newImageLayout); });
//}
//vk::CommandPool getCommandPool() const {
//    if (!s_cmdPool) {
//        vk::CommandPoolCreateInfo cmdPoolInfo;
//        cmdPoolInfo.queueFamilyIndex = queueIndices.graphics;
//        cmdPoolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
//        s_cmdPool = device.createCommandPool(cmdPoolInfo);
//    }
//    return s_cmdPool;
//}

//void destroyCommandPool() const {
//    if (s_cmdPool) {
//        device.destroy(s_cmdPool);
//        s_cmdPool = vk::CommandPool();
//    }
//}
//std::vector<vk::CommandBuffer> allocateCommandBuffers(uint32_t count, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const {
//    std::vector<vk::CommandBuffer> result;
//    vk::CommandBufferAllocateInfo commandBufferAllocateInfo;
//    commandBufferAllocateInfo.commandPool = getCommandPool();
//    commandBufferAllocateInfo.commandBufferCount = count;
//    commandBufferAllocateInfo.level = vk::CommandBufferLevel::ePrimary;
//    result = device.allocateCommandBuffers(commandBufferAllocateInfo);
//    return result;
//}

//vk::CommandBuffer createCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const { return allocateCommandBuffers(1, level)[0]; }
//static std::vector<vk::ExtensionProperties> getDeviceExtensions(const vk::PhysicalDevice& physicalDevice) {
//    return physicalDevice.enumerateDeviceExtensionProperties();
//}
//static std::set<std::string> getDeviceExtensionNames(const vk::PhysicalDevice& physicalDevice) {
//    std::set<std::string> extensionNames;
//    for (auto& ext : getDeviceExtensions(physicalDevice)) {
//        extensionNames.insert(ext.extensionName);
//    }
//    return extensionNames;
//}
//static bool isDeviceExtensionPresent(const vk::PhysicalDevice& physicalDevice, const std::string& extension) {
//    return getDeviceExtensionNames(physicalDevice).count(extension) != 0;
//}
//void addInstanceExtensionPicker(const InstanceExtensionsPickerFunction& function) { instanceExtensionsPickers.push_back(function); }
//void setDevicePicker(const DevicePickerFunction& picker) { devicePicker = picker; }
//void setDeviceFeaturesPicker(const DeviceFeaturesPickerFunction& picker) { deviceFeaturesPicker = picker; }
//void setDeviceExtensionsPicker(const DeviceExtensionsPickerFunction& picker) { deviceExtensionsPicker = picker; }

//#if 0
//struct DevicePicker {
//    std::unordered_set<std::string> requiredDeviceExtensions;
//
//    void pickDevice(const std::vector<PhysicalDevice>& physicalDevices, const ::vk::SurfaceKHR& surface);
//
//    void requireDeviceExtensions(const vk::ArrayProxy<const std::string>& requestedExtensions) {
//        requiredDeviceExtensions.insert(requestedExtensions.begin(), requestedExtensions.end());
//    }
//
//    DevicePickerFunction devicePicker = [](const std::vector<vk::PhysicalDevice>& devices) -> vk::PhysicalDevice { return devices[0]; };
//};
//#endif
#endif

#if 0
namespace queues {

struct DeviceCreateInfo : public ::vk::DeviceCreateInfo {
    std::vector<::vk::DeviceQueueCreateInfo> deviceQueues;
    std::vector<std::vector<float>> deviceQueuesPriorities;

    void addQueueFamily(uint32_t queueFamilyIndex, ::vk::ArrayProxy<float> priorities) {
        deviceQueues.push_back({ {}, queueFamilyIndex });
        std::vector<float> prioritiesVector;
        prioritiesVector.resize(priorities.size());
        memcpy(prioritiesVector.data(), priorities.data(), sizeof(float) * priorities.size());
        deviceQueuesPriorities.push_back(prioritiesVector);
    }

    void addQueueFamily(uint32_t queueFamilyIndex, size_t count = 1) {
        std::vector<float> priorities;
        priorities.resize(count);
        std::fill(priorities.begin(), priorities.end(), 0.0f);
        addQueueFamily(queueFamilyIndex, priorities);
    }

    void update() {
        assert(deviceQueuesPriorities.size() == deviceQueues.size());
        auto size = deviceQueues.size();
        for (auto i = 0; i < size; ++i) {
            auto& deviceQueue = deviceQueues[i];
            auto& deviceQueuePriorities = deviceQueuesPriorities[i];
            deviceQueue.queueCount = static_cast<uint32_t>(deviceQueuePriorities.size());
            deviceQueue.pQueuePriorities = deviceQueuePriorities.data();
        }

        this->queueCreateInfoCount = static_cast<uint32_t>(deviceQueues.size());
        this->pQueueCreateInfos = deviceQueues.data();
    }
};
}  // namespace queues
#endif

#if 0
#if defined(__ANDROID__)
requireExtension(VK_KHR_SURFACE_EXTENSION_NAME);
requireExtension(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#else
requireExtensions(glfw::getRequiredInstanceExtensions());
#endif

requireDeviceExtensions({ VK_KHR_SWAPCHAIN_EXTENSION_NAME });



// Load a SPIR-V shader
inline vk::PipelineShaderStageCreateInfo loadShader(const std::string& fileName, vk::ShaderStageFlagBits stage) const {
    vk::PipelineShaderStageCreateInfo shaderStage;
    shaderStage.stage = stage;
    shaderStage.module = vkx::loadShader(fileName, device, stage);
    shaderStage.pName = "main"; // todo : make param
    assert(shaderStage.module);
    shaderModules.push_back(shaderStage.module);
    return shaderStage;
}

#endif
// InstanceExtensionsPickerFunctions instanceExtensionsPickers;
