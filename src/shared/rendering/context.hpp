#pragma once

#include <algorithm>
#include <functional>
#include <list>
#include <unordered_set>
#include <set>
#include <string>
#include <vector>
#include <queue>
#include <bit>

//#include <glm/glm.hpp>
//#include <gli/gli.hpp>

#include <vulkan/vulkan.hpp>

#include <vks/forward.hpp>
#include <vks/debug.hpp>
#include <vks/queues.hpp>

//#include <vks/image.hpp>
//#include <vks/buffer.hpp>
#include <vks/helpers.hpp>
#include <vks/device.hpp>

#ifndef NDEBUG
#define TRUE_IF_DEBUG true
#else
#define TRUE_IF_DEBUG false
#endif

namespace vks {

struct SimpleContext {
    static SimpleContext& get();

private:
    static void instanceInit();
    static StringList filterLayers(const StringList& desiredLayers);

    SimpleContext() = default;

public:
    // Create application wide Vulkan instance
    static const LayerMap& getAvailableLayers();
    static const ExtensionMap& getExtensions();
    static StringBag getExtensionNames();
    static bool isExtensionPresent(const std::string& extensionName);

    void requireExtensions(const vk::ArrayProxy<const std::string>& requestedExtensions) {
        requiredExtensions.insert(requestedExtensions.begin(), requestedExtensions.end());
    }
    void requireExtension(const std::string& requestedExtension) { requiredExtensions.insert(requestedExtension); }

    void requireDeviceExtensions(const vk::ArrayProxy<const std::string>& requestedExtensions) {
        requiredDeviceExtensions.insert(requestedExtensions.begin(), requestedExtensions.end());
    }
    void requireDeviceExtension(const std::string& requestedExtension) { requiredDeviceExtensions.insert(requestedExtension); }

    void setValidationEnabled(bool enable) {
        if (instance != vk::Instance()) {
            throw std::runtime_error("Cannot change validations state after instance creation");
        }
        enableValidation = enable;
    }

    void createInstance(uint32_t version = VK_MAKE_VERSION(1, 3, 0));
    void pickDevice(const vk::SurfaceKHR& surface = nullptr, const DeviceInfo::Predicate& predicate = {});
    vk::Device createDevice();

    void destroy() {
        if (enableValidation) {
            debug::freeDebugCallback(instance);
        }
        if (device) {
            device.destroy();
            device = nullptr;
        }
        instance.destroy();
        instance = nullptr;
    }

public:
    // Vulkan instance, stores all per-application states
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::PipelineCache pipelineCache;
    DeviceFeatures enabledFeatures;
    vk::PhysicalDeviceDynamicRenderingFeatures enabledDynamicRenderingFeatures;

    DeviceInfo deviceInfo;
    QueuesInfo queuesInfo;
    StringBag requiredExtensions;
    StringBag requiredDeviceExtensions;

    // Default to true when example is created with enabled validation layers
    bool enableValidation = TRUE_IF_DEBUG;
    // Set to true when the debug marker extension is detected
    bool enableDebugMarkers = false;

private:
};

using Context = SimpleContext;

}  // namespace vks
