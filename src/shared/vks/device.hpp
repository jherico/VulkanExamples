#pragma once
#include <vulkan/vulkan.hpp>
#include <functional>
#include <iostream>
#include <vks/helpers.hpp>

namespace vks {

using LayerMap = std::unordered_map<std::string, const ::vk::LayerProperties>;
using ExtensionMap = std::unordered_map<std::string, const ::vk::ExtensionProperties>;

struct DeviceFeatures {
    vk::PhysicalDeviceFeatures core10;
    vk::PhysicalDeviceVulkan11Features core11;
    vk::PhysicalDeviceVulkan12Features core12;
    vk::PhysicalDeviceVulkan13Features core13;
    vk::PhysicalDeviceMemoryPriorityFeaturesEXT memoryPriorityEXT;
    vk::PhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferEXT;
    // VK_EXT_descriptor_buffer
    // VK_EXT_descriptor_buffer
    // VK_KHR_global_priority
    // VK_EXT_attachment_feedback_loop_layout
    // VK_EXT_border_color_swizzle
    // VK_EXT_conditional_rendering
    // VK_EXT_depth_clip_control
    // VK_EXT_multi_draw
    // VK_EXT_vertex_input_dynamic_state
    // VK_KHR_present_wait

    void load(const vk::PhysicalDevice& device, const ExtensionMap& extensions) {
        vk::PhysicalDeviceFeatures2 features2 = getFeatures2(extensions);
        device.getFeatures2(&features2);
        core10 = features2.features;
    }

    vk::PhysicalDeviceFeatures2 getFeatures2(const ExtensionMap& extensions) {
        vk::PhysicalDeviceFeatures2 features2;
        injectNext(features2, core11);
        injectNext(features2, core12);
        injectNext(features2, core13);
        if (extensions.count(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME) != 0) {
            injectNext(features2, memoryPriorityEXT);
        }
        if (extensions.count(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) != 0) {
            injectNext(features2, descriptorBufferEXT);
        }
        features2.features = core10;
        return features2;
    }
};

struct DeviceProperties {
    vk::PhysicalDeviceProperties core10;
    vk::PhysicalDeviceVulkan11Properties core11;
    vk::PhysicalDeviceVulkan12Properties core12;
    vk::PhysicalDeviceVulkan13Properties core13;
    vk::PhysicalDeviceMaintenance4Properties maint4;
    vk::PhysicalDeviceDescriptorBufferPropertiesEXT descriptorBufferEXT;
    vk::PhysicalDevicePushDescriptorPropertiesKHR pushDescriptorsKHR;
    vk::PhysicalDeviceConservativeRasterizationPropertiesEXT conservativeRasterizationEXT;

    static vk::DeviceSize getAlignedSize(vk::DeviceSize alignment, vk::DeviceSize size);

    vk::DeviceSize getUniformAlignedSize(vk::DeviceSize size) const;
    vk::DeviceSize getUniformAlignedOffset(vk::DeviceSize size, size_t count) const;
    vk::DeviceSize getTexelAlignedSize(vk::DeviceSize size) const;
    vk::DeviceSize getTexelAlignedOffset(vk::DeviceSize size, size_t count) const;
    vk::DeviceSize getStorageAlignedSize(vk::DeviceSize size) const;
    vk::DeviceSize getStorageAlignedOffset(vk::DeviceSize size, size_t count) const;

    vk::PhysicalDeviceProperties2 getProperties2(const ExtensionMap& extensions) {
        vk::PhysicalDeviceProperties2 properties2;
        properties2.properties = core10;
        injectNext(properties2, core11);
        injectNext(properties2, core12);
        injectNext(properties2, core13);
        injectNext(properties2, maint4);
        if (extensions.count(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME) != 0) {
            injectNext(properties2, conservativeRasterizationEXT);
        }
        if (extensions.count(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) != 0) {
            injectNext(properties2, descriptorBufferEXT);
        }
        if (extensions.count(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) != 0) {
            injectNext(properties2, pushDescriptorsKHR);
        }
        return properties2;
    }

    void load(const vk::PhysicalDevice& device, const ExtensionMap& extensions) {
        vk::PhysicalDeviceProperties2 properties2 = getProperties2(extensions);
        device.getProperties2(&properties2);
        core10 = properties2.properties;
    }
};

struct DeviceMemoryProperties {
    vk::PhysicalDeviceMemoryProperties core;
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetEXT;

    void load(const vk::PhysicalDevice& device, const ExtensionMap& extensions) {
        vk::PhysicalDeviceMemoryProperties2 properties2;
        if (extensions.count(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) != 0) {
            injectNext(properties2, budgetEXT);
        }
        device.getMemoryProperties2(&properties2);
        core = properties2.memoryProperties;
    }
};

struct DeviceInfo {
    using Predicate = std::function<bool(const vk::PhysicalDevice& physicalDevice)>;

    ExtensionMap extensions;
    DeviceProperties properties;
    DeviceFeatures features;
    DeviceMemoryProperties memoryProperties;
    vk::Format supportedDepthFormat{ vk::Format::eUndefined };

    static ExtensionMap getExtensions(const vk::PhysicalDevice&);
    static vk::PhysicalDevice pickDevice(const vk::Instance& instance, const vk::SurfaceKHR& surface, const Predicate& predicate = nullptr);
    static vk::Format getSupportedDepthFormat(const vk::PhysicalDevice& physicalDevice);

    DeviceInfo() = default;
    DeviceInfo(const vk::PhysicalDevice& physcialDevice);

    bool hasExtension(const std::string& extensionName) const { return extensions.count(extensionName) != 0; }
};
}  // namespace vks