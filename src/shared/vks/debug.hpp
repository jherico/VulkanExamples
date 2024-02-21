#pragma once

#include <algorithm>
#include <functional>
#include <list>
#include <string>
#include <iostream>

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

namespace vks {

inline void inspectPNextChainImpl(const vk::BaseInStructure* s) {
    while (s) {
        auto pNext = s->pNext;
        auto sType = s->sType;
        std::cout << "pNext: " << pNext << " " << vk::to_string(sType) << "" << std::endl;
        s = ((const vk::BaseInStructure*)pNext);
    }
}

template <typename T>
void inspectPNextChain(const T& t) {
    inspectPNextChainImpl(&reinterpret_cast<const vk::BaseInStructure&>(t));
}

namespace debug {

using MessageSeverity = vk::DebugUtilsMessageSeverityFlagsEXT;
using MessageSeverityBits = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using MessageSeverityTraits = vk::FlagTraits<MessageSeverityBits>;

using MessageType = vk::DebugUtilsMessageTypeFlagsEXT;
using MessageTypeBits = vk::DebugUtilsMessageTypeFlagBitsEXT;
using MessageTypeTraits = vk::FlagTraits<MessageTypeBits>;

using MessageCallbackData = vk::DebugUtilsMessengerCallbackDataEXT;

// Default validation layers
extern std::list<std::string> validationLayerNames;
using FlagTraits = vk::FlagTraits<vk::DebugReportFlagBitsEXT>;

using MessageHandler = std::function<VkBool32(const MessageSeverityBits&, const MessageType&, const MessageCallbackData&, void*)>;

void setupDebugging(vk::InstanceCreateInfo& instanceCreateInfo,
                    const MessageSeverity& severityFlags = MessageSeverityTraits::allFlags,
                    const MessageType& typeFlags = MessageTypeTraits::allFlags);
void setupDebugging(const vk::Instance& instance,
                    const MessageSeverity& severityFlags = MessageSeverityTraits::allFlags,
                    const MessageType& typeFlags = MessageTypeTraits::allFlags);

// Clear debug callback
void freeDebugCallback(const vk::Instance& instance);

void report(const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkCommandBuffer& cmdBuffer, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkQueue& queue, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkImage& image, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkSampler& sampler, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkBuffer& buffer, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkDeviceMemory& memory, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkShaderModule& shaderModule, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkPipeline& pipeline, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkPipelineLayout& pipelineLayout, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkRenderPass& renderPass, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkFramebuffer& framebuffer, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkDescriptorSetLayout& descriptorSetLayout, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkDescriptorSet& descriptorSet, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkSemaphore& semaphore, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkFence& fence, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);
void report(const VkEvent& _event, const char* message, const vk::DebugReportFlagsEXT& flags = FlagTraits::allFlags);

// Setup and functions for the VK_EXT_debug_marker_extension
// Extension spec can be found at https://github.com/KhronosGroup/Vulkan-Docs/blob/1.0-VK_EXT_debug_marker/doc/specs/vulkan/appendices/VK_EXT_debug_marker.txt
// Note that the extension will only be present if run from an offline debugging application
// The actual check for extension presence and enabling it on the device is done in the example base class
// See ExampleBase::createInstance and ExampleBase::createDevice (base/vkx::ExampleBase.cpp)

namespace marker {
// Set to true if function pointer for the debug marker are available
extern bool active;

// Get function pointers for the debug report extensions from the device
void setup(const vk::Instance& instance, const vk::Device& device);

// Sets the label of an object
template <typename T>
void setObjectName(const vk::Device& device, const T& object, const char* name) {
    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{ T::objectType, reinterpret_cast<const uint64_t&>(object), name });
}

// Sets the label of an object
template <typename T>
void setObjectName(const vk::Device& device, const T& object, const std::string& name) {
    setObjectName(device, object, name.c_str());
}

// Set the tag for an object
template <typename T, typename TAG>
void setObjectTag(const vk::Device& device, const T& object, const TAG& tag, uint64_t tagName = 0) {
    device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectTagInfoEXT{ T::objectType, reinterpret_cast<const uint64_t&>(object), tagName, sizeof(TAG), &tag });
}

// Start a new debug marker region
void beginRegion(const vk::CommandBuffer& cmdbuffer, const std::string& pMarkerName, const glm::vec4& color);

// Insert a new debug marker into the command buffer
void insert(const vk::CommandBuffer& cmdbuffer, const std::string& markerName, const glm::vec4& color);

// End the current debug marker region
void endRegion(const vk::CommandBuffer& cmdBuffer);

class Marker {
public:
    Marker(const vk::CommandBuffer& cmdBuffer, const std::string& name, const glm::vec4& color = glm::vec4(0.8f))
        : cmdBuffer(cmdBuffer) {
        if (active) {
            beginRegion(cmdBuffer, name, color);
        }
    }

    ~Marker() {
        if (active) {
            endRegion(cmdBuffer);
        }
    }

private:
    const vk::CommandBuffer& cmdBuffer;
};
}  // namespace marker
}  // namespace debug
}  // namespace vks
