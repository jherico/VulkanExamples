/*
 * Vulkan examples debug wrapper
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "debug.hpp"
#include <iostream>
#include <mutex>
#include <sstream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "org.saintandreas.vulkan"
#endif

using namespace vks;

namespace color {

#define MAKE_COLOR_MANIPULATOR(name, code)                                                  \
    template <typename CharT, typename Traits = std::char_traits<CharT>>                    \
    inline std::basic_ostream<CharT, Traits>& name(std::basic_ostream<CharT, Traits>& os) { \
        return os << code;                                                                  \
    }

// These color definitions are based on the color scheme used by Git (the
// distributed version control system) as declared in the file
// `git/color.h`. You can add more manipulators as desired.

MAKE_COLOR_MANIPULATOR(normal, "")
MAKE_COLOR_MANIPULATOR(reset, "\033[m")
MAKE_COLOR_MANIPULATOR(bold, "\033[1m")
MAKE_COLOR_MANIPULATOR(red, "\033[31m")
MAKE_COLOR_MANIPULATOR(green, "\033[32m")
MAKE_COLOR_MANIPULATOR(yellow, "\033[33m")
MAKE_COLOR_MANIPULATOR(blue, "\033[34m")
MAKE_COLOR_MANIPULATOR(magenta, "\033[35m")
MAKE_COLOR_MANIPULATOR(cyan, "\033[36m")
MAKE_COLOR_MANIPULATOR(bold_red, "\033[1;31m")
MAKE_COLOR_MANIPULATOR(bold_green, "\033[1;32m")
MAKE_COLOR_MANIPULATOR(bold_yellow, "\033[1;33m")
MAKE_COLOR_MANIPULATOR(bold_blue, "\033[1;34m")
MAKE_COLOR_MANIPULATOR(bold_magenta, "\033[1;35m")
MAKE_COLOR_MANIPULATOR(bold_cyan, "\033[1;36m")
MAKE_COLOR_MANIPULATOR(bg_red, "\033[41m")
MAKE_COLOR_MANIPULATOR(bg_green, "\033[42m")
MAKE_COLOR_MANIPULATOR(bg_yellow, "\033[43m")
MAKE_COLOR_MANIPULATOR(bg_blue, "\033[44m")
MAKE_COLOR_MANIPULATOR(bg_magenta, "\033[45m")
MAKE_COLOR_MANIPULATOR(bg_cyan, "\033[46m")

}  // namespace color

namespace vks { namespace debug {

std::list<std::string> validationLayerNames = {
    // This is a meta layer that enables all of the standard
    // validation layers in the correct order :
    // threading, parameter_validation, device_limits, object_tracker, image, core_validation, swapchain, and unique_objects
    "VK_LAYER_KHRONOS_validation"
};

std::ostream& withSavedState(std::ostream& out, const std::function<void(std::ostream&)>& f) {
    std::ios state(nullptr);
    state.copyfmt(out);
    f(out);
    out.copyfmt(state);
    return out;
}

std::ostream& operator<<(std::ostream& out, const vk::DebugUtilsLabelEXT& label) {
    if (label.pLabelName) {
        out << label.pLabelName << "\t";
    }
    out << "color = {" << label.color[0] << ", " << label.color[1] << ", " << label.color[2] << ", " << label.color[3] << "}";
    return out;
}

static void emitLabels(std::ostream& buf, uint32_t count, const vk::DebugUtilsLabelEXT* labels) {
    for (uint32_t i = 0; i < count; i++) {
        buf << "\t\t" << labels[i] << "\n";
    }
}

std::ostream& operator<<(std::ostream& out, const vk::DebugUtilsObjectNameInfoEXT& object) {
    out << vk::to_string(object.objectType) << "\t0x" << std::hex << object.objectHandle << std::dec;
    if (object.pObjectName) {
        out << "\t" << object.pObjectName;
    }
    return out;
}

static void emitObjects(std::ostream& buf, uint32_t count, const vk::DebugUtilsObjectNameInfoEXT* pObjects) {
    for (uint32_t i = 0; i < count; ++i) {
        buf << "\t\t" << pObjects[i] << "\n";
    }
}

std::string trimValidationMessage(const std::string& message) {
    auto idIndex = message.find("MessageID = ");
    if (idIndex == std::string::npos) {
        return message;
    }
    auto lastPipe = message.find("| ", idIndex);
    if (lastPipe == std::string::npos) {
        return message;
    }
    return message.substr(lastPipe + 2);
}

std::ostream& operator<<(std::ostream& buf, const MessageCallbackData& callbackData) {
    if (callbackData.messageIdNumber != 0) {
        buf << "[0x" << std::hex << callbackData.messageIdNumber << std::dec << "]\t";
    }
    if (callbackData.pMessageIdName) {
        buf << "(" << callbackData.pMessageIdName << ")\t";
    }
    if (callbackData.messageIdNumber != 0) {
        buf << "\"" << trimValidationMessage(callbackData.pMessage) << "\"\n";
    } else {
        buf << "\"" << callbackData.pMessage << "\"\n";
    }
    if (callbackData.queueLabelCount != 0) {
        buf << "\tQueue Labels:\n";
        emitLabels(buf, callbackData.queueLabelCount, callbackData.pQueueLabels);
    }
    if (callbackData.cmdBufLabelCount != 0) {
        buf << "\tCommandBuffer Labels:\n";
        emitLabels(buf, callbackData.cmdBufLabelCount, callbackData.pCmdBufLabels);
    }
    if (callbackData.objectCount) {
        buf << "\tObjects:\n";
        emitObjects(buf, callbackData.objectCount, callbackData.pObjects);
    }
    return buf;
}

static VkBool32 defaultMessageHandler(const MessageSeverityBits& severityFlags,
                                      const MessageType& typeFlags,
                                      const MessageCallbackData& callbackData,
                                      void* pUserData) {
    std::string message;
    {
        std::stringstream buf;
        buf << "Severity: " << vk::to_string(severityFlags) << "\t";
        buf << "Type: " << vk::to_string(typeFlags) << "\n";
        buf << callbackData;
        message = buf.str();
    }

    std::cout << message;

#ifdef __ANDROID__
    __android_log_write(ANDROID_LOG_DEBUG, LOG_TAG, message.c_str());
#endif
#ifdef _MSC_VER
    OutputDebugStringA(message.c_str());
#endif
    return VK_FALSE;
}

static MessageHandler messageHandler = defaultMessageHandler;
vk::DebugUtilsMessengerEXT msgCallback;

VkBool32 messageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                         VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                         const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                         void* pUserData) {
    if (messageHandler) {
        return messageHandler(reinterpret_cast<MessageSeverityBits&>(messageSeverity), MessageType{ messageTypes }, MessageCallbackData{ *pCallbackData },
                              pUserData);
    };
    return VK_FALSE;
}

void setupDebugging(vk::InstanceCreateInfo& instanceCreateInfo,
                    const vk::DebugUtilsMessageSeverityFlagsEXT& severityFlags,
                    const vk::DebugUtilsMessageTypeFlagsEXT& typeFlags) {
    static vk::DebugUtilsMessengerCreateInfoEXT dbgCreateInfo;
    dbgCreateInfo.pNext = instanceCreateInfo.pNext;
    dbgCreateInfo.messageSeverity = severityFlags;
    dbgCreateInfo.messageType = typeFlags;
    dbgCreateInfo.pfnUserCallback = messageCallback;
    instanceCreateInfo.pNext = &dbgCreateInfo;
}

static vk::Instance instance;
void setupDebugging(const vk::Instance& instance,
                    const vk::DebugUtilsMessageSeverityFlagsEXT& severityFlags,
                    const vk::DebugUtilsMessageTypeFlagsEXT& typeFlags) {
    vks::debug::instance = instance;
    vk::DebugUtilsMessengerCreateInfoEXT dbgCreateInfo = {};
    dbgCreateInfo.messageSeverity = severityFlags;
    dbgCreateInfo.messageType = typeFlags;
    dbgCreateInfo.pfnUserCallback = messageCallback;
    msgCallback = instance.createDebugUtilsMessengerEXT(dbgCreateInfo, nullptr);
}

void freeDebugCallback(const vk::Instance& instance) {
    instance.destroyDebugUtilsMessengerEXT(msgCallback, nullptr);
}

namespace marker {
bool active = false;

void setup(const vk::Instance& instance, const vk::Device& device) {
    // Set flag if at least one function pointer is present
    active = (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDebugUtilsMessengerEXT != VK_NULL_HANDLE);
}

static std::array<float, 4> toFloatArray(const glm::vec4& color) {
    return { color.r, color.g, color.b, color.a };
}

void beginRegion(const vk::CommandBuffer& cmdbuffer, const std::string& markerName, const glm::vec4& color) {
    // Check for valid function pointer (may not be present if not running in a debugging application)
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT) {
        // cmdBuffer.bein
        auto colorArray = toFloatArray(color);
        cmdbuffer.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{ markerName.c_str(), colorArray });
    }
}

void insert(const vk::CommandBuffer& cmdbuffer, const std::string& markerName, const glm::vec4& color) {
    // Check for valid function pointer (may not be present if not running in a debugging application)
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdDebugMarkerInsertEXT) {
        auto colorArray = toFloatArray(color);
        cmdbuffer.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{ markerName.c_str(), colorArray });
    }
}

void endRegion(const vk::CommandBuffer& cmdbuffer) {
    // Check for valid function (may not be present if not runnin in a debugging application)
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT) {
        cmdbuffer.endDebugUtilsLabelEXT();
    }
}

};  // namespace marker
}}  // namespace vks::debug
