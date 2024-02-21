#pragma once

#include <string>
#include <vulkan/vulkan.hpp>

namespace vks { namespace shaders {

// vk::ShaderModule loadShaderModule(const vk::Device& device, const std::string& filename);
// vk::ShaderModule loadShaderModule(const vk::Device& device, const vk::ArrayProxy<const uint8_t>& code);

#if 0
// Load a SPIR-V shader from an spv file
vk::PipelineShaderStageCreateInfo loadShader(
    const vk::Device& device,
                                             const std::string& fileName,
                                             vk::ShaderStageFlagBits stage,
                                             const char* entryPoint = "main");
#endif

// Load a SPIR-V shader from a byte array
vk::PipelineShaderStageCreateInfo loadShader(const vk::Device& device,
                                             const vk::ArrayProxy<const uint8_t>& code,
                                             vk::ShaderStageFlagBits stage,
                                             const char* entryPoint = "main");

// Load a SPIR-V shader from a uint32_t array
vk::PipelineShaderStageCreateInfo loadShader(const vk::Device& device,
                                             const vk::ArrayProxy<const uint32_t>& code,
                                             vk::ShaderStageFlagBits stage,
                                             const char* entryPoint = "main");
}}  // namespace vks::shaders
