#include "shaders.hpp"
#include <common/filesystem.hpp>
#include <common/storage.hpp>

// vk::ShaderModule vks::shaders::loadShaderModule(const vk::Device& device, const vk::ArrayProxy<const uint8_t>& code) {
//     return device.createShaderModule(vk::ShaderModuleCreateInfo{ {}, code.size(), (const uint32_t*)code.data()});
// }

#if 0
// Load a SPIR-V shader
vk::PipelineShaderStageCreateInfo vks::shaders::loadShader(const vk::Device& device,
                                                           const std::string& fileName,
                                                           vk::ShaderStageFlagBits stage,
                                                           const char* entryPoint) {

    auto file = vks::storage::Storage::readFile(fileName);
    return loadShader(device, file->span(), stage, entryPoint);
}
#endif

// Load a SPIR-V shader
vk::PipelineShaderStageCreateInfo vks::shaders::loadShader(const vk::Device& device,
                                                           const vk::ArrayProxy<const uint8_t>& code,
                                                           vk::ShaderStageFlagBits stage,
                                                           const char* entryPoint) {
    vk::PipelineShaderStageCreateInfo shaderStage;
    shaderStage.stage = stage;
    shaderStage.module = device.createShaderModule({ {}, code.size(), (uint32_t*)code.data() });
    shaderStage.pName = entryPoint;
    return shaderStage;
}

// Load a SPIR-V shader
vk::PipelineShaderStageCreateInfo vks::shaders::loadShader(const vk::Device& device,
                                                           const vk::ArrayProxy<const uint32_t>& code,
                                                           vk::ShaderStageFlagBits stage,
                                                           const char* entryPoint) {
    vk::PipelineShaderStageCreateInfo shaderStage;
    shaderStage.stage = stage;
    shaderStage.module = device.createShaderModule({ {}, code });
    shaderStage.pName = entryPoint;
    return shaderStage;
}
