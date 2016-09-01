//
//  Created by Bradley Austin Davis on 2016/03/19
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#pragma once

#include <vector>
#include <algorithm>
#include <vulkan/vulkan.hpp>

namespace vkx {
    namespace shader {
        using SpvBuffer = std::vector<uint32_t>;
        void initGlsl();
        void finalizeGlsl();
        void initDebugReport(const vk::Instance& instance);

        SpvBuffer glslToSpv(vk::ShaderStageFlagBits shaderType, const std::string& shaderSource);
        vk::ShaderModule glslToShaderModule(const vk::Device& device, const vk::ShaderStageFlagBits shaderType, const std::string& shaderSource);
    }
}
