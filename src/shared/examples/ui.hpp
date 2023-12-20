/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "vks/context.hpp"
#ifdef __ANDROID__
#include <android/native_activity.h>
#endif

namespace vkx { namespace ui {

struct UIOverlayCreateInfo {
    vk::Queue copyQueue;
    vk::RenderPass renderPass;
    std::vector<vk::Framebuffer> framebuffers;
    vk::Format colorformat;
    vk::Format depthformat;
    vk::Extent2D size;
    std::vector<vk::PipelineShaderStageCreateInfo> shaders;
    vk::SampleCountFlagBits rasterizationSamples{ vk::SampleCountFlagBits::e1 };
    uint32_t subpassCount{ 1 };
    std::vector<vk::ClearValue> clearValues = {};
    uint32_t attachmentCount = 1;
};

class UIOverlay {
private:
    UIOverlayCreateInfo createInfo;
    const vks::Context& context;
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    int32_t vertexCount = 0;
    int32_t indexCount = 0;

    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;
    vk::PipelineLayout pipelineLayout;
    const vk::PipelineCache& pipelineCache{ context.pipelineCache };
    vk::Pipeline pipeline;
    vk::RenderPass renderPass;
    vk::CommandPool commandPool;
    vk::Fence fence;

    vks::Image font;

    struct PushConstBlock {
        glm::vec2 scale;
        glm::vec2 translate;
    };
    PushConstBlock pushConstBlock;

    void prepareResources();
    void preparePipeline();
    void prepareRenderPass();
    void updateCommandBuffers();

public:
    bool visible = true;
    float scale = 1.0f;

    std::vector<vk::CommandBuffer> cmdBuffers;

    explicit UIOverlay(const vks::Context& context)
        : context(context) {}
    ~UIOverlay();

    void create(const UIOverlayCreateInfo& createInfo);
    void destroy();

    void update();
    void resize(const vk::Extent2D& newSize, const std::vector<vk::Framebuffer>& framebuffers);

    void submit(const vk::Queue& queue, uint32_t bufferindex, vk::SubmitInfo submitInfo) const;

    bool header(const char* caption) const;
    bool checkBox(const char* caption, bool* value) const;
    bool checkBox(const char* caption, int32_t* value) const;
    bool inputFloat(const char* caption, float* value, float step, const char* precision = "%.3f") const;
    bool sliderFloat(const char* caption, float* value, float min, float max) const;
    bool sliderInt(const char* caption, int32_t* value, int32_t min, int32_t max) const;
    bool comboBox(const char* caption, int32_t* itemindex, const std::vector<std::string>& items) const;
    bool button(const char* caption) const;
    void text(const char* formatstr, ...) const;
};
}}  // namespace vkx::ui
