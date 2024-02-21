/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once
#if ENABLE_UI

#include <rendering/context.hpp>
#include <rendering/texture.hpp>

namespace vkx { namespace ui {

struct UIOverlayCreateInfo {
    std::vector<vk::ImageView> colorAttachmentViews;
    vk::SampleCountFlagBits rasterizationSamples{ vk::SampleCountFlagBits::e1 };
    vk::Format colorFormat;
    vk::Extent2D size;
    //vk::ImageView depthStencilView;
    //vk::Format depthFormat;
    //std::vector<vk::PipelineShaderStageCreateInfo> shaders;
    //uint32_t subpassCount{ 1 };
    //std::vector<vk::ClearValue> clearValues = {};
    //uint32_t attachmentCount = 1;
};

class UIOverlay {
private:
    UIOverlayCreateInfo createInfo;
    vks::QueueManager queueManager;
    const vk::Device& device{ queueManager.device };
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;

    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::CommandPool commandPool;
    vk::Fence fence;
    vk::Sampler fontSampler;
    vks::texture::Texture2D font;

    struct PushConstBlock {
        glm::vec2 scale;
        glm::vec2 translate;
    };
    PushConstBlock pushConstBlock;

    void prepareResources();
    void preparePipeline();
    void updateCommandBuffers();

public:
    bool visible = true;
    float scale = 1.0f;

    std::vector<vk::CommandBuffer> cmdBuffers;

    explicit UIOverlay() {}
    ~UIOverlay();

    void create(const vks::QueueManager& queueManager, const UIOverlayCreateInfo& createInfo);
    void destroy();

    void update();
    void resize(const vk::Extent2D& newSize, const std::vector<vk::ImageView>& views);

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
#endif