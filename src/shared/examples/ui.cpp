/*
 * UI overlay class using ImGui
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "ui.hpp"
#include <vks/debug.hpp>

#if ENABLE_UI

#include <imgui.h>

#include <rendering/recycler.hpp>
#include <vks/helpers.hpp>
#include <vks/pipelines.hpp>

#include <common/utils.hpp>
#include <shaders/base/uioverlay.frag.inl>
#include <shaders/base/uioverlay.vert.inl>

#if defined(__ANDROID__)
#include <android/android.hpp>
#endif

using namespace vkx;
using namespace vkx::ui;

void UIOverlay::create(const vks::QueueManager& queueManager, const UIOverlayCreateInfo& createInfo_) {
    this->queueManager = queueManager;
    createInfo = createInfo_;
#if defined(__ANDROID__)
    // Screen density
    if (vkx::android::screenDensity >= ACONFIGURATION_DENSITY_XXXHIGH) {
        scale = 4.5f;
    } else if (vkx::android::screenDensity >= ACONFIGURATION_DENSITY_XXHIGH) {
        scale = 3.5f;
    } else if (vkx::android::screenDensity >= ACONFIGURATION_DENSITY_XHIGH) {
        scale = 2.5f;
    } else if (vkx::android::screenDensity >= ACONFIGURATION_DENSITY_HIGH) {
        scale = 2.0f;
    };
    vkx::logMessage(vkx::LogLevel::LOG_DEBUG, "Android UI scale %f", scale);
#endif

    // Init ImGui
    // Color scheme
    ImGuiStyle& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.0f, 0.0f, 0.0f, 0.1f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.8f, 0.0f, 0.0f, 0.4f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
    // Dimensions
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)createInfo.size.width, (float)createInfo.size.height);
    io.FontGlobalScale = scale;

    prepareResources();
    preparePipeline();
}

/** Free up all Vulkan resources acquired by the UI overlay */
UIOverlay::~UIOverlay() {
    destroy();
}

void UIOverlay::destroy() {
    if (commandPool) {
        vertexBuffer.destroy();
        indexBuffer.destroy();
        font.destroy();
        if (fontSampler) {
            device.destroy(fontSampler);
            fontSampler = nullptr;
        }
        device.destroy(descriptorSetLayout);
        descriptorSetLayout = nullptr;
        device.destroy(descriptorPool);
        descriptorPool = nullptr;
        device.destroy(pipelineLayout);
        pipelineLayout = nullptr;
        device.destroy(pipeline);
        pipeline = nullptr;
        device.free(commandPool, cmdBuffers);
        cmdBuffers.clear();
        device.destroy(commandPool);
        commandPool = nullptr;
        device.destroy(fence);
        fence = nullptr;
        commandPool = nullptr;
    }
}

/** Prepare all vulkan resources required to render the UI overlay */
void UIOverlay::prepareResources() {
    ImGuiIO& io = ImGui::GetIO();

    // Create font texture
    {
        std::vector<uint8_t> fontData;
        int texWidth, texHeight;
        unsigned char* fontBuffer;
        io.Fonts->GetTexDataAsRGBA32(&fontBuffer, &texWidth, &texHeight);
        vk::DeviceSize uploadSize = texWidth * texHeight * 4 * sizeof(char);
        fontData.resize(uploadSize);
        memcpy(fontData.data(), fontBuffer, uploadSize);
        vks::Image::Builder fontBuilder{ (uint32_t)texWidth, (uint32_t)texHeight };
        fontBuilder.withFormat(vk::Format::eR8G8B8A8Unorm);
        fontBuilder.withUsage(vk::ImageUsageFlagBits::eSampled);
        font.fromBuffer(fontData, fontBuilder);
    }

    // Create sampler
    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.magFilter = vk::Filter::eNearest;
    samplerCreateInfo.minFilter = vk::Filter::eNearest;
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    fontSampler = device.createSampler(samplerCreateInfo);

    // Command buffer
    commandPool = queueManager.createCommandPool();

    // Descriptor pool
    vk::DescriptorPoolSize poolSize{ vk::DescriptorType::eCombinedImageSampler, 1 };
    descriptorPool = device.createDescriptorPool({ {}, 2, poolSize });

    // Descriptor set layout
    vk::DescriptorSetLayoutBinding setLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    descriptorSetLayout = device.createDescriptorSetLayout({ {}, setLayoutBinding });

    // Descriptor set
    descriptorSet = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, descriptorSetLayout })[0];

    auto fontImageInfo = font.makeDescriptor(fontSampler);
    device.updateDescriptorSets(vk::WriteDescriptorSet{ descriptorSet, 0, 0, vk::DescriptorType::eCombinedImageSampler, fontImageInfo }, nullptr);

    // Pipeline layout
    // Push constants for UI rendering parameters
    vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstBlock) };
    pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, descriptorSetLayout, pushConstantRange });

    // Command buffer execution fence
    fence = device.createFence(vk::FenceCreateInfo{});
}

/** Prepare a separate pipeline for the UI overlay rendering decoupled from the main application */
void UIOverlay::preparePipeline() {
    // Setup graphics pipeline for UI rendering
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder(device, pipelineLayout);
    pipelineBuilder.dynamicRendering(createInfo.colorFormat);
    pipelineBuilder.multisampleState.rasterizationSamples = createInfo.rasterizationSamples;
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    // Enable blending
    pipelineBuilder.colorBlendState.blendAttachmentStates.resize(1);
    for (uint32_t i = 0; i < 1; i++) {
        auto& blendAttachmentState = pipelineBuilder.colorBlendState.blendAttachmentStates[i];
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
    }
    pipelineBuilder.loadShader(vkx::shaders::base::uioverlay::vert, vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::shaders::base::uioverlay::frag, vk::ShaderStageFlagBits::eFragment);

    // Vertex bindings an attributes based on ImGui vertex definition
    pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, sizeof(ImDrawVert), vk::VertexInputRate::eVertex } };
    pipelineBuilder.vertexInputState.attributeDescriptions = {
        { 0, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, pos) },   // Location 0: Position
        { 1, 0, vk::Format::eR32G32Sfloat, offsetof(ImDrawVert, uv) },    // Location 1: UV
        { 2, 0, vk::Format::eR8G8B8A8Unorm, offsetof(ImDrawVert, col) },  // Location 2: Color
    };
    pipeline = pipelineBuilder.create({});
}

/** Update the command buffers to reflect UI changes */
void UIOverlay::updateCommandBuffers() {
    vk::CommandBufferBeginInfo cmdBufInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse };

    ImGuiIO& io = ImGui::GetIO();

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eLoad;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.clearValue = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));

    // vk::RenderingAttachmentInfo depthAttachmentInfo;
    // depthAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
    // depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    // depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    // depthAttachmentInfo.clearValue = vk::ClearDepthStencilValue{ 0.0, 0 };
    // depthAttachmentInfo.imageView = createInfo.depthStencilView;

    vk::RenderingInfo renderingInfo;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.layerCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    // renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, createInfo.size };

    const vk::Viewport viewport{ 0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y, 0.0f, 1.0f };
    const vk::Rect2D scissor{ {}, vk::Extent2D{ (uint32_t)io.DisplaySize.x, (uint32_t)io.DisplaySize.y } };
    // UI scale and translate via push constants
    pushConstBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
    pushConstBlock.translate = glm::vec2(-1.0f);

    if (cmdBuffers.size()) {
        vks::Recycler::get().trashCommandBuffers(commandPool, cmdBuffers);
        cmdBuffers.clear();
    }

    cmdBuffers = device.allocateCommandBuffers({ commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)createInfo.colorAttachmentViews.size() });

    for (size_t i = 0; i < cmdBuffers.size(); ++i) {
        const auto& cmdBuffer = cmdBuffers[i];
        cmdBuffer.begin(cmdBufInfo);
        vks::debug::marker::beginRegion(cmdBuffer, "UI overlay", glm::vec4(1.0f, 0.94f, 0.3f, 1.0f));
        colorAttachmentInfo.imageView = createInfo.colorAttachmentViews[i];
        cmdBuffer.beginRendering(renderingInfo);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, {});
        cmdBuffer.bindVertexBuffers(0, vertexBuffer.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint16);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, scissor);
        cmdBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, vk::ArrayProxy<const PushConstBlock>{ pushConstBlock });

        // Render commands
        ImDrawData* imDrawData = ImGui::GetDrawData();
        int32_t vertexOffset = 0;
        int32_t indexOffset = 0;
        for (int32_t j = 0; j < imDrawData->CmdListsCount; j++) {
            const ImDrawList* cmd_list = imDrawData->CmdLists[j];
            for (int32_t k = 0; k < cmd_list->CmdBuffer.Size; k++) {
                const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[k];
                vk::Rect2D scissorRect;
                scissorRect.offset.x = std::max((int32_t)(pcmd->ClipRect.x), 0);
                scissorRect.offset.y = std::max((int32_t)(pcmd->ClipRect.y), 0);
                scissorRect.extent.width = (uint32_t)(pcmd->ClipRect.z - pcmd->ClipRect.x);
                scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - pcmd->ClipRect.y);
                cmdBuffer.setScissor(0, scissorRect);
                cmdBuffer.drawIndexed(pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
                indexOffset += pcmd->ElemCount;
            }
            vertexOffset += cmd_list->VtxBuffer.Size;
        }
        cmdBuffer.endRendering();
        vks::debug::marker::endRegion(cmdBuffer);
        cmdBuffer.end();
    }
}

template <typename T>
void copyToBuffer(const ImVector<T>& v, vks::Buffer& buffer, vk::DeviceSize& offset) {
    const vk::DeviceSize result = v.Size * sizeof(T);
    buffer.copy(result, v.Data, offset);
    offset += result;
}

void copyToBuffers(const ImVector<ImDrawList*>& cmdLists, vks::Buffer& vertexBuffer, vks::Buffer& indexBuffer) {
    vk::DeviceSize vertexOffset = 0;
    vk::DeviceSize indexOffset = 0;
    for (int n = 0; n < cmdLists.Size; n++) {
        const auto& cmdList = *cmdLists[n];
        copyToBuffer(cmdList.VtxBuffer, vertexBuffer, vertexOffset);
        copyToBuffer(cmdList.IdxBuffer, indexBuffer, indexOffset);
    }
}

void allocate(vks::Buffer& buffer, vk::DeviceSize size, const vk::BufferUsageFlags& usageFlags) {
    vk::DeviceSize oldSize = buffer.createInfo.size;
    if (!buffer || oldSize < size) {
        if (buffer) {
            vks::Recycler::get().trash<vks::Buffer>(std::move(buffer));
        }
        buffer = {};
        // Avoid frequent reallocation by doubling either the old size or the required size, whichever is larger
        auto allocateSize = std::max(oldSize, size) * 2;
        vks::Buffer::Builder builder{ allocateSize };
        builder.withBufferUsage(usageFlags);
        builder.withAllocPreferredFlags(vk::MemoryPropertyFlagBits::eHostCoherent);
        builder.withAllocCreateFlags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);
        // Create the buffers with double the required size to reduce
        auto& deviceInfo = vks::Context::get().deviceInfo;
        buffer.create(builder);
        //const auto& memoryTypes = deviceInfo.memoryProperties.core.memoryTypes;
        //const auto& memoryType = memoryTypes[buffer.getMemoryType()];
        //std::string memoryTypeStr = vk::to_string(memoryType.propertyFlags);
    }
}

static boolean isCoherent(const vks::Buffer& buffer, const vk::PhysicalDeviceMemoryProperties& memoryProperties) {
    const vk::MemoryType& memoryType = memoryProperties.memoryTypes[buffer.getMemoryType()];
    return vk::MemoryPropertyFlagBits::eHostCoherent == (memoryType.propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
}

/** Update vertex and index buffer containing the imGui elements when required */
void UIOverlay::update() {
    ImDrawData* imDrawData = ImGui::GetDrawData();
    bool updateCmdBuffers = false;

    if (!imDrawData) {
        return;
    }

    // Note: Alignment is done inside buffer creation
    vk::DeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
    vk::DeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);
    if (!vertexBufferSize || !indexBufferSize) {
        return;
    }

    // Update buffers only if vertex or index count has been changed compared to current buffer size
    // REALLOCATE buffers only if the old buffers were too small

    // Vertex buffer
    allocate(vertexBuffer, vertexBufferSize, vk::BufferUsageFlagBits::eVertexBuffer);
    allocate(indexBuffer, indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer);

    if (indexCount != imDrawData->TotalIdxCount || vertexCount != imDrawData->TotalVtxCount) {
        indexCount = imDrawData->TotalIdxCount;
        vertexCount = imDrawData->TotalVtxCount;
        updateCmdBuffers = true;
    }

    // Upload data
    copyToBuffers(imDrawData->CmdLists, vertexBuffer, indexBuffer);

    // Flush to make writes visible to GPU
    static const auto memoryProperties = vks::Context::get().deviceInfo.memoryProperties.core;
    if (!isCoherent(vertexBuffer, memoryProperties)) {
        vertexBuffer.flush();
    }
    if (!isCoherent(indexBuffer, memoryProperties)) {
        indexBuffer.flush();
    }

    if (updateCmdBuffers) {
        updateCommandBuffers();
    }
}

void UIOverlay::resize(const vk::Extent2D& size, const std::vector<vk::ImageView>& views) {
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)(size.width), (float)(size.height));
    createInfo.size = size;
    createInfo.colorAttachmentViews = views;
    // createInfo.depthStencilView = depthStencilView;
    updateCommandBuffers();
}

/** Submit the overlay command buffers to a graphicsQueue */
void UIOverlay::submit(const vk::Queue& queue, uint32_t bufferindex, vk::SubmitInfo submitInfo) const {
    if (!visible) {
        return;
    }

    submitInfo.pCommandBuffers = &cmdBuffers[bufferindex];
    submitInfo.commandBufferCount = 1;

    queue.submit(submitInfo, fence);
    // We get the result because the function specifies nodiscard, but since we've set the timeout to UINT64_MAX we shouldn't ever get a timeout result.
    auto waitResult = device.waitForFences(fence, VK_TRUE, UINT64_MAX);
    device.resetFences(fence);
}

bool UIOverlay::header(const char* caption) const {
    return ImGui::CollapsingHeader(caption, ImGuiTreeNodeFlags_DefaultOpen);
}

bool UIOverlay::checkBox(const char* caption, bool* value) const {
    return ImGui::Checkbox(caption, value);
}

bool UIOverlay::checkBox(const char* caption, int32_t* value) const {
    bool val = (*value == 1);
    bool res = ImGui::Checkbox(caption, &val);
    *value = val;
    return res;
}

bool UIOverlay::inputFloat(const char* caption, float* value, float step, const char* format) const {
    return ImGui::InputFloat(caption, value, step, step * 10.0f, format);
}

bool UIOverlay::sliderFloat(const char* caption, float* value, float min, float max) const {
    return ImGui::SliderFloat(caption, value, min, max);
}

bool UIOverlay::sliderInt(const char* caption, int32_t* value, int32_t min, int32_t max) const {
    return ImGui::SliderInt(caption, value, min, max);
}

bool UIOverlay::comboBox(const char* caption, int32_t* itemindex, const std::vector<std::string>& items) const {
    if (items.empty()) {
        return false;
    }
    std::vector<const char*> charitems;
    charitems.reserve(items.size());
    for (size_t i = 0; i < items.size(); i++) {
        charitems.push_back(items[i].c_str());
    }
    auto itemCount = static_cast<uint32_t>(charitems.size());
    return ImGui::Combo(caption, itemindex, &charitems[0], itemCount, itemCount);
}

bool UIOverlay::button(const char* caption) const {
    return ImGui::Button(caption);
}

void UIOverlay::text(const char* formatstr, ...) const {
    va_list args;
    va_start(args, formatstr);
    ImGui::TextV(formatstr, args);
    va_end(args);
}
#endif