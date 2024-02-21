/*
 * Vulkan Example - Compute shader image processing
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/compute.hpp>
#include <examples/example.hpp>
#include <examples/keycodes.hpp>

#include <shaders/computeshader/edgedetect.comp.inl>
#include <shaders/computeshader/emboss.comp.inl>
#include <shaders/computeshader/sharpen.comp.inl>
#include <shaders/computeshader/texture.frag.inl>
#include <shaders/computeshader/texture.vert.inl>

using Pair = std::pair<std::string, vk::ArrayProxyNoTemporaries<const uint32_t>>;

const std::vector<Pair> AVAILABLE_SHADERS{
    { "sharpen", vkx::shaders::computeshader::sharpen::comp },
    { "edgedetect", vkx::shaders::computeshader::edgedetect::comp },
    { "emboss", vkx::shaders::computeshader::emboss::comp },
};

const std::vector<std::string> SHADER_NAMES{
    "sharpen",
    "edgedetect",
    "emboss",
};

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
};

struct SampledTexture : public vks::texture::Texture2D {};

struct SharedResources {
    vks::texture::Texture2D colorMap;
    vks::texture::Texture2D computed;
    vk::Extent3D extent;

    void prepare(const std::string& colorMapFile, vk::Sampler defaultSampler) {
        auto format = vk::Format::eR8G8B8A8Unorm;
        auto usageFlags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
        colorMap.loadFromFile(colorMapFile, format, usageFlags, vk::ImageLayout::eGeneral);
        extent = colorMap.image.createInfo.extent;

        const auto& context = vks::Context::get();
        const auto& device = context.device;

        vks::Image::Builder builder{ colorMap.image.createInfo.extent };
        builder.withFormat(format).withUsage(usageFlags);
        // Get device properties for the requested texture format
        vk::FormatProperties formatProperties;
        formatProperties = context.physicalDevice.getFormatProperties(format);
        // Check if requested image format supports image storage operations
        assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);
        computed.build(builder);
    }

    void destroy() {
        auto& device = vks::Context::get().device;
        colorMap.destroy();
        computed.destroy();
    }
} sharedResources;

class ComputeImage : public vkx::Compute {
    using Parent = vkx::Compute;

public:
    vk::Sampler sampler;
    vk::DescriptorPool descriptorPool;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;
    std::vector<vk::Pipeline> pipelines;
    std::vector<vk::CommandBuffer> commandBuffers;
    int32_t pipelineIndex{ 0 };
    std::array<vk::ImageMemoryBarrier2, 2> acquireBarriers, releaseBarriers;

    void destroy() override {
        computeQueue.handle.waitIdle();
        computeQueue.freeCommandBuffers(commandBuffers);
        device.destroy(sampler);
        device.destroy(descriptorPool);
        // Clean up used Vulkan resources
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        for (auto& pipeline : pipelines) {
            device.destroy(pipeline);
        }
        Parent::destroy();
    }

    void prepare(uint32_t swapchainCount) {
        Parent::prepare(swapchainCount);
        // Create sampler
        vk::SamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.magFilter = vk::Filter::eLinear;
        samplerCreateInfo.minFilter = vk::Filter::eLinear;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
        samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
        samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
        samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        sampler = device.createSampler(samplerCreateInfo);

        acquireBarriers[0] = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead,
            vk::ImageLayout::eReadOnlyOptimal,
            vk::ImageLayout::eGeneral,
            // Src and dst queues
            context.queuesInfo.graphics.index,
            context.queuesInfo.compute.index,
            // Buffer
            sharedResources.colorMap.image.image,
            sharedResources.colorMap.image.getWholeRange(),
        };
        acquireBarriers[1] = acquireBarriers[0];
        acquireBarriers[1].image = sharedResources.computed.image.image;
        acquireBarriers[1].dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

        releaseBarriers[0] = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Layouts
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eReadOnlyOptimal,
            // Src and dst queues
            context.queuesInfo.compute.index,
            context.queuesInfo.graphics.index,
            // Buffer
            sharedResources.colorMap.image.image,
            sharedResources.colorMap.image.getWholeRange(),
        };
        releaseBarriers[1] = releaseBarriers[0];
        releaseBarriers[1].image = sharedResources.computed.image.image;
        acquireBarriers[1].srcAccessMask = vk::AccessFlagBits2::eShaderWrite;

        prepareDescriptors();
        preparePipelines();
    }

    void prepareDescriptors() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            // Compute pipelines uses storage images for reading and writing
            { vk::DescriptorType::eStorageImage, 2 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });

        // Create compute pipeline
        // Compute pipelines are created separate from graphics pipelines
        // even if they use the same graphicsQueue

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Sampled image (read)
            { 0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 1 : Sampled image (write)
            { 1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });

        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];
        auto colorMapDesc = sharedResources.colorMap.makeDescriptor(sampler, vk::ImageLayout::eGeneral);
        auto computeTargetDesc = sharedResources.computed.makeDescriptor(sampler, vk::ImageLayout::eGeneral);
        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            // Binding 0 : Sampled image (read)
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &colorMapDesc },
            // Binding 1 : Sampled image (write)
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageImage, &computeTargetDesc },
        };
        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Create compute shader pipelines
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        vk::ComputePipelineCreateInfo computePipelineCreateInfo{ {}, {}, pipelineLayout };
        // One pipeline for each effect
        for (auto& pair : AVAILABLE_SHADERS) {
            const auto& shaderName = pair.first;
            const auto& shaderCode = pair.second;
            computePipelineCreateInfo.stage = vks::shaders::loadShader(device, shaderCode, vk::ShaderStageFlagBits::eCompute);
            pipelines.push_back(device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo, nullptr).value);
            device.destroy(computePipelineCreateInfo.stage.module);
        }
        commandBuffers = computeQueue.allocateCommandBuffers((uint32_t)AVAILABLE_SHADERS.size());
        rebuildCommandBuffers();
    }

    void buildCommandBuffers() {
        for (const auto& commandBuffer : commandBuffers) {
            commandBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            const auto& pipeline = pipelines[pipelineIndex];
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, acquireBarriers });
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);
            commandBuffer.dispatch(sharedResources.extent.width / 16, sharedResources.extent.height / 16, 1);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, releaseBarriers });
            commandBuffer.end();
        }
    }

    void rebuildCommandBuffers() {
        auto size = (uint32_t)commandBuffers.size();
        vks::Recycler::get().trashCommandBuffers(computeQueue.pool, commandBuffers);
        commandBuffers = computeQueue.allocateCommandBuffers(size);
        buildCommandBuffers();
    }

    void switchPipeline(int32_t dir) {
        pipelineIndex += dir;
        pipelineIndex %= pipelines.size();
        rebuildCommandBuffers();
    }
};

class VulkanExample : public vkx::ExampleBase {
    using Parent = ExampleBase;

public:
    struct {
        vks::model::Model quad;
    } meshes;

    vks::Buffer uniformDataVS;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVS;

    struct Graphics {
        vk::PipelineLayout pipelineLayout;
        vk::DescriptorSet descriptorSetPreCompute;
        vk::DescriptorSet descriptorSetPostCompute;
        vk::Pipeline pipeline;
        vk::DescriptorSetLayout descriptorSetLayout;
    } graphics;
    vk::Sampler computeSampler;

    std::array<vk::ImageMemoryBarrier2, 2> acquireBarriers, releaseBarriers;
    ComputeImage compute;

    VulkanExample() {
        camera.dolly(-2.0f);
        title = "Vulkan Example - Compute shader image processing";
    }

    ~VulkanExample() {
        waitIdle();

        device.destroy(computeSampler);
        compute.destroy();
        device.destroy(graphics.pipeline);
        device.destroy(graphics.pipelineLayout);
        device.destroy(graphics.descriptorSetLayout);

        meshes.quad.destroy();
        uniformDataVS.destroy();
        sharedResources.destroy();
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, acquireBarriers });
        Parent::updateCommandBufferPreDraw(cmdBuffer);
    }

    void updateCommandBufferPostDraw(const vk::CommandBuffer& cmdBuffer) override {
        Parent::updateCommandBufferPostDraw(cmdBuffer);
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, releaseBarriers });
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });

        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        // Left (pre compute)
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSetPreCompute, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);

        vk::Viewport viewport = vks::util::viewport((float)size.width / 2, (float)size.height, 0.0f, 1.0f);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);

        // Right (post compute)
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, graphics.pipelineLayout, 0, graphics.descriptorSetPostCompute, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics.pipeline);

        viewport.x = viewport.width;
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
    }

    // Setup vertices for a single uv-mapped quad
    void generateQuad() {
#define dim 1.0f
        std::vector<Vertex> vertexBuffer = { { { dim, dim, 0.0f }, { 1.0f, 1.0f } },
                                             { { -dim, dim, 0.0f }, { 0.0f, 1.0f } },
                                             { { -dim, -dim, 0.0f }, { 0.0f, 0.0f } },
                                             { { dim, -dim, 0.0f }, { 1.0f, 0.0f } } };
#undef dim
        meshes.quad.vertices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 2 },
            // Graphics pipeline uses image samplers for display
            { vk::DescriptorType::eCombinedImageSampler, 4 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        graphics.descriptorSetLayout = device.createDescriptorSetLayout({ {}, setLayoutBindings });
        graphics.pipelineLayout = device.createPipelineLayout({ {}, 1, &graphics.descriptorSetLayout });
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &graphics.descriptorSetLayout };
        graphics.descriptorSetPostCompute = device.allocateDescriptorSets(allocInfo)[0];
        auto computedDescriptor = sharedResources.computed.makeDescriptor(defaultSampler);
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { graphics.descriptorSetPostCompute, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor },
            // Binding 1 : Fragment shader texture sampler
            { graphics.descriptorSetPostCompute, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &computedDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);

        // Base image (before compute post process)
        graphics.descriptorSetPreCompute = device.allocateDescriptorSets(allocInfo)[0];
        auto colorMapDescriptor = sharedResources.colorMap.makeDescriptor(defaultSampler);
        writeDescriptorSets = {
            // Binding 0 : Vertex shader uniform buffer
            { graphics.descriptorSetPreCompute, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor },
            // Binding 1 : Fragment shader texture sampler
            { graphics.descriptorSetPreCompute, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorMapDescriptor, &uniformDataVS.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, graphics.pipelineLayout };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.depthStencilState = { false };
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::computeshader::texture::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::computeshader::texture::frag, vk::ShaderStageFlagBits::eFragment);

        // Binding description
        pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, sizeof(Vertex), vk::VertexInputRate::eVertex } };

        // Attribute descriptions
        // Describes memory layout and shader positions
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Location 0 : Position
            { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
            // Location 1 : Texture coordinates
            { 1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv) },
        };

        graphics.pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformDataVS = loader.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader uniform buffer block
        uboVS.projection = glm::perspective(glm::radians(60.0f), (float)(size.width / 2) / size.height, 0.1f, 256.0f);
        uboVS.model = camera.matrices.view;
        uniformDataVS.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();

        sharedResources.prepare(getAssetPath() + "textures/het_kanonschot_rgba8.ktx", defaultSampler);

        acquireBarriers[0] = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderRead,
            // Layouts
            vk::ImageLayout::eGeneral,
            vk::ImageLayout::eReadOnlyOptimal,
            // Src and dst queues
            context.queuesInfo.compute.index,
            context.queuesInfo.graphics.index,
            // Buffer
            sharedResources.colorMap.image.image,
            sharedResources.colorMap.image.getWholeRange(),
        };
        acquireBarriers[1] = acquireBarriers[0];
        acquireBarriers[1].image = sharedResources.computed.image.image;

        releaseBarriers[0] = {
            // Src stage & access
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::AccessFlagBits2::eShaderRead,
            // Dst stage & access
            vk::PipelineStageFlagBits2::eNone,
            vk::AccessFlagBits2::eNone,
            // Layouts
            vk::ImageLayout::eReadOnlyOptimal,
            vk::ImageLayout::eGeneral,
            // Src and dst queues
            context.queuesInfo.graphics.index,
            context.queuesInfo.compute.index,
            // Buffer
            sharedResources.colorMap.image.image,
            sharedResources.colorMap.image.getWholeRange(),
        };
        releaseBarriers[1] = releaseBarriers[0];
        releaseBarriers[1].image = sharedResources.computed.image.image;

        compute.prepare(swapChain.imageCount);

        // Set up the initial release barriers
        loader.withPrimaryCommandBuffer(computeQueue, [&](const vk::CommandBuffer& cmdBuffer) {
            using namespace vks::util;
            setImageLayout(cmdBuffer, sharedResources.colorMap.image, ImageTransitionState::UNDEFINED, ImageTransitionState::GENERAL);
            setImageLayout(cmdBuffer, sharedResources.computed.image, ImageTransitionState::UNDEFINED, ImageTransitionState::GENERAL);
            cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, compute.releaseBarriers });
        });

        generateQuad();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void postRender() override {
        vks::frame::QueuedCommandBuilder builder{ compute.commandBuffers[currentIndex], vkx::RenderStates::COMPUTE_POST,
                                                  vk::PipelineStageFlagBits2::eComputeShader };
        builder.withQueueFamilyIndex(computeQueue.familyInfo.index);
        queueCommandBuffer(builder);
    }

    void viewChanged() override { updateUniformBuffers(); }

    void keyPressed(uint32_t keyCode) override {
        switch (keyCode) {
            case KEY_KPADD:
            case GAMEPAD_BUTTON_R1:
                compute.switchPipeline(1);
                break;
            case KEY_KPSUB:
            case GAMEPAD_BUTTON_L1:
                compute.switchPipeline(-1);
                break;
        }
    }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.comboBox("Shader", &compute.pipelineIndex, SHADER_NAMES)) {
                compute.buildCommandBuffers();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
