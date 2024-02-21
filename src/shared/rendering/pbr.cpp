#include "pbr.hpp"

#include <chrono>
#include <cmath>
#include <common/utils.hpp>
#include <iostream>
#include <rendering/context.hpp>
#include <rendering/offscreen.hpp>
#include <rendering/texture.hpp>
#include <vks/pipelines.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shaders/pbr/filtercube.vert.inl>
#include <shaders/pbr/genbrdflut.frag.inl>
#include <shaders/pbr/genbrdflut.vert.inl>
#include <shaders/pbr/irradiancecube.frag.inl>
#include <shaders/pbr/prefilterenvmap.frag.inl>

#ifndef M_PI
constexpr float M_PI = 3.14159265359f;
#endif

// Generate a BRDF integration map used as a look-up-table (stores roughness / NdotV)
void vkx::pbr::generateBRDFLUT(vks::texture::Texture2D& target) {
    constexpr vk::Format format = vk::Format::eR16G16Sfloat;  // R16G16 is supported pretty much everywhere
    constexpr int32_t dim = 512;
    constexpr vk::Extent2D extent{ dim, dim };

    const auto& context = vks::Context::get();
    const auto& device = context.device;
    auto tStart = std::chrono::high_resolution_clock::now();

    // Create the texture image and view
    {
        vks::Image::Builder builder{ dim };
        builder.withFormat(format);
        builder.withUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        target.build(builder);
        vks::debug::marker::setObjectName(device, target.image.image, "BRDF LUT");
        vks::debug::marker::setObjectName(device, target.imageView, "BRDF LUT");
    }

    // Desriptors
    vk::DescriptorSetLayout descriptorsetlayout = device.createDescriptorSetLayout({});
    // Descriptor Pool
    std::vector<vk::DescriptorPoolSize> poolSizes{ { vk::DescriptorType::eCombinedImageSampler, 1 } };
    vk::DescriptorPool descriptorpool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });

    // Descriptor sets
    vk::DescriptorSet descriptorset = device.allocateDescriptorSets({ descriptorpool, 1, &descriptorsetlayout })[0];

    // Pipeline layout
    vk::PipelineLayout pipelinelayout = device.createPipelineLayout({ {}, 1, &descriptorsetlayout });

    // Pipeline
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelinelayout };
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.depthStencilState = { false };
    // Look-up-table (from BRDF) pipeline
    pipelineBuilder.loadShader(vkx::shaders::pbr::genbrdflut::vert, vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::shaders::pbr::genbrdflut::frag, vk::ShaderStageFlagBits::eFragment);
    vk::Pipeline pipeline = pipelineBuilder.create(context.pipelineCache);

    // Render

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.imageView = target.imageView;
    colorAttachmentInfo.clearValue = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderingInfo renderingInfo;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.layerCount = 1;
    renderingInfo.renderArea.extent = extent;

    vks::Loader::get().withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuf) {
        using namespace vks::util;
        setImageLayout(cmdBuf, target.image, ImageTransitionState::UNDEFINED, ImageTransitionState::COLOR_ATTACHMENT);
        cmdBuf.beginRendering(renderingInfo);
        cmdBuf.setViewport(0, vks::util::viewport(extent));
        cmdBuf.setScissor(0, vks::util::rect2D(extent));
        cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuf.draw(3, 1, 0, 0);
        cmdBuf.endRendering();
        setImageLayout(cmdBuf, target.image, ImageTransitionState::COLOR_ATTACHMENT, ImageTransitionState::SAMPLED);
    });

    // todo: cleanup
    device.destroy(pipeline);
    device.destroy(pipelinelayout);
    device.destroy(descriptorsetlayout);
    device.destroy(descriptorpool);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
}

// Generate an irradiance cube map from the environment cube map
void vkx::pbr::generateIrradianceCube(vks::texture::TextureCubeMap& target,
                                      const vks::model::Model& skybox,
                                      const vks::model::VertexLayout& vertexLayout,
                                      const vk::DescriptorImageInfo& skyboxDescriptor) {
    auto tStart = std::chrono::high_resolution_clock::now();
    const auto& context = vks::Context::get();
    const auto& device = context.device;

    const vk::Format format = vk::Format::eR32G32B32A32Sfloat;
    const int32_t dim = 64;
    const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;
    auto extent = vk::Extent2D{ dim, dim };

    {
        vks::Image::Builder builder{ dim };
        builder.withFormat(format);
        builder.withFlags(vk::ImageCreateFlagBits::eCubeCompatible);
        builder.withArrayLayers(6);
        builder.withMipLevels(numMips);
        builder.withUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
        target.build(builder, vk::ImageViewType::eCube);
        vks::debug::marker::setObjectName(device, target.image.image, "Irradiance Cubemap");
        vks::debug::marker::setObjectName(device, target.imageView, "Irradiance Cubemap");
    }

    // Descriptors
    vk::DescriptorSetLayoutBinding setLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    vk::DescriptorSetLayout descriptorsetlayout = device.createDescriptorSetLayout({ {}, 1, &setLayoutBinding });

    // Descriptor Pool
    vk::DescriptorPoolSize poolSize{ vk::DescriptorType::eCombinedImageSampler, 1 };
    vk::DescriptorPool descriptorpool = device.createDescriptorPool({ {}, 2, 1, &poolSize });

    // Descriptor sets
    vk::DescriptorSet descriptorset = device.allocateDescriptorSets({ descriptorpool, 1, &descriptorsetlayout })[0];
    vk::WriteDescriptorSet writeDescriptorSet{ descriptorset, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &skyboxDescriptor };
    device.updateDescriptorSets(writeDescriptorSet, nullptr);

    // Pipeline layout
    struct PushBlock {
        glm::mat4 mvp;
        // Sampling deltas
        float deltaPhi = (2.0f * float(M_PI)) / 180.0f;
        float deltaTheta = (0.5f * float(M_PI)) / 64.0f;
    } pushBlock;
    vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushBlock) };

    vk::PipelineLayout pipelinelayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorsetlayout, 1, &pushConstantRange });

    // Pipeline
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelinelayout };
    pipelineBuilder.dynamicRendering(vk::Format::eR32G32B32A32Sfloat);
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.depthStencilState = { false };
    pipelineBuilder.vertexInputState.bindingDescriptions = {
        { 0, vertexLayout.stride(), vk::VertexInputRate::eVertex },
    };
    pipelineBuilder.vertexInputState.attributeDescriptions = {
        { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
    };

    pipelineBuilder.loadShader(vkx::shaders::pbr::filtercube::vert, vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::shaders::pbr::irradiancecube::frag, vk::ShaderStageFlagBits::eFragment);
    vk::Pipeline pipeline = pipelineBuilder.create(context.pipelineCache);

    const std::vector<glm::mat4> matrices = {
        // POSITIVE_X
        glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // NEGATIVE_X
        glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // POSITIVE_Y
        glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // NEGATIVE_Y
        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // POSITIVE_Z
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // NEGATIVE_Z
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    };

    const auto& loader = vks::Loader::get();
    // Offfscreen framebuffer
    vkx::offscreen::Renderer offscreen;
    {
        vkx::offscreen::Builder builder{ { dim, dim } };
        builder.appendColorFormat(format, vk::ImageUsageFlagBits::eTransferSrc);
        offscreen.prepare(builder);
    }

    loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuf) {
        vk::Viewport viewport{ 0, 0, (float)dim, (float)dim, 0.0f, 1.0f };
        vk::Rect2D scissor{ vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } };

        cmdBuf.setViewport(0, viewport);
        cmdBuf.setScissor(0, scissor);

        vk::ImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        subresourceRange.levelCount = numMips;
        subresourceRange.layerCount = 6;
        using namespace vks::util;

        // Change image layout for all cubemap faces to transfer destination
        setImageLayout(cmdBuf, target.image, ImageTransitionState::UNDEFINED, ImageTransitionState::TRANSFER_DST);

        for (uint32_t m = 0; m < numMips; m++) {
            for (uint32_t f = 0; f < 6; f++) {
                offscreen.setLayout(cmdBuf, ImageTransitionState::COLOR_ATTACHMENT);
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                cmdBuf.setViewport(0, 1, &viewport);
                // Render scene from cube face's point of view
                cmdBuf.beginRendering(offscreen.renderingInfo);
                // Update shader push constant block
                float fovy = (float)(M_PI / 2.0);
                float aspect = 1.0f;
                float nearClip = 0.1f;
                float farClip = 512.0f;
                auto perspective = glm::perspective(fovy, aspect, nearClip, farClip);
                pushBlock.mvp = perspective * matrices[f];
                cmdBuf.pushConstants<PushBlock>(pipelinelayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushBlock);
                cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
                cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descriptorset, nullptr);
                cmdBuf.bindVertexBuffers(0, skybox.vertices.buffer, { 0 });
                cmdBuf.bindIndexBuffer(skybox.indices.buffer, 0, vk::IndexType::eUint32);
                cmdBuf.drawIndexed(skybox.indexCount, 1, 0, 0, 0);
                cmdBuf.endRendering();
                offscreen.setLayout(cmdBuf, ImageTransitionState::TRANSFER_SRC);

                // Copy region for transfer from framebuffer to cube face
                vk::ImageCopy copyRegion;
                copyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                copyRegion.srcSubresource.baseArrayLayer = 0;
                copyRegion.srcSubresource.mipLevel = 0;
                copyRegion.srcSubresource.layerCount = 1;
                copyRegion.srcOffset = vk::Offset3D{};
                copyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                copyRegion.dstSubresource.baseArrayLayer = f;
                copyRegion.dstSubresource.mipLevel = m;
                copyRegion.dstSubresource.layerCount = 1;
                copyRegion.dstOffset = vk::Offset3D{};
                copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
                copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
                copyRegion.extent.depth = 1;

                cmdBuf.copyImage(                                                                 //
                    offscreen.colorTargets[0].image.image, vk::ImageLayout::eTransferSrcOptimal,  //
                    target.image.image, vk::ImageLayout::eTransferDstOptimal,                     //
                    copyRegion);
            }
        }
        setImageLayout(cmdBuf, target.image, ImageTransitionState::TRANSFER_DST, ImageTransitionState::SAMPLED);
    });

    // todo: cleanup
    offscreen.destroy();
    device.destroy(descriptorpool);
    device.destroy(descriptorsetlayout);
    device.destroy(pipeline);
    device.destroy(pipelinelayout);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating irradiance cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
}

// Prefilter environment cubemap
// See https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
void vkx::pbr::generatePrefilteredCube(vks::texture::TextureCubeMap& target,
                                       const vks::model::Model& skybox,
                                       const vks::model::VertexLayout& vertexLayout,
                                       const vk::DescriptorImageInfo& skyboxDescriptor) {
    auto tStart = std::chrono::high_resolution_clock::now();

    const auto& context = vks::Context::get();
    const auto& device = context.device;

    const vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    const int32_t dim = 512;
    const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

    // Pre-filtered cube map
    {
        vks::Image::Builder builder{ dim };
        builder.withFormat(format);
        builder.withFlags(vk::ImageCreateFlagBits::eCubeCompatible);
        builder.withArrayLayers(6);
        builder.withMipLevels(numMips);
        builder.withUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled);
        target.build(builder, vk::ImageViewType::eCube);
        vks::debug::marker::setObjectName(device, target.image.image, "Prefiltered Cubemap");
        vks::debug::marker::setObjectName(device, target.imageView, "Prefiltered Cubemap");
    }

    // Descriptors
    vk::DescriptorSetLayoutBinding setLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    vk::DescriptorSetLayout descriptorsetlayout = device.createDescriptorSetLayout({ {}, 1, &setLayoutBinding });
    // Descriptor Pool
    vk::DescriptorPoolSize poolSize{ vk::DescriptorType::eCombinedImageSampler, 1 };
    vk::DescriptorPool descriptorpool = device.createDescriptorPool({ {}, 2, 1, &poolSize });
    // Descriptor sets
    vk::DescriptorSet descriptorset = device.allocateDescriptorSets({ descriptorpool, 1, &descriptorsetlayout })[0];
    vk::WriteDescriptorSet writeDescriptorSet{ descriptorset, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &skyboxDescriptor };
    device.updateDescriptorSets(writeDescriptorSet, nullptr);

    // Pipeline layout
    struct PushBlock {
        glm::mat4 mvp;
        float roughness;
        uint32_t numSamples = 32u;
    } pushBlock;

    vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushBlock) };
    vk::PipelineLayout pipelinelayout = device.createPipelineLayout({ {}, 1, &descriptorsetlayout, 1, &pushConstantRange });

    // Pipeline
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelinelayout };
    pipelineBuilder.dynamicRendering(format);
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.depthStencilState = { false };
    pipelineBuilder.vertexInputState.bindingDescriptions = {
        { 0, vertexLayout.stride(), vk::VertexInputRate::eVertex },
    };
    pipelineBuilder.vertexInputState.attributeDescriptions = {
        { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
    };

    pipelineBuilder.loadShader(vkx::shaders::pbr::filtercube::vert, vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::shaders::pbr::prefilterenvmap::frag, vk::ShaderStageFlagBits::eFragment);
    vk::Pipeline pipeline = pipelineBuilder.create(context.pipelineCache);

    // Reuse render pass from example pass
    const std::vector<glm::mat4> matrices = {
        // POSITIVE_X
        glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // NEGATIVE_X
        glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // POSITIVE_Y
        glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // NEGATIVE_Y
        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // POSITIVE_Z
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        // NEGATIVE_Z
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    };

    const auto& loader = vks::Loader::get();

    // Offfscreen framebuffer
    vkx::offscreen::Renderer offscreen;
    {
        vkx::offscreen::Builder builder{ { dim, dim } };
        builder.appendColorFormat(format, vk::ImageUsageFlagBits::eTransferSrc, vk::ClearColorValue{ 0.0f, 0.0f, 0.2f, 0.0f });
        offscreen.prepare(builder);
    }

    loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuf) {
        vk::Viewport viewport{ 0, 0, (float)dim, (float)dim, 0, 1 };
        vk::Rect2D scissor{ vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } };
        cmdBuf.setViewport(0, viewport);
        cmdBuf.setScissor(0, scissor);
        vk::ImageSubresourceRange subresourceRange{ vk::ImageAspectFlagBits::eColor, 0, numMips, 0, 6 };
        // Change image layout for all cubemap faces to transfer destination
        using namespace vks::util;
        auto& offscreenImage = offscreen.colorTargets[0].image.image;
        auto& outputImage = target.image.image;
        setImageLayout(cmdBuf, target.image, ImageTransitionState::UNDEFINED, ImageTransitionState::TRANSFER_DST);
        for (uint32_t m = 0; m < numMips; m++) {
            pushBlock.roughness = (float)m / (float)(numMips - 1);
            for (uint32_t f = 0; f < 6; f++) {
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                cmdBuf.setViewport(0, viewport);
                offscreen.setLayout(cmdBuf, ImageTransitionState::COLOR_ATTACHMENT);
                // Render scene from cube face's point of view
                cmdBuf.beginRendering(offscreen.renderingInfo);

                // Update shader push constant block
                pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

                cmdBuf.pushConstants<PushBlock>(pipelinelayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushBlock);
                cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
                cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descriptorset, nullptr);

                std::vector<vk::DeviceSize> offsets{ 0 };
                cmdBuf.bindVertexBuffers(0, skybox.vertices.buffer, offsets);
                cmdBuf.bindIndexBuffer(skybox.indices.buffer, 0, vk::IndexType::eUint32);
                cmdBuf.drawIndexed(skybox.indexCount, 1, 0, 0, 0);

                cmdBuf.endRendering();
                offscreen.setLayout(cmdBuf, ImageTransitionState::TRANSFER_SRC);

                // Copy region for transfer from framebuffer to cube face
                vk::ImageCopy copyRegion;
                copyRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                copyRegion.srcSubresource.baseArrayLayer = 0;
                copyRegion.srcSubresource.mipLevel = 0;
                copyRegion.srcSubresource.layerCount = 1;
                copyRegion.srcOffset = vk::Offset3D{};

                copyRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                copyRegion.dstSubresource.baseArrayLayer = f;
                copyRegion.dstSubresource.mipLevel = m;
                copyRegion.dstSubresource.layerCount = 1;
                copyRegion.dstOffset = vk::Offset3D{};

                copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
                copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
                copyRegion.extent.depth = 1;
                cmdBuf.copyImage(offscreenImage, vk::ImageLayout::eTransferSrcOptimal, outputImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
            }
        }
        setImageLayout(cmdBuf, target.image, ImageTransitionState::TRANSFER_DST, ImageTransitionState::SAMPLED);
    });
    offscreen.destroy();
    device.destroy(descriptorpool, nullptr);
    device.destroy(descriptorsetlayout, nullptr);
    device.destroy(pipeline, nullptr);
    device.destroy(pipelinelayout, nullptr);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating pre-filtered enivornment cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
}
