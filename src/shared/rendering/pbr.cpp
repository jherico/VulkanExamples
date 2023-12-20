#include "pbr.hpp"

#include <chrono>
#include <iostream>
#include "vks/texture.hpp"
#include "vks/context.hpp"
#include "vks/pipelines.hpp"
#include "utils.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Generate a BRDF integration map used as a look-up-table (stores roughness / NdotV)
void vkx::pbr::generateBRDFLUT(const vks::Context& context, vks::texture::Texture2D& target) {
    auto tStart = std::chrono::high_resolution_clock::now();

    const vk::Format format = vk::Format::eR16G16Sfloat;  // R16G16 is supported pretty much everywhere
    const int32_t dim = 512;

    const auto& device = context.device;
    target.device = device;
    target.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    {
        // Image
        vk::ImageCreateInfo imageCI;
        imageCI.imageType = vk::ImageType::e2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = 1;
        imageCI.arrayLayers = 1;
        imageCI.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
        (vks::Image&)target = context.createImage(imageCI);
        // Image view
        vk::ImageViewCreateInfo viewCI;
        viewCI.viewType = vk::ImageViewType::e2D;
        viewCI.format = format;
        viewCI.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewCI.subresourceRange.levelCount = 1;
        viewCI.subresourceRange.layerCount = 1;
        viewCI.image = target.image;
        target.view = context.device.createImageView(viewCI);
        // Sampler
        vk::SamplerCreateInfo samplerCI;
        samplerCI.magFilter = vk::Filter::eLinear;
        samplerCI.minFilter = vk::Filter::eLinear;
        samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.maxLod = 1.0f;
        samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        target.sampler = device.createSampler(samplerCI);
        target.updateDescriptor();
    }

    // FB, Att, RP, Pipe, etc.
    vk::RenderPass renderpass;
    {
        vk::AttachmentDescription attDesc;
        // Color attachment
        attDesc.format = format;
        attDesc.loadOp = vk::AttachmentLoadOp::eClear;
        attDesc.storeOp = vk::AttachmentStoreOp::eStore;
        attDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attDesc.initialLayout = vk::ImageLayout::eUndefined;
        attDesc.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        vk::AttachmentReference colorReference{ 0, vk::ImageLayout::eColorAttachmentOptimal };

        vk::SubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Use subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 2> dependencies{
            vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                                   vk::DependencyFlagBits::eByRegion },
            vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead,
                                   vk::DependencyFlagBits::eByRegion },
        };

        // Create the actual renderpass
        vk::RenderPassCreateInfo renderPassCI;
        renderPassCI.attachmentCount = 1;
        renderPassCI.pAttachments = &attDesc;
        renderPassCI.subpassCount = 1;
        renderPassCI.pSubpasses = &subpassDescription;
        renderPassCI.dependencyCount = 2;
        renderPassCI.pDependencies = dependencies.data();
        renderpass = device.createRenderPass(renderPassCI);
    }

    vk::Framebuffer framebuffer;
    {
        vk::FramebufferCreateInfo framebufferCI;
        framebufferCI.renderPass = renderpass;
        framebufferCI.attachmentCount = 1;
        framebufferCI.pAttachments = &target.view;
        framebufferCI.width = dim;
        framebufferCI.height = dim;
        framebufferCI.layers = 1;
        framebuffer = device.createFramebuffer(framebufferCI);
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
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelinelayout, renderpass };
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.depthStencilState = { false };
    // Look-up-table (from BRDF) pipeline
    pipelineBuilder.loadShader(vkx::getAssetPath() + "shaders/pbr/genbrdflut.vert.spv", vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::getAssetPath() + "shaders/pbr/genbrdflut.frag.spv", vk::ShaderStageFlagBits::eFragment);
    vk::Pipeline pipeline = pipelineBuilder.create(context.pipelineCache);

    // Render
    vk::ClearValue clearValues[1];
    clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.0f, 1.0f });

    vk::RenderPassBeginInfo renderPassBeginInfo;
    renderPassBeginInfo.renderPass = renderpass;
    renderPassBeginInfo.renderArea.extent.width = dim;
    renderPassBeginInfo.renderArea.extent.height = dim;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = clearValues;
    renderPassBeginInfo.framebuffer = framebuffer;

    context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuf) {
        cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        vk::Viewport viewport{ 0, 0, (float)dim, (float)dim, 0, 1 };
        vk::Rect2D scissor{ vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } };
        cmdBuf.setViewport(0, viewport);
        cmdBuf.setScissor(0, scissor);
        cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuf.draw(3, 1, 0, 0);
        cmdBuf.endRenderPass();
    });
    context.queue.waitIdle();

    // todo: cleanup
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipelinelayout);
    device.destroyRenderPass(renderpass);
    device.destroyFramebuffer(framebuffer);
    device.destroyDescriptorSetLayout(descriptorsetlayout);
    device.destroyDescriptorPool(descriptorpool);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
}

// Generate an irradiance cube map from the environment cube map
void vkx::pbr::generateIrradianceCube(const vks::Context& context,
                                      vks::texture::Texture& target,
                                      const vks::model::Model& skybox,
                                      const vks::model::VertexLayout& vertexLayout,
                                      const vk::DescriptorImageInfo& skyboxDescriptor) {
    auto tStart = std::chrono::high_resolution_clock::now();

    const auto& device = context.device;
    target.device = device;

    const vk::Format format = vk::Format::eR32G32B32A32Sfloat;
    const int32_t dim = 64;
    const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

    {
        // Pre-filtered cube map
        // Image
        vk::ImageCreateInfo imageCI;
        imageCI.imageType = vk::ImageType::e2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = numMips;
        imageCI.arrayLayers = 6;
        imageCI.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        imageCI.flags = vk::ImageCreateFlagBits::eCubeCompatible;

        target = context.createImage(imageCI);

        // Image view
        vk::ImageViewCreateInfo viewCI;
        viewCI.viewType = vk::ImageViewType::eCube;
        viewCI.format = format;
        viewCI.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewCI.subresourceRange.levelCount = numMips;
        viewCI.subresourceRange.layerCount = 6;
        viewCI.image = target.image;
        target.view = device.createImageView(viewCI);
        // Sampler
        vk::SamplerCreateInfo samplerCI;
        samplerCI.magFilter = vk::Filter::eLinear;
        samplerCI.minFilter = vk::Filter::eLinear;
        samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.maxLod = static_cast<float>(numMips);
        samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        target.sampler = device.createSampler(samplerCI);
        target.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        target.updateDescriptor();
    }

    vk::RenderPass renderpass;
    {
        // FB, Att, RP, Pipe, etc.
        vk::AttachmentDescription attDesc = {};
        // Color attachment
        attDesc.format = format;
        attDesc.loadOp = vk::AttachmentLoadOp::eClear;
        attDesc.storeOp = vk::AttachmentStoreOp::eStore;
        attDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attDesc.initialLayout = vk::ImageLayout::eUndefined;
        attDesc.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        vk::AttachmentReference colorReference{ 0, vk::ImageLayout::eColorAttachmentOptimal };
        vk::SubpassDescription subpassDescription{ {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorReference };

        // Use subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 2> dependencies{
            vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                                   vk::DependencyFlagBits::eByRegion },
            vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead,
                                   vk::DependencyFlagBits::eByRegion },
        };

        // Renderpass
        vk::RenderPassCreateInfo renderPassCI;
        renderPassCI.attachmentCount = 1;
        renderPassCI.pAttachments = &attDesc;
        renderPassCI.subpassCount = 1;
        renderPassCI.pSubpasses = &subpassDescription;
        renderPassCI.dependencyCount = 2;
        renderPassCI.pDependencies = dependencies.data();

        renderpass = device.createRenderPass(renderPassCI);
    }

    struct {
        vks::Image image;
        vk::Framebuffer framebuffer;
    } offscreen;

    // Offfscreen framebuffer
    {
        // Color attachment
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent.width = dim;
        imageCreateInfo.extent.height = dim;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
        offscreen.image = context.createImage(imageCreateInfo);

        vk::ImageViewCreateInfo colorImageView;
        colorImageView.viewType = vk::ImageViewType::e2D;
        colorImageView.format = format;
        colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorImageView.subresourceRange.levelCount = 1;
        colorImageView.subresourceRange.layerCount = 1;
        colorImageView.image = offscreen.image.image;
        offscreen.image.view = device.createImageView(colorImageView);

        vk::FramebufferCreateInfo fbufCreateInfo;
        fbufCreateInfo.renderPass = renderpass;
        fbufCreateInfo.attachmentCount = 1;
        fbufCreateInfo.pAttachments = &offscreen.image.view;
        fbufCreateInfo.width = dim;
        fbufCreateInfo.height = dim;
        fbufCreateInfo.layers = 1;
        offscreen.framebuffer = device.createFramebuffer(fbufCreateInfo);
        context.setImageLayout(offscreen.image.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
    }

    // Descriptors
    vk::DescriptorSetLayout descriptorsetlayout;
    vk::DescriptorSetLayoutBinding setLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment };
    descriptorsetlayout = device.createDescriptorSetLayout({ {}, 1, &setLayoutBinding });

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
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelinelayout, renderpass };
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.depthStencilState = { false };
    pipelineBuilder.vertexInputState.bindingDescriptions = {
        { 0, vertexLayout.stride(), vk::VertexInputRate::eVertex },
    };
    pipelineBuilder.vertexInputState.attributeDescriptions = {
        { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
    };

    pipelineBuilder.loadShader(vkx::getAssetPath() + "shaders/pbr/filtercube.vert.spv", vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::getAssetPath() + "shaders/pbr/irradiancecube.frag.spv", vk::ShaderStageFlagBits::eFragment);
    vk::Pipeline pipeline = pipelineBuilder.create(context.pipelineCache);

    // Render
    vk::ClearValue clearValues[1];
    clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.2f, 0.0f });

    vk::RenderPassBeginInfo renderPassBeginInfo{ renderpass, offscreen.framebuffer, vk::Rect2D{ vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } },
                                                 1, clearValues };

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

    context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuf) {
        vk::Viewport viewport{ 0, 0, (float)dim, (float)dim, 0.0f, 1.0f };
        vk::Rect2D scissor{ vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } };

        cmdBuf.setViewport(0, viewport);
        cmdBuf.setScissor(0, scissor);

        vk::ImageSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        subresourceRange.levelCount = numMips;
        subresourceRange.layerCount = 6;

        // Change image layout for all cubemap faces to transfer destination
        context.setImageLayout(cmdBuf, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, subresourceRange);

        for (uint32_t m = 0; m < numMips; m++) {
            for (uint32_t f = 0; f < 6; f++) {
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                cmdBuf.setViewport(0, 1, &viewport);
                // Render scene from cube face's point of view
                cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

                // Update shader push constant block
                pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

                cmdBuf.pushConstants<PushBlock>(pipelinelayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushBlock);

                cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
                cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descriptorset, nullptr);

                std::vector<vk::DeviceSize> offsets{ 0 };

                cmdBuf.bindVertexBuffers(0, skybox.vertices.buffer, offsets);
                cmdBuf.bindIndexBuffer(skybox.indices.buffer, 0, vk::IndexType::eUint32);
                cmdBuf.drawIndexed(skybox.indexCount, 1, 0, 0, 0);

                cmdBuf.endRenderPass();

                context.setImageLayout(cmdBuf, offscreen.image.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
                                       vk::ImageLayout::eTransferSrcOptimal);

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

                cmdBuf.copyImage(offscreen.image.image, vk::ImageLayout::eTransferSrcOptimal, target.image, vk::ImageLayout::eTransferDstOptimal, copyRegion);

                // Transform framebuffer color attachment back
                context.setImageLayout(cmdBuf, offscreen.image.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
                                       vk::ImageLayout::eColorAttachmentOptimal);
            }
        }
        context.setImageLayout(cmdBuf, target.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, subresourceRange);
    });

    // todo: cleanup
    device.destroyRenderPass(renderpass, nullptr);
    device.destroyFramebuffer(offscreen.framebuffer, nullptr);
    offscreen.image.destroy();
    device.destroyDescriptorPool(descriptorpool, nullptr);
    device.destroyDescriptorSetLayout(descriptorsetlayout, nullptr);
    device.destroyPipeline(pipeline, nullptr);
    device.destroyPipelineLayout(pipelinelayout, nullptr);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating irradiance cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
}

// Prefilter environment cubemap
// See https://placeholderart.wordpress.com/2015/07/28/implementation-notes-runtime-environment-map-filtering-for-image-based-lighting/
void vkx::pbr::generatePrefilteredCube(const vks::Context& context,
                                       vks::texture::Texture& target,
                                       const vks::model::Model& skybox,
                                       const vks::model::VertexLayout& vertexLayout,
                                       const vk::DescriptorImageInfo& skyboxDescriptor) {
    auto tStart = std::chrono::high_resolution_clock::now();

    const auto& device = context.device;
    target.device = device;
    target.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    const vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    const int32_t dim = 512;
    const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

    // Pre-filtered cube map
    // Image
    {
        vk::ImageCreateInfo imageCI;
        imageCI.imageType = vk::ImageType::e2D;
        imageCI.format = format;
        imageCI.extent.width = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth = 1;
        imageCI.mipLevels = numMips;
        imageCI.arrayLayers = 6;
        imageCI.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
        imageCI.flags = vk::ImageCreateFlagBits::eCubeCompatible;
        target = context.createImage(imageCI);
        // Image view
        vk::ImageViewCreateInfo viewCI;
        viewCI.viewType = vk::ImageViewType::eCube;
        viewCI.format = format;
        viewCI.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        viewCI.subresourceRange.levelCount = numMips;
        viewCI.subresourceRange.layerCount = 6;
        viewCI.image = target.image;
        target.view = device.createImageView(viewCI);
        // Sampler
        vk::SamplerCreateInfo samplerCI;
        samplerCI.magFilter = vk::Filter::eLinear;
        samplerCI.minFilter = vk::Filter::eLinear;
        samplerCI.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerCI.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        samplerCI.maxLod = static_cast<float>(numMips);
        samplerCI.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        target.sampler = device.createSampler(samplerCI);
        target.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        target.updateDescriptor();
    }

    vk::RenderPass renderpass;
    {
        // FB, Att, RP, Pipe, etc.
        vk::AttachmentDescription attDesc = {};
        // Color attachment
        attDesc.format = format;
        attDesc.loadOp = vk::AttachmentLoadOp::eClear;
        attDesc.storeOp = vk::AttachmentStoreOp::eStore;
        attDesc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        attDesc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        attDesc.initialLayout = vk::ImageLayout::eUndefined;
        attDesc.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;
        vk::AttachmentReference colorReference{ 0, vk::ImageLayout::eColorAttachmentOptimal };

        vk::SubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;

        // Use subpass dependencies for layout transitions
        std::array<vk::SubpassDependency, 2> dependencies{
            vk::SubpassDependency{ VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                   vk::AccessFlagBits::eMemoryRead, vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                                   vk::DependencyFlagBits::eByRegion },
            vk::SubpassDependency{ 0, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eMemoryRead,
                                   vk::DependencyFlagBits::eByRegion },
        };

        // Renderpass
        renderpass = device.createRenderPass({ {}, 1, &attDesc, 1, &subpassDescription, (uint32_t)dependencies.size(), dependencies.data() });
    }

    struct {
        vks::Image image;
        vk::Framebuffer framebuffer;
    } offscreen;

    // Offfscreen framebuffer
    {
        // Color attachment
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent.width = dim;
        imageCreateInfo.extent.height = dim;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
        offscreen.image = context.createImage(imageCreateInfo);

        vk::ImageViewCreateInfo colorImageView;
        colorImageView.viewType = vk::ImageViewType::e2D;
        colorImageView.format = format;
        colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorImageView.subresourceRange.levelCount = 1;
        colorImageView.subresourceRange.layerCount = 1;
        colorImageView.image = offscreen.image.image;
        offscreen.image.view = device.createImageView(colorImageView);

        vk::FramebufferCreateInfo fbufCreateInfo;
        fbufCreateInfo.renderPass = renderpass;
        fbufCreateInfo.attachmentCount = 1;
        fbufCreateInfo.pAttachments = &offscreen.image.view;
        fbufCreateInfo.width = dim;
        fbufCreateInfo.height = dim;
        fbufCreateInfo.layers = 1;
        offscreen.framebuffer = device.createFramebuffer(fbufCreateInfo);
        context.setImageLayout(offscreen.image.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
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
    vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelinelayout, renderpass };
    pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
    pipelineBuilder.depthStencilState = { false };
    pipelineBuilder.vertexInputState.bindingDescriptions = {
        { 0, vertexLayout.stride(), vk::VertexInputRate::eVertex },
    };
    pipelineBuilder.vertexInputState.attributeDescriptions = {
        { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
    };

    pipelineBuilder.loadShader(vkx::getAssetPath() + "shaders/pbr/filtercube.vert.spv", vk::ShaderStageFlagBits::eVertex);
    pipelineBuilder.loadShader(vkx::getAssetPath() + "shaders/pbr/prefilterenvmap.frag.spv", vk::ShaderStageFlagBits::eFragment);
    vk::Pipeline pipeline = pipelineBuilder.create(context.pipelineCache);

    // Render

    vk::ClearValue clearValues[1];
    clearValues[0].color = vks::util::clearColor({ 0.0f, 0.0f, 0.2f, 0.0f });

    vk::RenderPassBeginInfo renderPassBeginInfo{ renderpass,
                                                 offscreen.framebuffer,
                                                 { vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } },
                                                 1,
                                                 clearValues };
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

    context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuf) {
        vk::Viewport viewport{ 0, 0, (float)dim, (float)dim, 0, 1 };
        vk::Rect2D scissor{ vk::Offset2D{}, vk::Extent2D{ (uint32_t)dim, (uint32_t)dim } };
        cmdBuf.setViewport(0, viewport);
        cmdBuf.setScissor(0, scissor);
        vk::ImageSubresourceRange subresourceRange{ vk::ImageAspectFlagBits::eColor, 0, numMips, 0, 6 };
        // Change image layout for all cubemap faces to transfer destination
        context.setImageLayout(cmdBuf, target.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, subresourceRange);

        for (uint32_t m = 0; m < numMips; m++) {
            pushBlock.roughness = (float)m / (float)(numMips - 1);
            for (uint32_t f = 0; f < 6; f++) {
                viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
                viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
                cmdBuf.setViewport(0, viewport);

                // Render scene from cube face's point of view
                cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

                // Update shader push constant block
                pushBlock.mvp = glm::perspective((float)(M_PI / 2.0), 1.0f, 0.1f, 512.0f) * matrices[f];

                cmdBuf.pushConstants<PushBlock>(pipelinelayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushBlock);
                cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
                cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelinelayout, 0, descriptorset, nullptr);

                std::vector<vk::DeviceSize> offsets{ 0 };
                cmdBuf.bindVertexBuffers(0, skybox.vertices.buffer, offsets);
                cmdBuf.bindIndexBuffer(skybox.indices.buffer, 0, vk::IndexType::eUint32);
                cmdBuf.drawIndexed(skybox.indexCount, 1, 0, 0, 0);

                cmdBuf.endRenderPass();

                context.setImageLayout(cmdBuf, offscreen.image.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eColorAttachmentOptimal,
                                       vk::ImageLayout::eTransferSrcOptimal);

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

                cmdBuf.copyImage(offscreen.image.image, vk::ImageLayout::eTransferSrcOptimal, target.image, vk::ImageLayout::eTransferDstOptimal, copyRegion);
                // Transform framebuffer color attachment back
                context.setImageLayout(cmdBuf, offscreen.image.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferSrcOptimal,
                                       vk::ImageLayout::eColorAttachmentOptimal);
            }
        }
        context.setImageLayout(cmdBuf, target.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, subresourceRange);
    });

    // todo: cleanup
    device.destroyRenderPass(renderpass, nullptr);
    device.destroyFramebuffer(offscreen.framebuffer, nullptr);
    offscreen.image.destroy();
    device.destroyDescriptorPool(descriptorpool, nullptr);
    device.destroyDescriptorSetLayout(descriptorsetlayout, nullptr);
    device.destroyPipeline(pipeline, nullptr);
    device.destroyPipelineLayout(pipelinelayout, nullptr);

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating pre-filtered enivornment cube with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
}
