/*
* Vulkan Example - Example for VK_EXT_debug_marker extension. To be used in conjuction with a debugging app like RenderDoc (https://renderdoc.org)
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanExampleBase.h"

// FIXME validation layers seem to crash when debug markers are enabled
// FIXME renderdoc crashes when using Vulkan 1.1


// Offscreen properties
#define OFFSCREEN_DIM 256
#define OFFSCREEN_FORMAT  vk::Format::eR8G8B8A8Unorm
#define OFFSCREEN_FILTER vk::Filter::eLinear;

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

// Extension spec can be found at https://github.com/KhronosGroup/Vulkan-Docs/blob/1.0-VK_EXT_debug_marker/doc/specs/vulkan/appendices/VK_EXT_debug_marker.txt
// Note that the extension will only be present if run from an offline debugging application
// The actual check for extension presence and enabling it on the device is done in the example base class
// See ExampleBase::createInstance and ExampleBase::createDevice (base/vkx::ExampleBase.cpp)
namespace DebugMarker {
    bool active = false;

    PFN_vkDebugMarkerSetObjectTagEXT pfnDebugMarkerSetObjectTag = VK_NULL_HANDLE;
    PFN_vkDebugMarkerSetObjectNameEXT pfnDebugMarkerSetObjectName = VK_NULL_HANDLE;
    PFN_vkCmdDebugMarkerBeginEXT pfnCmdDebugMarkerBegin = VK_NULL_HANDLE;
    PFN_vkCmdDebugMarkerEndEXT pfnCmdDebugMarkerEnd = VK_NULL_HANDLE;
    PFN_vkCmdDebugMarkerInsertEXT pfnCmdDebugMarkerInsert = VK_NULL_HANDLE;

    // Get function pointers for the debug report extensions from the device
    void setup(VkDevice device) {
        pfnDebugMarkerSetObjectTag = (PFN_vkDebugMarkerSetObjectTagEXT)vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectTagEXT");
        pfnDebugMarkerSetObjectName = (PFN_vkDebugMarkerSetObjectNameEXT)vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT");
        pfnCmdDebugMarkerBegin = (PFN_vkCmdDebugMarkerBeginEXT)vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT");
        pfnCmdDebugMarkerEnd = (PFN_vkCmdDebugMarkerEndEXT)vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT");
        pfnCmdDebugMarkerInsert = (PFN_vkCmdDebugMarkerInsertEXT)vkGetDeviceProcAddr(device, "vkCmdDebugMarkerInsertEXT");

        // Set flag if at least one function pointer is present
        active = (pfnDebugMarkerSetObjectName != VK_NULL_HANDLE);
        //active = false;
    }

    // Sets the debug name of an object
    // All Objects in Vulkan are represented by their 64-bit handles which are passed into this function
    // along with the object type
    void setObjectName(VkDevice device, uint64_t object, VkDebugReportObjectTypeEXT objectType, const char *name) {
        // Check for valid function pointer (may not be present if not running in a debugging application)
        if (active && pfnDebugMarkerSetObjectName) {
            VkDebugMarkerObjectNameInfoEXT nameInfo = {};
            nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType = objectType;
            nameInfo.object = object;
            nameInfo.pObjectName = name;
            pfnDebugMarkerSetObjectName(device, &nameInfo);
        }
    }

    // Set the tag for an object
    void setObjectTag(VkDevice device, uint64_t object, VkDebugReportObjectTypeEXT objectType, uint64_t name, size_t tagSize, const void* tag) {
        // Check for valid function pointer (may not be present if not running in a debugging application)
        if (active && pfnDebugMarkerSetObjectTag) {
            VkDebugMarkerObjectTagInfoEXT tagInfo = {};
            tagInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT;
            tagInfo.objectType = objectType;
            tagInfo.object = object;
            tagInfo.tagName = name;
            tagInfo.tagSize = tagSize;
            tagInfo.pTag = tag;
            pfnDebugMarkerSetObjectTag(device, &tagInfo);
        }
    }

    // Start a new debug marker region
    void beginRegion(VkCommandBuffer cmdbuffer, const char* pMarkerName, glm::vec4 color) {
        // Check for valid function pointer (may not be present if not running in a debugging application)
        if (active && pfnCmdDebugMarkerBegin) {
            VkDebugMarkerMarkerInfoEXT markerInfo = {};
            markerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
            memcpy(markerInfo.color, &color[0], sizeof(float) * 4);
            markerInfo.pMarkerName = pMarkerName;
            pfnCmdDebugMarkerBegin(cmdbuffer, &markerInfo);
        }
    }

    // Insert a new debug marker into the command buffer
    void insert(VkCommandBuffer cmdbuffer, std::string markerName, glm::vec4 color) {
        // Check for valid function pointer (may not be present if not running in a debugging application)
        if (active && pfnCmdDebugMarkerInsert) {
            VkDebugMarkerMarkerInfoEXT markerInfo = {};
            markerInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
            memcpy(markerInfo.color, &color[0], sizeof(float) * 4);
            markerInfo.pMarkerName = markerName.c_str();
            pfnCmdDebugMarkerInsert(cmdbuffer, &markerInfo);
        }
    }

    // End the current debug marker region
    void endRegion(VkCommandBuffer cmdBuffer) {
        // Check for valid function (may not be present if not runnin in a debugging application)
        if (active && pfnCmdDebugMarkerEnd) {
            pfnCmdDebugMarkerEnd(cmdBuffer);
        }
    }
};
// Vertex layout used in this example
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color;
};

class VulkanExample : public vkx::ExampleBase {
public:
    bool wireframe = true;
    bool glow = true;

    struct {
        vks::model::Model scene;
        vks::model::Model sceneGlow;
    } meshes;

    static void drawMesh(const vk::CommandBuffer& cmdBuffer, const vks::model::Model& model) {
        const auto& vertices = model.vertices;
        const auto& indices = model.indices;
        const auto& meshes = model.parts;
        vk::DeviceSize offsets = 0;

        cmdBuffer.bindVertexBuffers(0, vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(indices.buffer, 0, vk::IndexType::eUint32);
        for (auto mesh : meshes) {
            // Add debug marker for mesh name
            DebugMarker::insert(cmdBuffer, "Draw \"" + mesh.name + "\"", glm::vec4(0.0f));
            cmdBuffer.drawIndexed(mesh.indexCount, 1, mesh.indexBase, 0, 0);
        }
    }
    struct {
        vks::Buffer vsScene;
    } uniformData;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(0.0f, 5.0f, 15.0f, 1.0f);
    } uboVS;

    struct {
        vk::Pipeline toonshading;
        vk::Pipeline color;
        vk::Pipeline wireframe;
        vk::Pipeline postprocess;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    struct {
        vk::DescriptorSet scene;
        vk::DescriptorSet fullscreen;
    } descriptorSets;

    // vk::Framebuffer for offscreen rendering

    struct FrameBuffer {
        int32_t width, height;
        vk::Framebuffer framebuffer;
        vks::Image color, depth;
        vks::Image textureTarget;
    } offscreenFrameBuf;

    vk::Semaphore offscreenSemaphore;
    vk::CommandBuffer offscreenCmdBuffer;

    // Random tag data
    struct {
        const char name[17] = "debug marker tag";
    } demoTag;

    VulkanExample() {
        // current debugging tools don't yet work with Vulkan 1.1, so target 1.0
        // FIXME when RenderDoc works with 1.1, update this
        version = VK_MAKE_VERSION(1, 0, 0);
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ -4.35f, 16.25f, 0.0f });
        camera.setTranslation({ 0.1f, 1.1f, -8.5f });
        title = "Vulkan Example - VK_EXT_debug_marker";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources 
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.toonshading);
        device.destroyPipeline(pipelines.color);
        device.destroyPipeline(pipelines.wireframe);
        device.destroyPipeline(pipelines.postprocess);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // Destroy and free mesh resources 
        meshes.scene.destroy();
        meshes.sceneGlow.destroy();

        uniformData.vsScene.destroy();

        // Offscreen
        // Texture target
        offscreenFrameBuf.textureTarget.destroy();
        // Frame buffer
        device.destroyFramebuffer(offscreenFrameBuf.framebuffer);
        // Color attachment
        offscreenFrameBuf.color.destroy();
        // Depth attachment
        offscreenFrameBuf.depth.destroy();
    }

    // Prepare a texture target and framebuffer for offscreen rendering
    void prepareOffscreen() {
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuffer) {
            vk::FormatProperties formatProperties;

            // Get device properites for the requested texture format
            formatProperties = context.physicalDevice.getFormatProperties(OFFSCREEN_FORMAT);
            // Check if blit destination is supported for the requested format
            // Only try for optimal tiling, linear tiling usually won't support blit as destination anyway
            assert(formatProperties.optimalTilingFeatures &  vk::FormatFeatureFlagBits::eBlitDst);
            // Texture target
            auto& tex = offscreenFrameBuf.textureTarget;

            // Prepare blit target texture
            tex.extent.width = OFFSCREEN_DIM;
            tex.extent.height = OFFSCREEN_DIM;
            vk::ImageCreateInfo imageCreateInfo;
            imageCreateInfo.imageType = vk::ImageType::e2D;
            imageCreateInfo.format = OFFSCREEN_FORMAT;
            imageCreateInfo.extent = vk::Extent3D{ OFFSCREEN_DIM, OFFSCREEN_DIM, 1 };
            imageCreateInfo.mipLevels = 1;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
            imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
            imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;
            // Texture will be sampled in a shader and is also the blit destination
            imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
            offscreenFrameBuf.textureTarget = context.createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);

            // Transform image layout to transfer destination
            context.setImageLayout(
                cmdBuffer,
                tex.image,
                vk::ImageAspectFlagBits::eColor,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eShaderReadOnlyOptimal);

            // Create sampler
            vk::SamplerCreateInfo sampler;
            sampler.magFilter = OFFSCREEN_FILTER;
            sampler.minFilter = OFFSCREEN_FILTER;
            sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
            sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
            sampler.addressModeV = sampler.addressModeU;
            sampler.addressModeW = sampler.addressModeU;
            sampler.mipLodBias = 0.0f;
            sampler.maxAnisotropy = 0;
            sampler.compareOp = vk::CompareOp::eNever;
            sampler.minLod = 0.0f;
            sampler.maxLod = 0.0f;
            sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
            tex.sampler = device.createSampler(sampler);

            // Create image view
            vk::ImageViewCreateInfo view;
            view.viewType = vk::ImageViewType::e2D;
            view.format = OFFSCREEN_FORMAT;
            view.components = { vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA };
            view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
            view.image = tex.image;
            tex.view = device.createImageView(view);

            // Name for debugging
            DebugMarker::setObjectName(device, (uint64_t)(VkImage)tex.image, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, "Off-screen texture target image");
            DebugMarker::setObjectName(device, (uint64_t)(VkSampler)tex.sampler, VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT, "Off-screen texture target sampler");

            // Frame buffer
            offscreenFrameBuf.width = OFFSCREEN_DIM;
            offscreenFrameBuf.height = OFFSCREEN_DIM;

            // Find a suitable depth format
            vk::Format fbDepthFormat = context.getSupportedDepthFormat();

            // Color attachment
            vk::ImageCreateInfo image;
            image.imageType = vk::ImageType::e2D;
            image.format = OFFSCREEN_FORMAT;
            image.extent.width = offscreenFrameBuf.width;
            image.extent.height = offscreenFrameBuf.height;
            image.extent.depth = 1;
            image.mipLevels = 1;
            image.arrayLayers = 1;
            image.samples = vk::SampleCountFlagBits::e1;
            image.tiling = vk::ImageTiling::eOptimal;
            // vk::Image of the framebuffer is blit source
            image.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;

            vk::ImageViewCreateInfo colorImageView;
            colorImageView.viewType = vk::ImageViewType::e2D;
            colorImageView.format = OFFSCREEN_FORMAT;
            colorImageView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            colorImageView.subresourceRange.levelCount = 1;
            colorImageView.subresourceRange.layerCount = 1;

            offscreenFrameBuf.color = context.createImage(image, vk::MemoryPropertyFlagBits::eDeviceLocal);


            context.setImageLayout(
                cmdBuffer,
                offscreenFrameBuf.color.image,
                vk::ImageAspectFlagBits::eColor,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal);

            colorImageView.image = offscreenFrameBuf.color.image;
            offscreenFrameBuf.color.view = device.createImageView(colorImageView);

            // Depth stencil attachment
            image.format = fbDepthFormat;
            image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

            vk::ImageViewCreateInfo depthStencilView;
            depthStencilView.viewType = vk::ImageViewType::e2D;
            depthStencilView.format = fbDepthFormat;
            depthStencilView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
            depthStencilView.subresourceRange.levelCount = 1;
            depthStencilView.subresourceRange.layerCount = 1;

            offscreenFrameBuf.depth = context.createImage(image, vk::MemoryPropertyFlagBits::eDeviceLocal);

            context.setImageLayout(
                cmdBuffer,
                offscreenFrameBuf.depth.image,
                vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eDepthStencilAttachmentOptimal);

            depthStencilView.image = offscreenFrameBuf.depth.image;
            offscreenFrameBuf.depth.view = device.createImageView(depthStencilView);

            vk::ImageView attachments[2];
            attachments[0] = offscreenFrameBuf.color.view;
            attachments[1] = offscreenFrameBuf.depth.view;

            vk::FramebufferCreateInfo fbufCreateInfo;
            fbufCreateInfo.renderPass = renderPass;
            fbufCreateInfo.attachmentCount = 2;
            fbufCreateInfo.pAttachments = attachments;
            fbufCreateInfo.width = offscreenFrameBuf.width;
            fbufCreateInfo.height = offscreenFrameBuf.height;
            fbufCreateInfo.layers = 1;
            offscreenFrameBuf.framebuffer = device.createFramebuffer(fbufCreateInfo);
        });

        // Command buffer for offscreen rendering
        offscreenCmdBuffer = context.createCommandBuffer(vk::CommandBufferLevel::ePrimary);

        // Name for debugging
        DebugMarker::setObjectName(device, (uint64_t)(VkImage)offscreenFrameBuf.color.image, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, "Off-screen color framebuffer");
        DebugMarker::setObjectName(device, (uint64_t)(VkImage)offscreenFrameBuf.depth.image, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, "Off-screen depth framebuffer");
    }

    // Command buffer for rendering color only scene for glow
    void buildOffscreenCommandBuffer() {
        vk::CommandBufferBeginInfo cmdBufInfo;
        cmdBufInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;

        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor(glm::vec4(0));
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = offscreenFrameBuf.framebuffer;
        renderPassBeginInfo.renderArea.extent.width = offscreenFrameBuf.width;
        renderPassBeginInfo.renderArea.extent.height = offscreenFrameBuf.height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        offscreenCmdBuffer.begin(cmdBufInfo);

        // Start a new debug marker region
        DebugMarker::beginRegion(offscreenCmdBuffer, "Off-screen scene rendering", glm::vec4(1.0f, 0.78f, 0.05f, 1.0f));

        vk::Viewport viewport = vks::util::viewport((float)offscreenFrameBuf.width, (float)offscreenFrameBuf.height, 0.0f, 1.0f);
        offscreenCmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor = vks::util::rect2D(offscreenFrameBuf.width, offscreenFrameBuf.height, 0, 0);
        offscreenCmdBuffer.setScissor(0, scissor);

        offscreenCmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

        offscreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.scene, nullptr);
        offscreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.color);

        // Draw glow scene
        drawMesh(offscreenCmdBuffer, meshes.sceneGlow);

        offscreenCmdBuffer.endRenderPass();

        // Make sure color writes to the framebuffer are finished before using it as transfer source
        context.setImageLayout(
            offscreenCmdBuffer,
            offscreenFrameBuf.color.image,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eTransferSrcOptimal);

        // Transform texture target to transfer destination
        context.setImageLayout(
            offscreenCmdBuffer,
            offscreenFrameBuf.textureTarget.image,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eTransferDstOptimal);

        // Blit offscreen color buffer to our texture target
        vk::ImageBlit imgBlit;

        imgBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        imgBlit.srcSubresource.layerCount = 1;

        imgBlit.srcOffsets[1].x = offscreenFrameBuf.width;
        imgBlit.srcOffsets[1].y = offscreenFrameBuf.height;
        imgBlit.srcOffsets[1].z = 1;

        imgBlit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        imgBlit.dstSubresource.layerCount = 1;

        imgBlit.dstOffsets[1].x = offscreenFrameBuf.textureTarget.extent.width;
        imgBlit.dstOffsets[1].y = offscreenFrameBuf.textureTarget.extent.height;
        imgBlit.dstOffsets[1].z = 1;

        // Blit from framebuffer image to texture image
        // vkCmdBlitImage does scaling and (if necessary and possible) also does format conversions
        offscreenCmdBuffer.blitImage(offscreenFrameBuf.color.image, vk::ImageLayout::eTransferSrcOptimal, offscreenFrameBuf.textureTarget.image, vk::ImageLayout::eTransferDstOptimal, imgBlit, vk::Filter::eLinear);

        // Transform framebuffer color attachment back 
        context.setImageLayout(
            offscreenCmdBuffer,
            offscreenFrameBuf.color.image,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageLayout::eTransferSrcOptimal,
            vk::ImageLayout::eColorAttachmentOptimal);

        // Transform texture target back to shader read
        // Makes sure that writes to the texture are finished before
        // it's accessed in the shader
        context.setImageLayout(
            offscreenCmdBuffer,
            offscreenFrameBuf.textureTarget.image,
            vk::ImageAspectFlagBits::eColor,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal);

        DebugMarker::endRegion(offscreenCmdBuffer);

        offscreenCmdBuffer.end();
    }
    void loadAssets() override {
        meshes.scene.loadFromFile(context, getAssetPath() + "models/treasure_smooth.dae", vertexLayout, 1.0f);
        meshes.sceneGlow.loadFromFile(context, getAssetPath() + "models/treasure_glow.dae", vertexLayout, 1.0f);

        // Name the meshes
        // ASSIMP does not load mesh names from the COLLADA file used in this example
        // so we need to set them manually
        // These names are used in command buffer creation for setting debug markers
        // Scene
        std::vector<std::string> names = { "hill", "rocks", "cave", "tree", "mushroom stems", "blue mushroom caps", "red mushroom caps", "grass blades", "chest box", "chest fittings" };
        for (size_t i = 0; i < names.size(); i++) {
            meshes.scene.parts[i].name = names[i];
            meshes.scene.parts[i].name = names[i];
        }

        // Name the buffers for debugging
        // Scene
        DebugMarker::setObjectName(device, (uint64_t)(VkBuffer)meshes.scene.vertices.buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Scene vertex buffer");
        DebugMarker::setObjectName(device, (uint64_t)(VkBuffer)meshes.scene.indices.buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Scene index buffer");
        // Glow
        DebugMarker::setObjectName(device, (uint64_t)(VkBuffer)meshes.sceneGlow.vertices.buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Glow vertex buffer");
        DebugMarker::setObjectName(device, (uint64_t)(VkBuffer)meshes.sceneGlow.indices.buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Glow index buffer");
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {

        // Start a new debug marker region
        DebugMarker::beginRegion(cmdBuffer, "Render scene", glm::vec4(0.5f, 0.76f, 0.34f, 1.0f));

        cmdBuffer.setViewport(0, vks::util::viewport(size));

        vk::Rect2D scissor = vks::util::rect2D(wireframe ? size.width / 2 : size.width, size.height, 0, 0);
        cmdBuffer.setScissor(0, scissor);

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.scene, nullptr);

        // Solid rendering

        // Start a new debug marker region
        DebugMarker::beginRegion(cmdBuffer, "Toon shading draw", glm::vec4(0.78f, 0.74f, 0.9f, 1.0f));

        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.toonshading);
        drawMesh(cmdBuffer, meshes.scene);

        DebugMarker::endRegion(cmdBuffer);

        // Wireframe rendering
        if (wireframe) {
            // Insert debug marker
            DebugMarker::beginRegion(cmdBuffer, "Wireframe draw", glm::vec4(0.53f, 0.78f, 0.91f, 1.0f));

            scissor.offset.x = size.width / 2;
            cmdBuffer.setScissor(0, scissor);

            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.wireframe);
            drawMesh(cmdBuffer, meshes.scene);

            DebugMarker::endRegion(cmdBuffer);

            scissor.offset.x = 0;
            scissor.extent.width = size.width;
            cmdBuffer.setScissor(0, scissor);
        }

        // Post processing
        if (glow) {
            DebugMarker::beginRegion(cmdBuffer, "Apply post processing", glm::vec4(0.93f, 0.89f, 0.69f, 1.0f));

            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.postprocess);
            // Full screen quad is generated by the vertex shaders, so we reuse four vertices (for four invocations) from current vertex buffer
            cmdBuffer.draw(4, 1, 0, 0);

            DebugMarker::endRegion(cmdBuffer);
        }

        // End current debug marker region
        DebugMarker::endRegion(cmdBuffer);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one combined image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 1, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader combined sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });

        // Name for debugging
        DebugMarker::setObjectName(device, (uint64_t)(VkPipelineLayout)pipelineLayout, VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT, "Shared pipeline layout");
        DebugMarker::setObjectName(device, (uint64_t)(VkDescriptorSetLayout)descriptorSetLayout, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT, "Shared descriptor set layout");
    }

    void setupDescriptorSet() {
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        vk::DescriptorImageInfo texDescriptor{ offscreenFrameBuf.textureTarget.sampler, offscreenFrameBuf.textureTarget.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsScene.descriptor },
            // Binding 1 : Color map 
            { descriptorSets.scene, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        // Phong lighting pipeline
        vks::pipelines::GraphicsPipelineBuilder builder(device, pipelineLayout, renderPass);
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        builder.loadShader(getAssetPath() + "shaders/debugmarker/toon.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/debugmarker/toon.frag.spv", vk::ShaderStageFlagBits::eFragment);
        DebugMarker::setObjectName(device, (uint64_t)(VkShaderModule)builder.shaderStages[0].module, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Toon shading vertex shader");
        DebugMarker::setObjectName(device, (uint64_t)(VkShaderModule)builder.shaderStages[1].module, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Toon shading fragment shader");
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelines.toonshading = builder.create(context.pipelineCache);

        // Color only pipeline
        builder.destroyShaderModules();
        builder.loadShader(getAssetPath() + "shaders/debugmarker/colorpass.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/debugmarker/colorpass.frag.spv", vk::ShaderStageFlagBits::eFragment);
        DebugMarker::setObjectName(device, (uint64_t)(VkShaderModule)builder.shaderStages[0].module, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Color-only vertex shader");
        DebugMarker::setObjectName(device, (uint64_t)(VkShaderModule)builder.shaderStages[1].module, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Color-only fragment shader");
        pipelines.color = builder.create(context.pipelineCache);

        // Wire frame rendering pipeline
        builder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        builder.rasterizationState.lineWidth = 1.0f;
        pipelines.wireframe = builder.create(context.pipelineCache);

        // Post processing effect
        builder.destroyShaderModules();
        builder.loadShader(getAssetPath() + "shaders/debugmarker/postprocess.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/debugmarker/postprocess.frag.spv", vk::ShaderStageFlagBits::eFragment);
        DebugMarker::setObjectName(device, (uint64_t)(VkShaderModule)builder.shaderStages[0].module, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Postprocess vertex shader");
        DebugMarker::setObjectName(device, (uint64_t)(VkShaderModule)builder.shaderStages[1].module, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, "Postprocess fragment shader");
        builder.depthStencilState = false;
        builder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;

        auto& blendAttachmentState = builder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        blendAttachmentState.blendEnable = VK_TRUE;
        blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne;
        blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
        blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
        blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eDstAlpha;
        pipelines.postprocess = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = context.createUniformBuffer(uboVS);

        // Name uniform buffer for debugging
        DebugMarker::setObjectName(device, (uint64_t)(VkBuffer)uniformData.vsScene.buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, "Scene uniform buffer block");
        // Add some random tag
        DebugMarker::setObjectTag(device, (uint64_t)(VkBuffer)uniformData.vsScene.buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, 0, sizeof(demoTag), &demoTag);

        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uniformData.vsScene.copy(uboVS);
    }

    void draw() override {
        prepareFrame();

        // Submit offscreen rendering command buffer
        // todo : use event to ensure that offscreen result is finished bfore render command buffer is started
        //if (glow) {
        //    vk::SubmitInfo submitInfo;
        //    submitInfo.pWaitDstStageMask = this->submitInfo.pWaitDstStageMask;
        //    submitInfo.commandBufferCount = 1;
        //    submitInfo.pCommandBuffers = &offscreenCmdBuffer;
        //    submitInfo.waitSemaphoreCount = 1;
        //    submitInfo.pWaitSemaphores = &semaphores.acquireComplete;
        //    submitInfo.signalSemaphoreCount = 1;
        //    submitInfo.pSignalSemaphores = &offscreenSemaphore;
        //    queue.submit(submitInfo, nullptr);
        //}
        //drawCurrentCommandBuffer(glow ? offscreenSemaphore : vk::Semaphore());
        drawCurrentCommandBuffer();
        submitFrame();
    }

    void prepare() override {
        ExampleBase::prepare();
        offscreenSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo());
        DebugMarker::setup(device);
        prepareOffscreen();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildOffscreenCommandBuffer();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
    }

    void viewChanged() override {
        updateUniformBuffers();
    }

    void keyPressed(uint32_t keyCode) override {
        switch (keyCode) {
        case KEY_W:
        case GAMEPAD_BUTTON_X:
            wireframe = !wireframe;
            buildCommandBuffers();
            break;
        case KEY_G:
        case GAMEPAD_BUTTON_A:
            glow = !glow;
            buildCommandBuffers();
            break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
