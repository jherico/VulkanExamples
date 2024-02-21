/*
 * Vulkan Example - Example for VK_EXT_debug_marker extension. To be used in conjuction with a debugging app like RenderDoc (https://renderdoc.org)
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/debugmarker/colorpass.frag.inl>
#include <shaders/debugmarker/colorpass.vert.inl>
#include <shaders/debugmarker/postprocess.frag.inl>
#include <shaders/debugmarker/postprocess.vert.inl>
#include <shaders/debugmarker/toon.frag.inl>
#include <shaders/debugmarker/toon.vert.inl>

// Offscreen properties
#define OFFSCREEN_DIM 256
#define OFFSCREEN_FORMAT vk::Format::eR8G8B8A8Unorm
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
    // active = false;
}

// Sets the debug name of an object
// All Objects in Vulkan are represented by their 64-bit handles which are passed into this function
// along with the object type
void setObjectName(uint64_t object, vk::ObjectType objectType, const char* name) {
    const auto& device = vks::Context::get().device;
    // Check for valid function pointer (may not be present if not running in a debugging application)
    if (active && pfnDebugMarkerSetObjectName) {
        vk::DebugUtilsObjectNameInfoEXT nameInfo;
        nameInfo.objectType = objectType;
        nameInfo.objectHandle = object;
        nameInfo.pObjectName = name;
        device.setDebugUtilsObjectNameEXT(nameInfo);
    }
}

template <typename T>
struct assert_false : std::false_type {};

template <typename T>
void setObjectName(T object, const char* name) {
    static_assert(assert_false<T>::value, "Does not contain type");
}

template <>
void setObjectName(vk::Sampler object, const char* name) {
    setObjectName(reinterpret_cast<uint64_t>(object.operator VkSampler()), vk::ObjectType::eSampler, name);
}

template <>
void setObjectName(vk::Image object, const char* name) {
    setObjectName(reinterpret_cast<uint64_t>(object.operator VkImage()), vk::ObjectType::eImage, name);
}

template <>
void setObjectName(vk::Buffer object, const char* name) {
    setObjectName(reinterpret_cast<uint64_t>(object.operator VkBuffer()), vk::ObjectType::eBuffer, name);
}

template <>
void setObjectName(vk::ShaderModule object, const char* name) {
    setObjectName(reinterpret_cast<uint64_t>(object.operator VkShaderModule()), vk::ObjectType::eShaderModule, name);
}

template <>
void setObjectName(vk::PipelineLayout object, const char* name) {
    setObjectName(reinterpret_cast<uint64_t>(object.operator VkPipelineLayout()), vk::ObjectType::ePipelineLayout, name);
}

template <>
void setObjectName(vk::DescriptorSetLayout object, const char* name) {
    setObjectName(reinterpret_cast<uint64_t>(object.operator VkDescriptorSetLayout()), vk::ObjectType::eDescriptorSetLayout, name);
}

// Set the tag for an object
void setObjectTag(uint64_t object, vk::ObjectType objectType, uint64_t name, size_t tagSize, const void* tag) {
    const auto& device = vks::Context::get().device;
    // Check for valid function pointer (may not be present if not running in a debugging application)
    if (active && pfnDebugMarkerSetObjectTag) {
        vk::DebugUtilsObjectTagInfoEXT tagInfo;
        tagInfo.objectType = objectType;
        tagInfo.objectHandle = object;
        tagInfo.tagName = name;
        tagInfo.tagSize = tagSize;
        tagInfo.pTag = tag;
        device.setDebugUtilsObjectTagEXT(tagInfo);
    }
}

// Set the tag for an object
template <typename T>
void setObjectTag(T object, uint64_t name, size_t tagSize, const void* tag) {
    static_assert(assert_false<T>::value, "Does not contain type");
}

template <>
void setObjectTag(vk::Buffer object, uint64_t name, size_t tagSize, const void* tag) {
    setObjectTag(reinterpret_cast<uint64_t>(object.operator VkBuffer()), vk::ObjectType::eBuffer, name, tagSize, tag);
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
};  // namespace DebugMarker
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

    struct Offscreen {
        vk::Extent2D size{ OFFSCREEN_DIM, OFFSCREEN_DIM };
        // vk::Framebuffer framebuffer;
        vks::Image color, depth;
        vks::Image textureTarget;
        vk::ImageView textureTargetView, colorView, depthView;
        vk::Sampler textureTargetSampler;
        vk::Semaphore offscreenSemaphore;
        vk::CommandBuffer offscreenCmdBuffer;

        void prepare(vk::CommandBuffer newCommandBuffer) {
            auto& loader = vks::Loader::get();
            auto& context = vks::Context::get();
            auto& device = context.device;
            offscreenSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo());

            loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& cmdBuffer) {
                vk::FormatProperties formatProperties;

                // Get device properites for the requested texture format
                formatProperties = context.physicalDevice.getFormatProperties(OFFSCREEN_FORMAT);
                // Check if blit destination is supported for the requested format
                // Only try for optimal tiling, linear tiling usually won't support blit as destination anyway
                assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);

                // The texture target used for rendering into the main draw
                {
                    // Texture target
                    auto& tex = textureTarget;
                    auto& view = textureTargetView;
                    auto& sampler = textureTargetSampler;

                    // Prepare blit target texture
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
                    // A target image should have dedicate allocation
                    tex.create(imageCreateInfo, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
                    view = tex.createView();

                    // Transform image layout to read-only
                    using namespace vks::util;
                    setImageLayout(cmdBuffer, tex, ImageTransitionState::UNDEFINED, ImageTransitionState::SAMPLED);

                    // Create sampler
                    vk::SamplerCreateInfo samplerCreateInfo;
                    samplerCreateInfo.magFilter = OFFSCREEN_FILTER;
                    samplerCreateInfo.minFilter = OFFSCREEN_FILTER;
                    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
                    samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
                    samplerCreateInfo.addressModeV = samplerCreateInfo.addressModeU;
                    samplerCreateInfo.addressModeW = samplerCreateInfo.addressModeU;
                    samplerCreateInfo.mipLodBias = 0.0f;
                    samplerCreateInfo.maxAnisotropy = 0;
                    samplerCreateInfo.compareOp = vk::CompareOp::eNever;
                    samplerCreateInfo.minLod = 0.0f;
                    samplerCreateInfo.maxLod = 0.0f;
                    samplerCreateInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
                    sampler = device.createSampler(samplerCreateInfo);
                    // Name for debugging
                    DebugMarker::setObjectName(tex.image, "Off-screen texture target image");
                    DebugMarker::setObjectName(sampler, "Off-screen texture target sampler");
                    vk::DebugReportObjectTypeEXT::eSampler;
                }

                // Find a suitable depth format
                vk::Format fbDepthFormat = context.deviceInfo.supportedDepthFormat;

                // Color attachment
                vk::ImageCreateInfo image;
                image.imageType = vk::ImageType::e2D;
                image.format = OFFSCREEN_FORMAT;
                image.extent.width = size.width;
                image.extent.height = size.height;
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

                color.create(image, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
                colorView = color.createView();

                // Depth stencil attachment
                image.format = fbDepthFormat;
                image.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
                depth.create(image, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
                depthView = depth.createView();

                using vks::util::ImageTransitionState;
                vks::util::setImageLayout(cmdBuffer, depth, ImageTransitionState::UNDEFINED, ImageTransitionState::DEPTH_ATTACHMENT);
            });

            // Command buffer for offscreen rendering
            offscreenCmdBuffer = newCommandBuffer;

            // Name for debugging
            DebugMarker::setObjectName(color.image, "Off-screen color framebuffer");
            DebugMarker::setObjectName(depth.image, "Off-screen depth framebuffer");
        }

        void destroy() {
            auto& context = vks::Context::get();
            auto& device = context.device;

            textureTarget.destroy();
            device.destroy(textureTargetSampler);
            textureTargetSampler = nullptr;
            device.destroy(textureTargetView);
            textureTargetView = nullptr;

            device.destroy(colorView);
            colorView = nullptr;
            color.destroy();

            device.destroy(depthView);
            depthView = nullptr;
            depth.destroy();
        }
    } offscreen;

    // Random tag data
    struct {
        const char name[17] = "debug marker tag";
    } demoTag;

    VulkanExample() {
        // current debugging tools don't yet work with Vulkan 1.1, so target 1.0
        // FIXME when RenderDoc works with 1.1, update this
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ -4.35f, 16.25f, 0.0f });
        camera.setTranslation({ 0.1f, 1.1f, -8.5f });
        title = "Vulkan Example - VK_EXT_debug_marker";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.toonshading);
        pipelines.toonshading = nullptr;
        device.destroy(pipelines.color);
        pipelines.color = nullptr;
        device.destroy(pipelines.wireframe);
        pipelines.wireframe = nullptr;
        device.destroy(pipelines.postprocess);
        pipelines.postprocess = nullptr;

        device.destroy(pipelineLayout);
        pipelineLayout = nullptr;
        device.destroy(descriptorSetLayout);
        descriptorSetLayout = nullptr;

        // Destroy and free mesh resources
        meshes.scene.destroy();
        meshes.sceneGlow.destroy();

        uniformData.vsScene.destroy();

        offscreen.destroy();
    }

    // Command buffer for rendering color only scene for glow
    void buildOffscreenCommandBuffer() {
        vk::RenderingAttachmentInfo colorAttachmentInfo;
        colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
        colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachmentInfo.clearValue = vks::util::clearColor(glm::vec4(0));
        colorAttachmentInfo.imageView = offscreen.colorView;

        vk::RenderingAttachmentInfo depthAttachmentInfo;
        depthAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
        depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
        depthAttachmentInfo.clearValue = vk::ClearDepthStencilValue{ 1.0, 0 };
        depthAttachmentInfo.imageView = offscreen.depthView;

        vk::RenderingInfo renderingInfo;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.layerCount = 1;
        renderingInfo.pColorAttachments = &colorAttachmentInfo;
        renderingInfo.pDepthAttachment = &depthAttachmentInfo;
        renderingInfo.pStencilAttachment = &depthAttachmentInfo;
        renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, offscreen.size };

        auto& offscreenCmdBuffer = offscreen.offscreenCmdBuffer;
        offscreenCmdBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        // Start a new debug marker region
        DebugMarker::beginRegion(offscreenCmdBuffer, "Off-screen scene rendering", glm::vec4(1.0f, 0.78f, 0.05f, 1.0f));
        offscreenCmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        offscreenCmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        offscreenCmdBuffer.beginRendering(renderingInfo);
        offscreenCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.scene, nullptr);
        offscreenCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.color);

        // Draw glow scene
        drawMesh(offscreenCmdBuffer, meshes.sceneGlow);

        offscreenCmdBuffer.endRendering();

        // Make sure color writes to the framebuffer are finished before using it as transfer source
        vks::util::setImageLayout(offscreenCmdBuffer, offscreen.color, vks::util::ImageTransitionState::COLOR_ATTACHMENT,
                                  vks::util::ImageTransitionState::TRANSFER_SRC);
        auto& textureTarget = offscreen.textureTarget;
        vks::util::setImageLayout(offscreenCmdBuffer, textureTarget, vks::util::ImageTransitionState::SAMPLED, vks::util::ImageTransitionState::TRANSFER_DST);

        // Transform texture target to transfer destination

        // Blit offscreen color buffer to our texture target
        {
            vk::ImageBlit imgBlit;

            imgBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            imgBlit.srcSubresource.layerCount = 1;

            imgBlit.srcOffsets[1].x = offscreen.size.width;
            imgBlit.srcOffsets[1].y = offscreen.size.height;
            imgBlit.srcOffsets[1].z = 1;

            imgBlit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            imgBlit.dstSubresource.layerCount = 1;

            imgBlit.dstOffsets[1].x = offscreen.textureTarget.createInfo.extent.width;
            imgBlit.dstOffsets[1].y = offscreen.textureTarget.createInfo.extent.height;
            imgBlit.dstOffsets[1].z = 1;

            // Blit from framebuffer image to texture image
            // vkCmdBlitImage does scaling and (if necessary and possible) also does format conversions
            offscreenCmdBuffer.blitImage(offscreen.color.image, vk::ImageLayout::eTransferSrcOptimal, offscreen.textureTarget.image,
                                         vk::ImageLayout::eTransferDstOptimal, imgBlit, vk::Filter::eLinear);
        }
        // Transform framebuffer color attachment back
        vks::util::setImageLayout(offscreenCmdBuffer, textureTarget, vks::util::ImageTransitionState::TRANSFER_SRC, vks::util::ImageTransitionState::RENDER);

        // Transform texture target back to shader read
        // Makes sure that writes to the texture are finished before
        // it's accessed in the shader
        vks::util::setImageLayout(offscreenCmdBuffer, textureTarget, vks::util::ImageTransitionState::TRANSFER_DST, vks::util::ImageTransitionState::SAMPLED);
        DebugMarker::endRegion(offscreenCmdBuffer);

        offscreenCmdBuffer.end();
    }

    void loadAssets() override {
        meshes.scene.loadFromFile(getAssetPath() + "models/treasure_smooth.dae", vertexLayout, 1.0f);
        meshes.sceneGlow.loadFromFile(getAssetPath() + "models/treasure_glow.dae", vertexLayout, 1.0f);

        // Name the meshes
        // ASSIMP does not load mesh names from the COLLADA file used in this example
        // so we need to set them manually
        // These names are used in command buffer creation for setting debug markers
        // Scene
        std::vector<std::string> names = { "hill",         "rocks",     "cave",          "tree", "mushroom stems", "blue mushroom caps", "red mushroom caps",
                                           "grass blades", "chest box", "chest fittings" };
        for (size_t i = 0; i < names.size(); i++) {
            meshes.scene.parts[i].name = names[i];
            meshes.scene.parts[i].name = names[i];
        }

        // Name the buffers for debugging
        // Scene
        DebugMarker::setObjectName(meshes.scene.vertices.buffer, "Scene vertex buffer");
        DebugMarker::setObjectName(meshes.scene.indices.buffer, "Scene index buffer");
        // Glow
        DebugMarker::setObjectName(meshes.sceneGlow.vertices.buffer, "Glow vertex buffer");
        DebugMarker::setObjectName(meshes.sceneGlow.indices.buffer, "Glow index buffer");
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
        DebugMarker::setObjectName(pipelineLayout, "Shared pipeline layout");
        DebugMarker::setObjectName(descriptorSetLayout, "Shared descriptor set layout");
    }

    void setupDescriptorSet() {
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        vk::DescriptorImageInfo texDescriptor{ offscreen.textureTargetSampler, offscreen.textureTargetView, vk::ImageLayout::eGeneral };

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
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout };
        builder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        builder.loadShader(vkx::shaders::debugmarker::toon::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::debugmarker::toon::frag, vk::ShaderStageFlagBits::eFragment);
        DebugMarker::setObjectName(builder.shaderStages[0].module, "Toon shading vertex shader");
        DebugMarker::setObjectName(builder.shaderStages[1].module, "Toon shading fragment shader");
        vertexLayout.appendVertexLayout(builder.vertexInputState);
        pipelines.toonshading = builder.create(context.pipelineCache);

        // Color only pipeline
        builder.destroyShaderModules();
        builder.loadShader(vkx::shaders::debugmarker::colorpass::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::debugmarker::colorpass::frag, vk::ShaderStageFlagBits::eFragment);
        DebugMarker::setObjectName(builder.shaderStages[0].module, "Color-only vertex shader");
        DebugMarker::setObjectName(builder.shaderStages[1].module, "Color-only fragment shader");
        pipelines.color = builder.create(context.pipelineCache);

        // Wire frame rendering pipeline
        builder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        builder.rasterizationState.lineWidth = 1.0f;
        pipelines.wireframe = builder.create(context.pipelineCache);

        // Post processing effect
        builder.destroyShaderModules();
        builder.loadShader(vkx::shaders::debugmarker::postprocess::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::debugmarker::postprocess::frag, vk::ShaderStageFlagBits::eFragment);
        DebugMarker::setObjectName(builder.shaderStages[0].module, "Postprocess vertex shader");
        DebugMarker::setObjectName(builder.shaderStages[1].module, "Postprocess fragment shader");
        builder.depthStencilState = false;
        builder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;

        auto& blendAttachmentState = builder.colorBlendState.blendAttachmentStates[0];
        blendAttachmentState.colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
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
        uniformData.vsScene = loader.createUniformBuffer(uboVS);

        // Name uniform buffer for debugging
        DebugMarker::setObjectName(uniformData.vsScene.buffer, "Scene uniform buffer block");
        // Add some random tag
        DebugMarker::setObjectTag(uniformData.vsScene.buffer, 0, sizeof(demoTag), &demoTag);

        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = camera.matrices.perspective;
        uboVS.model = camera.matrices.view;
        uniformData.vsScene.copy(uboVS);
    }

    void preRender() override {
        if (glow) {
            vks::frame::QueuedCommandBuilder builder{ offscreen.offscreenCmdBuffer, vkx::RenderStates::OFFSCREEN_PRERENDER,
                                                      vk::PipelineStageFlagBits2::eAllGraphics };
            queueCommandBuffer(builder);
        }
    }

    void prepare() override {
        ExampleBase::prepare();
        DebugMarker::setup(device);
        offscreen.prepare(graphicsQueue.createCommandBuffer());
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

    void viewChanged() override { updateUniformBuffers(); }

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
