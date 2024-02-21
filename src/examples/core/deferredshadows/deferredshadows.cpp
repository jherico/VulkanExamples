/*
 * Vulkan Example - Deferred shading with shadows from multiple light sources using geometry shader instancing
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>
#include <rendering/offscreen.hpp>

#include <shaders/deferredshadows/debug.frag.inl>
#include <shaders/deferredshadows/debug.vert.inl>
#include <shaders/deferredshadows/deferred.frag.inl>
#include <shaders/deferredshadows/deferred.vert.inl>
#include <shaders/deferredshadows/mrt.frag.inl>
#include <shaders/deferredshadows/mrt.vert.inl>
#include <shaders/deferredshadows/shadow.frag.inl>
#include <shaders/deferredshadows/shadow.geom.inl>
#include <shaders/deferredshadows/shadow.vert.inl>

// Shadowmap properties
#if defined(__ANDROID__)
#define SHADOWMAP_DIM 1024
#else
#define SHADOWMAP_DIM 2048
#endif
// 16 bits of depth is enough for such a small scene
#define SHADOWMAP_FORMAT vk::Format::eD32SfloatS8Uint

#if defined(__ANDROID__)
// Use max. screen dimension as deferred framebuffer size
#define FB_DIM std::max(size.width, size.height)
#else
#define FB_DIM 2048
#endif

// Must match the LIGHT_COUNT define in the shadow and deferred shaders
#define LIGHT_COUNT 3

class VulkanExample : public vkx::ExampleBase {
public:
    bool debugDisplay = false;
    bool enableShadows = true;

    // Keep depth range as small as possible
    // for better shadow map precision
    float zNear = 0.1f;
    float zFar = 64.0f;
    float lightFOV = 100.0f;

    // Depth bias (and slope) are used to avoid shadowing artefacts
    float depthBiasConstant = 1.25f;
    float depthBiasSlope = 1.75f;

    struct Offscreen {
        vkx::offscreen::Renderer deferred;
        vkx::offscreen::Renderer shadow;
        vk::Sampler deferredSampler;
        vk::Sampler shadowSampler;
    } offscreen;

    struct Material {
        vks::texture::Texture2D colorMap;
        vks::texture::Texture2D normalMap;
    };
    struct {
        Material model;
        Material background;
    } materials;

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_UV,
        vks::model::VERTEX_COMPONENT_COLOR,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_TANGENT,
    });

    struct {
        vks::model::Model model;
        vks::model::Model background;
        vks::model::Model quad;
    } models;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 view;
        glm::vec4 instancePos[3];
        int layer;
    } uboVS, uboOffscreenVS;

    // This UBO stores the shadow matrices for all of the light sources
    // The matrices are indexed using geometry shader instancing
    // The instancePos is used to place the models using instanced draws
    struct {
        glm::mat4 mvp[LIGHT_COUNT];
        glm::vec4 instancePos[3];
    } uboShadowGS;

    struct Light {
        glm::vec4 position;
        glm::vec4 target;
        glm::vec4 color;
        glm::mat4 viewMatrix;
    };

    struct {
        glm::vec4 viewPos;
        Light lights[LIGHT_COUNT];
        uint32_t useShadows = 1;
    } uboFragmentLights;

    struct {
        vks::Buffer vsFullScreen;
        vks::Buffer vsOffscreen;
        vks::Buffer fsLights;
        vks::Buffer uboShadowGS;
    } uniformBuffers;

    struct {
        vk::Pipeline deferred;
        vk::Pipeline offscreen;
        vk::Pipeline debug;
        vk::Pipeline shadowpass;
    } pipelines;

    struct {
        // todo: rename, shared with deferred and shadow pass
        vk::PipelineLayout deferred;
        vk::PipelineLayout offscreen;
    } pipelineLayouts;

    struct {
        vk::DescriptorSet model;
        vk::DescriptorSet background;
        vk::DescriptorSet shadow;
    } descriptorSets;

    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    struct {
        vk::CommandBuffer deferred;
    } commandBuffers;

    VulkanExample() {
        title = "Deferred shading with shadows";
        camera.type = Camera::CameraType::firstperson;
#if defined(__ANDROID__)
        camera.movementSpeed = 2.5f;
#else
        camera.movementSpeed = 5.0f;
        camera.rotationSpeed = 0.25f;
#endif
        camera.position = { 2.15f, 0.3f, -8.75f };
        camera.setRotation(glm::vec3(-0.75f, 12.5f, 0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, zNear, zFar);
        timerSpeed *= 0.25f;
        paused = true;
        settings.overlay = true;
    }

    ~VulkanExample() {
        // Frame buffers
        device.destroy(pipelines.deferred);
        device.destroy(pipelines.offscreen);
        device.destroy(pipelines.shadowpass);
        device.destroy(pipelines.debug);

        device.destroy(pipelineLayouts.deferred);
        device.destroy(pipelineLayouts.offscreen);

        device.destroy(descriptorSetLayout);

        // Meshes
        models.model.destroy();
        models.background.destroy();
        models.quad.destroy();

        // Uniform buffers
        uniformBuffers.vsOffscreen.destroy();
        uniformBuffers.vsFullScreen.destroy();
        uniformBuffers.fsLights.destroy();
        uniformBuffers.uboShadowGS.destroy();

        // materials
        materials.model.colorMap.destroy();
        materials.model.normalMap.destroy();
        materials.background.colorMap.destroy();
        materials.background.normalMap.destroy();
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        const auto& deviceFeatures = context.deviceInfo.features.core10;
        auto& enabledFeatures = context.enabledFeatures.core10;
        // Geometry shader support is required for writing to multiple shadow map layers in one single pass
        enabledFeatures.geometryShader = deviceFeatures.geometryShader;
        if (!enabledFeatures.geometryShader) {
            throw std::runtime_error("Selected GPU does not support geometry shaders!");
        }
    }

    // Prepare a layered shadow map with each layer containing depth from a light's point of view
    // The shadow mapping pass uses geometry shader instancing to output the scene from the different
    // light sources' point of view to the layers of the depth attachment in one single pass
    void shadowSetup() {
        vkx::offscreen::Builder builder{ vk::Extent2D{ SHADOWMAP_DIM, SHADOWMAP_DIM } };
        builder.withDepthFormat(SHADOWMAP_FORMAT, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled);
        builder.withLayerCount(LIGHT_COUNT);
        builder.withViewType(vk::ImageViewType::e2DArray);
        offscreen.shadow.prepare(builder);
    }

    // Prepare the framebuffer for offscreen rendering with multiple attachments used as render targets inside the fragment shaders
    void deferredSetup() {
        vkx::offscreen::Builder builder{ vk::Extent2D{ FB_DIM, FB_DIM } };

        // Four attachments (3 color, 1 depth)

        // Color attachments
        // Attachment 0: (World space) Positions
        builder.appendColorFormat(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        // Attachment 1: (World space) Normals
        builder.appendColorFormat(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        // Attachment 2: Albedo (color)
        builder.appendColorFormat(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        // Depth attachment
        builder.withDepthFormat(defaultDepthStencilFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment);

        // Create sampler to sample from the color attachments
        offscreen.deferred.prepare(builder);

        vk::SamplerCreateInfo sampler_create_info;
        sampler_create_info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        sampler_create_info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        sampler_create_info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        offscreen.deferredSampler = device.createSampler(sampler_create_info);
    }

    // Put render commands for the scene into the given command buffer
    void renderScene(const vk::Extent2D& size, const vk::CommandBuffer& cmdBuffer, bool shadow) {
        vk::Viewport viewport = { 0, 0, (float)size.width, (float)size.height, 0.0f, 1.0f };
        commandBuffers.deferred.setViewport(0, viewport);
        vk::Rect2D scissor = { {}, size };
        commandBuffers.deferred.setScissor(0, scissor);

        // Background
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, shadow ? descriptorSets.shadow : descriptorSets.background,
                                     nullptr);
        cmdBuffer.bindVertexBuffers(0, models.background.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(models.background.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(models.background.indexCount, 1, 0, 0, 0);

        // Objects
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.offscreen, 0, shadow ? descriptorSets.shadow : descriptorSets.model,
                                     nullptr);
        cmdBuffer.bindVertexBuffers(0, models.model.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(models.model.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(models.model.indexCount, 3, 0, 0, 0);
    }

    // Build a secondary command buffer for rendering the scene values to the offscreen frame buffer attachments
    void buildDeferredCommandBuffer() {
        if (!commandBuffers.deferred) {
            commandBuffers.deferred = graphicsQueue.createCommandBuffer();
        }

        // First pass: Shadow map generation
        // -------------------------------------------------------------------------------------------------------

        commandBuffers.deferred.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        commandBuffers.deferred.beginRendering(offscreen.shadow.renderingInfo);
        // Set depth bias (aka "Polygon offset")
        commandBuffers.deferred.setDepthBias(depthBiasConstant, 0.0f, depthBiasSlope);
        commandBuffers.deferred.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.shadowpass);
        renderScene(offscreen.shadow.size, commandBuffers.deferred, true);
        commandBuffers.deferred.endRendering();

        // Second pass: Deferred calculations
        // -------------------------------------------------------------------------------------------------------

        // Clear values for all attachments written in the fragment sahder
        commandBuffers.deferred.beginRendering(offscreen.deferred.renderingInfo);
        vk::Viewport viewport = vks::util::viewport(offscreen.deferred.size);
        commandBuffers.deferred.setViewport(0, viewport);
        vk::Rect2D scissor = vks::util::rect2D(offscreen.deferred.size);
        commandBuffers.deferred.setScissor(0, scissor);
        commandBuffers.deferred.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.offscreen);
        renderScene(offscreen.deferred.size, commandBuffers.deferred, false);
        commandBuffers.deferred.endRenderPass();
        commandBuffers.deferred.end();
    }

    void loadAssets() override {
        models.model.loadFromFile(getAssetPath() + "models/armor/armor.dae", vertexLayout, 1.0f);

        vks::model::ModelCreateInfo modelCreateInfo;
        modelCreateInfo.scale = glm::vec3(15.0f);
        modelCreateInfo.uvscale = glm::vec2(1.0f, 1.5f);
        modelCreateInfo.center = glm::vec3(0.0f, 2.3f, 0.0f);
        models.background.loadFromFile(getAssetPath() + "models/openbox.dae", vertexLayout, modelCreateInfo);

        // materials
        std::string texFormatSuffix;
        vk::Format texFormat;
        const auto& deviceFeatures = context.deviceInfo.features.core10;

        // Get supported compressed texture format
        if (deviceFeatures.textureCompressionBC) {
            texFormatSuffix = "_bc3_unorm";
            texFormat = vk::Format::eBc3UnormBlock;
        } else if (deviceFeatures.textureCompressionASTC_LDR) {
            texFormatSuffix = "_astc_8x8_unorm";
            texFormat = vk::Format::eAstc8x8UnormBlock;
        } else if (deviceFeatures.textureCompressionETC2) {
            texFormatSuffix = "_etc2_unorm";
            texFormat = vk::Format::eEtc2R8G8B8A8UnormBlock;
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }

        materials.model.colorMap.loadFromFile(getAssetPath() + "models/armor/color" + texFormatSuffix + ".ktx", texFormat);
        materials.model.normalMap.loadFromFile(getAssetPath() + "models/armor/normal" + texFormatSuffix + ".ktx", texFormat);
        materials.background.colorMap.loadFromFile(getAssetPath() + "textures/stonefloor02_color" + texFormatSuffix + ".ktx", texFormat);
        materials.background.normalMap.loadFromFile(getAssetPath() + "textures/stonefloor02_normal" + texFormatSuffix + ".ktx", texFormat);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        vk::Viewport viewport{ 0, 0, (float)size.width, (float)size.height, 0.0f, 1.0f };
        drawCmdBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{ { 0, 0 }, size };
        drawCmdBuffer.setScissor(0, scissor);

        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.deferred, 0, descriptorSet, nullptr);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.deferred);
        drawCmdBuffer.bindVertexBuffers(0, models.quad.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(models.quad.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(6, 1, 0, 0, 0);

        if (debugDisplay) {
            // Visualize depth maps
            drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.debug);
            drawCmdBuffer.drawIndexed(6, LIGHT_COUNT, 0, 0, 0);
        }
    }

    /** @brief Create a single quad for fullscreen deferred pass and debug passes (debug pass uses instancing for light visualization) */
    void generateQuads() {
        struct Vertex {
            float pos[3];
            float uv[2];
            float col[3];
            float normal[3];
            float tangent[3];
        };

        std::vector<Vertex> vertexBuffer;

        vertexBuffer.push_back({ { 1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } });
        vertexBuffer.push_back({ { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } });
        vertexBuffer.push_back({ { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } });
        vertexBuffer.push_back({ { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } });

        models.quad.vertices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        for (uint32_t i = 0; i < 3; ++i) {
            uint32_t indices[6] = { 0, 1, 2, 2, 3, 0 };
            for (auto index : indices) {
                indexBuffer.push_back(i * 4 + index);
            }
        }
        models.quad.indexCount = static_cast<uint32_t>(indexBuffer.size());
        models.quad.indices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
        models.quad.device = device;
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 12 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 16 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 4, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // todo: split for clarity, esp. with GS instancing
        // Deferred shading layout (Shared with debug display)
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0: Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry },
            // Binding 1: Position texture
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 2: Normals texture
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 3: Albedo texture
            vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 4: Fragment shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 4, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 5: Shadow map
            vk::DescriptorSetLayoutBinding{ 5, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.deferred = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
        // Offscreen (scene) rendering pipeline layout
        pipelineLayouts.offscreen = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        // Textured quad descriptor set
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        // Model
        descriptorSets.model = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        // Background
        descriptorSets.background = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        // Shadow
        descriptorSets.shadow = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Image descriptors for the offscreen color attachments
        vk::DescriptorImageInfo texDescriptorPosition{
            defaultSampler,
            offscreen.deferred.colorTargets[0].imageView,
            vk::ImageLayout::eReadOnlyOptimal,
        };
        vk::DescriptorImageInfo texDescriptorNormal{
            defaultSampler,
            offscreen.deferred.colorTargets[1].imageView,
            vk::ImageLayout::eReadOnlyOptimal,
        };
        vk::DescriptorImageInfo texDescriptorAlbedo{
            defaultSampler,
            offscreen.deferred.colorTargets[2].imageView,
            vk::ImageLayout::eReadOnlyOptimal,
        };
        vk::DescriptorImageInfo texDescriptorShadowMap{
            offscreen.deferredSampler,
            offscreen.shadow.depthTarget.imageView,
            vk::ImageLayout::eDepthStencilReadOnlyOptimal,
        };

        auto colorMapDescriptor = materials.model.colorMap.makeDescriptor(defaultSampler);
        auto normalMapDescriptor = materials.model.normalMap.makeDescriptor(defaultSampler);

        auto bgColorMapDescriptor = materials.background.colorMap.makeDescriptor(defaultSampler);
        auto bgNormalMapDescriptor = materials.background.normalMap.makeDescriptor(defaultSampler);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0: Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.vsFullScreen.descriptor },
            // Binding 1: World space position texture
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorPosition },
            // Binding 2: World space normals texture
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorNormal },
            // Binding 3: Albedo texture
            { descriptorSet, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorAlbedo },
            // Binding 4: Fragment shader uniform buffer
            { descriptorSet, 4, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.fsLights.descriptor },
            // Binding 5: Shadow map
            { descriptorSet, 5, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptorShadowMap },

            // Model descriptor set
            // Binding 0: Vertex shader uniform buffer
            { descriptorSets.model, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.vsOffscreen.descriptor },
            // Binding 1: Color map
            { descriptorSets.model, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorMapDescriptor },
            // Binding 2: Normal map
            { descriptorSets.model, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &normalMapDescriptor },

            // Background descriptor set
            // Binding 0: Vertex shader uniform buffer
            { descriptorSets.background, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.vsOffscreen.descriptor },
            // Binding 1: Color map
            { descriptorSets.background, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &bgColorMapDescriptor },
            // Binding 2: Normal map
            { descriptorSets.background, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &bgNormalMapDescriptor },

            // Shadow descriptor set
            // Binding 0: Vertex shader uniform buffer
            { descriptorSets.shadow, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.uboShadowGS.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Final fullscreen pass pipeline
        vks::pipelines::GraphicsPipelineBuilder builder(device, pipelineLayouts.deferred);
        builder.dynamicRendering(
            {
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Unorm,
            },
            defaultDepthStencilFormat, defaultDepthStencilFormat);

        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        vertexLayout.appendVertexLayout(builder.vertexInputState);
        auto& blendAttachmentStates = builder.colorBlendState.blendAttachmentStates;
        blendAttachmentStates.resize(3);

        builder.loadShader(vkx::shaders::deferredshadows::deferred::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::deferredshadows::deferred::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.deferred = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Debug display pipeline
        builder.loadShader(vkx::shaders::deferredshadows::debug::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::deferredshadows::debug::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.debug = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Offscreen pipeline
        // Separate render pass
        // Separate layout
        builder.layout = pipelineLayouts.offscreen;
        builder.loadShader(vkx::shaders::deferredshadows::mrt::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::deferredshadows::mrt::frag, vk::ShaderStageFlagBits::eFragment);
        // Blend attachment states required for all color attachments
        // This is important, as color write mask will otherwise be 0x0 and you
        // won't see anything rendered to the attachment
        builder.colorBlendState.blendAttachmentStates.resize(3);
        pipelines.offscreen = builder.create(context.pipelineCache);
        builder.destroyShaderModules();

        // Shadow mapping pipeline
        // The shadow mapping pipeline uses geometry shader instancing (invocations layout modifier) to output
        // shadow maps for multiple lights sources into the different shadow map layers in one single render pass
        builder.dynamicRendering({}, SHADOWMAP_FORMAT, SHADOWMAP_FORMAT);
        builder.loadShader(vkx::shaders::deferredshadows::shadow::vert, vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(vkx::shaders::deferredshadows::shadow::frag, vk::ShaderStageFlagBits::eFragment);
        builder.loadShader(vkx::shaders::deferredshadows::shadow::geom, vk::ShaderStageFlagBits::eGeometry);
        // Shadow pass doesn't use any color attachments
        builder.colorBlendState.blendAttachmentStates.clear();
        // Cull front faces
        builder.rasterizationState.cullMode = vk::CullModeFlagBits::eFront;
        builder.depthStencilState.depthCompareOp = vk::CompareOp::eLessOrEqual;
        // Enable depth bias
        builder.rasterizationState.depthBiasEnable = VK_TRUE;
        // Add depth bias to dynamic state, so we can change it at runtime
        builder.dynamicState.dynamicStateEnables.push_back(vk::DynamicState::eDepthBias);
        pipelines.shadowpass = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Fullscreen vertex shader
        uniformBuffers.vsFullScreen = loader.createUniformBuffer(uboVS);

        // Deferred vertex shader
        uniformBuffers.vsOffscreen = loader.createUniformBuffer(uboOffscreenVS);

        // Deferred fragment shader
        uniformBuffers.fsLights = loader.createUniformBuffer(uboFragmentLights);

        // Shadow map vertex shader (matrices from shadow's pov)
        uniformBuffers.uboShadowGS = loader.createUniformBuffer(uboShadowGS);

        // Init some values
        uboOffscreenVS.instancePos[0] = glm::vec4(0.0f);
        uboOffscreenVS.instancePos[1] = glm::vec4(-4.0f, 0.0, -4.0f, 0.0f);
        uboOffscreenVS.instancePos[2] = glm::vec4(4.0f, 0.0, -4.0f, 0.0f);

        uboOffscreenVS.instancePos[1] = glm::vec4(-7.0f, 0.0, -4.0f, 0.0f);
        uboOffscreenVS.instancePos[2] = glm::vec4(4.0f, 0.0, -6.0f, 0.0f);

        // Update
        updateUniformBuffersScreen();
        updateUniformBufferDeferredMatrices();
        updateUniformBufferDeferredLights();
    }

    void updateUniformBuffersScreen() {
        uboVS.projection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        uboVS.model = glm::mat4(1.0f);
        uniformBuffers.vsFullScreen.copy(uboVS);
    }

    void updateUniformBufferDeferredMatrices() {
        uboOffscreenVS.projection = camera.matrices.perspective;
        uboOffscreenVS.view = camera.matrices.view;
        uboOffscreenVS.model = glm::mat4(1.0f);
        uniformBuffers.vsOffscreen.copy(uboOffscreenVS);
    }

    Light initLight(const glm::vec3& pos, const glm::vec3& target, const glm::vec3& color) {
        Light light;
        light.position = glm::vec4(pos, 1.0f);
        light.target = glm::vec4(target, 0.0f);
        light.color = glm::vec4(color, 0.0f);
        return light;
    }

    void initLights() {
        uboFragmentLights.lights[0] = initLight(glm::vec3(-14.0f, -0.5f, 15.0f), glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.5f, 0.5f));
        uboFragmentLights.lights[1] = initLight(glm::vec3(14.0f, -4.0f, 12.0f), glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        uboFragmentLights.lights[2] = initLight(glm::vec3(0.0f, -10.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
    }

    // Update fragment shader light position uniform block
    void updateUniformBufferDeferredLights() {
        // Animate
        // if (!paused)
        {
            uboFragmentLights.lights[0].position.x = -14.0f + std::abs(sin(glm::radians(timer * 360.0f)) * 20.0f);
            uboFragmentLights.lights[0].position.z = 15.0f + cos(glm::radians(timer * 360.0f)) * 1.0f;

            uboFragmentLights.lights[1].position.x = 14.0f - std::abs(sin(glm::radians(timer * 360.0f)) * 2.5f);
            uboFragmentLights.lights[1].position.z = 13.0f + cos(glm::radians(timer * 360.0f)) * 4.0f;

            uboFragmentLights.lights[2].position.x = 0.0f + sin(glm::radians(timer * 360.0f)) * 4.0f;
            uboFragmentLights.lights[2].position.z = 4.0f + cos(glm::radians(timer * 360.0f)) * 2.0f;
        }

        for (uint32_t i = 0; i < LIGHT_COUNT; i++) {
            // mvp from light's pov (for shadows)
            glm::mat4 shadowProj = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
            glm::mat4 shadowView =
                glm::lookAt(glm::vec3(uboFragmentLights.lights[i].position), glm::vec3(uboFragmentLights.lights[i].target), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 shadowModel = glm::mat4(1.0f);

            uboShadowGS.mvp[i] = shadowProj * shadowView * shadowModel;
            uboFragmentLights.lights[i].viewMatrix = uboShadowGS.mvp[i];
        }

        memcpy(uboShadowGS.instancePos, uboOffscreenVS.instancePos, sizeof(uboOffscreenVS.instancePos));

        uniformBuffers.uboShadowGS.copy(uboShadowGS);

        uboFragmentLights.viewPos = glm::vec4(camera.position, 0.0f) * glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

        uniformBuffers.fsLights.copy(uboFragmentLights);
    }

    void preRender() override {
        vks::frame::QueuedCommandBuilder builder{ commandBuffers.deferred, vkx::RenderStates::OFFSCREEN_PRERENDER,
                                                  vk::PipelineStageFlagBits2::eColorAttachmentOutput };
        queueCommandBuffer(builder);
    }

    void prepare() override {
        ExampleBase::prepare();
        generateQuads();
        deferredSetup();
        shadowSetup();
        initLights();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        buildDeferredCommandBuffer();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        updateUniformBufferDeferredLights();
    }

    void viewChanged() override { updateUniformBufferDeferredMatrices(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Display shadow targets", &debugDisplay)) {
                buildCommandBuffers();
                updateUniformBuffersScreen();
            }
            bool shadows = (uboFragmentLights.useShadows == 1);
            if (ui.checkBox("Shadows", &shadows)) {
                uboFragmentLights.useShadows = shadows;
                updateUniformBufferDeferredLights();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
