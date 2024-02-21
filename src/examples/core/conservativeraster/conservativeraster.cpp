/*
 * Vulkan Example - Conservative rasterization
 *
 * Note: Requires a device that supports the VK_EXT_conservative_rasterization extension
 *
 * Uses an offscreen buffer with lower resolution to demonstrate the effect of conservative rasterization
 *
 * Copyright (C) 2018 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/offscreen.hpp>

#include <shaders/conservativeraster/fullscreen.frag.inl>
#include <shaders/conservativeraster/fullscreen.vert.inl>
#include <shaders/conservativeraster/triangle.frag.inl>
#include <shaders/conservativeraster/triangle.vert.inl>
#include <shaders/conservativeraster/triangleoverlay.frag.inl>

#define FB_COLOR_FORMAT vk::Format::eB8G8R8A8Unorm
#define ZOOM_FACTOR 16

class VulkanExample : public vkx::OffscreenExampleBase {
public:
    bool conservativeRasterEnabled = true;

    struct Vertex {
        float position[3];
        float color[3];
    };

    struct Triangle {
        vks::Buffer vertices;
        vks::Buffer indices;
        uint32_t indexCount;
    } triangle;

    vks::Buffer uniformBuffer;

    struct UniformBuffers {
        vks::Buffer scene;
    } uniformBuffers;

    struct UboScene {
        glm::mat4 projection;
        glm::mat4 model;
    } uboScene;

    struct PipelineLayouts {
        vk::PipelineLayout scene;
        vk::PipelineLayout fullscreen;
    } pipelineLayouts;

    struct Pipelines {
        vk::Pipeline triangle;
        vk::Pipeline triangleConservativeRaster;
        vk::Pipeline triangleOverlay;
        vk::Pipeline fullscreen;
    } pipelines;

    struct DescriptorSetLayouts {
        vk::DescriptorSetLayout scene;
        vk::DescriptorSetLayout fullscreen;
    } descriptorSetLayouts;

    struct DescriptorSets {
        vk::DescriptorSet scene;
        vk::DescriptorSet fullscreen;
    } descriptorSets;

#if 0
    // Framebuffer for offscreen rendering
    struct OffscreenPass {
        vk::Extent2D extent;
        uint32_t& width = extent.width;
        uint32_t& height = extent.height;
        vks::Image color;
        vk::ImageView colorView;
        vk::Sampler sampler;
        vk::DescriptorImageInfo descriptor;

        void destroy() {
            const auto& device = vks::Context::get().device;
            device.destroy(colorView);
            device.destroy(sampler);
            color.destroy();
        }
    } offscreenPass;
#endif

    VulkanExample() {
        title = "Conservative rasterization";
        settings.overlay = true;

        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -2.0f));

        // Enable extension required for conservative rasterization
        context.requireDeviceExtensions({ VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME });
        // Reading device properties of conservative rasterization requires VK_KHR_get_physical_device_properties2 to be enabled
        context.requireExtensions({ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
    }

    ~VulkanExample() {
        device.destroy(pipelines.triangle);
        device.destroy(pipelines.triangleOverlay);
        device.destroy(pipelines.triangleConservativeRaster);
        device.destroy(pipelines.fullscreen);

        device.destroy(pipelineLayouts.scene);
        device.destroy(pipelineLayouts.fullscreen);

        device.destroy(descriptorSetLayouts.scene);
        device.destroy(descriptorSetLayouts.fullscreen);

        uniformBuffers.scene.destroy();
        triangle.vertices.destroy();
        triangle.indices.destroy();
    }

    void getEnabledFeatures() override {
        ExampleBase::getEnabledFeatures();
        // Conservative rasterization setup
        context.enabledFeatures.core10.fillModeNonSolid = deviceInfo.features.core10.fillModeNonSolid;
        context.enabledFeatures.core10.wideLines = deviceInfo.features.core10.wideLines;
    }

    /*
	    Setup offscreen framebuffer, attachments and render passes for lower resolution rendering of the scene
	*/
    void prepareOffscreen() override {
        vkx::offscreen::Builder builder({ width / ZOOM_FACTOR, height / ZOOM_FACTOR });
        auto& samplerCreateInfo = builder.samplerCreateInfo;
        samplerCreateInfo.magFilter = vk::Filter::eNearest;
        samplerCreateInfo.minFilter = vk::Filter::eNearest;
        samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
        builder.appendColorFormat(defaultColorFormat, vk::ImageUsageFlagBits::eSampled);
        offscreen.prepare(builder);
    }

    // Sets up the command buffer that renders the scene to the offscreen frame buffer
    void buildOffscreenCommandBuffer() override {
        using namespace vks::util;
        auto& drawCmdBuffer = offscreen.cmdBuffer;
        drawCmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
        setImageLayout(drawCmdBuffer, offscreen.colorTargets[0].image, ImageTransitionState::UNDEFINED, ImageTransitionState::COLOR_ATTACHMENT);
        drawCmdBuffer.beginRendering(offscreen.renderingInfo);
        drawCmdBuffer.setViewport(0, vks::util::viewport(offscreen.size));
        drawCmdBuffer.setScissor(0, vks::util::rect2D(offscreen.size));
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, conservativeRasterEnabled ? pipelines.triangleConservativeRaster : pipelines.triangle);
        drawCmdBuffer.bindVertexBuffers(0, triangle.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(triangle.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.drawIndexed(triangle.indexCount, 1, 0, 0, 0);
        drawCmdBuffer.endRendering();
        setImageLayout(drawCmdBuffer, offscreen.colorTargets[0].image, ImageTransitionState::COLOR_ATTACHMENT, ImageTransitionState::SAMPLED);
        drawCmdBuffer.end();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCmdBuffer) override {
        drawCmdBuffer.setViewport(0, viewport());
        drawCmdBuffer.setScissor(0, scissor());

        // Low-res triangle from offscreen framebuffer
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.fullscreen);
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.fullscreen, 0, 1, &descriptorSets.fullscreen, 0, nullptr);
        drawCmdBuffer.draw(3, 1, 0, 0);

        // Overlay actual triangle
        VkDeviceSize offsets[1] = { 0 };
        drawCmdBuffer.bindVertexBuffers(0, triangle.vertices.buffer, { 0 });
        drawCmdBuffer.bindIndexBuffer(triangle.indices.buffer, 0, vk::IndexType::eUint32);
        drawCmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.triangleOverlay);
        drawCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayouts.scene, 0, descriptorSets.scene, nullptr);
        drawCmdBuffer.draw(3, 1, 0, 0);
    }

    void loadAssets() override {
        // Create a single triangle
        struct Vertex {
            float position[3];
            float color[3];
        };

        std::vector<Vertex> vertexBuffer = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
        uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
        triangle.indexCount = static_cast<uint32_t>(indexBuffer.size());
        triangle.vertices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
        triangle.indices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 3 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 2 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Scene rendering
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.scene = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.scene = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.scene });

        // Fullscreen pass
        setLayoutBindings = {
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayouts.fullscreen = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayouts.fullscreen = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayouts.fullscreen });
    }

    void setupDescriptorSet() {
        // Scene rendering
        descriptorSets.scene = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.scene })[0];
        // Fullscreen pass
        descriptorSets.fullscreen = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayouts.fullscreen })[0];

        vk::DescriptorImageInfo offscreenImage{ offscreen.sampler, offscreen.colorTargets[0].imageView, vk::ImageLayout::eReadOnlyOptimal };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            vk::WriteDescriptorSet{ descriptorSets.scene, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.scene.descriptor },
            vk::WriteDescriptorSet{ descriptorSets.fullscreen, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &offscreenImage },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayouts.fullscreen };
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.depthStencilState = false;

        // Conservative rasterization pipeline state
        vk::PipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterStateCI{};
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eOverestimate;
        conservativeRasterStateCI.extraPrimitiveOverestimationSize =
            context.deviceInfo.properties.conservativeRasterizationEXT.maxExtraPrimitiveOverestimationSize;
        // Conservative rasterization state has to be chained into the pipeline rasterization state create info structure
        vks::injectNext(pipelineBuilder.rasterizationState, conservativeRasterStateCI);

        // Full screen pass
        // Empty vertex input state (full screen triangle generated in vertex shader)
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eDisabled;
        pipelineBuilder.loadShader(vkx::shaders::conservativeraster::fullscreen::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::conservativeraster::fullscreen::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.fullscreen = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Vertex bindings and attributes
        pipelineBuilder.layout = pipelineLayouts.scene;
        pipelineBuilder.vertexInputState.bindingDescriptions = {
            { 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
        };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            { 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) },
            { 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) },
        };
        // Original triangle outline (no conservative rasterization)
        // TODO: Check support for lines
        pipelineBuilder.rasterizationState.lineWidth = 2.0f;
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eDisabled;

        pipelineBuilder.loadShader(vkx::shaders::conservativeraster::triangle::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::conservativeraster::triangleoverlay::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.triangleOverlay = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        // Triangle rendering
        pipelineBuilder.dynamicRendering(FB_COLOR_FORMAT, defaultDepthStencilFormat);
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eFill;
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eDisabled;
        pipelineBuilder.loadShader(vkx::shaders::conservativeraster::triangle::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::conservativeraster::triangle::frag, vk::ShaderStageFlagBits::eFragment);
        // Default
        pipelines.triangle = pipelineBuilder.create(context.pipelineCache);

        // Conservative rasterization enabled
        conservativeRasterStateCI.conservativeRasterizationMode = vk::ConservativeRasterizationModeEXT::eOverestimate;
        pipelines.triangleConservativeRaster = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        uniformBuffers.scene = loader.createUniformBuffer(uboScene);
        updateUniformBuffersScene();
    }

    void updateUniformBuffersScene() {
        uboScene.projection = camera.matrices.perspective;
        uboScene.model = camera.matrices.view;
        uniformBuffers.scene.copy(uboScene);
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareOffscreen();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        buildOffscreenCommandBuffer();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffersScene(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Conservative rasterization", &conservativeRasterEnabled)) {
                waitIdle();
                buildOffscreenCommandBuffer();
            }
        }
        if (ui.header("Device properties")) {
            auto& conservativeRasterProps = context.deviceInfo.properties.conservativeRasterizationEXT;
            ui.text("maxExtraPrimitiveOverestimationSize:         %f", conservativeRasterProps.maxExtraPrimitiveOverestimationSize);
            ui.text("extraPrimitiveOverestimationSizeGranularity: %f", conservativeRasterProps.extraPrimitiveOverestimationSizeGranularity);
            ui.text("primitiveUnderestimation:                    %s", conservativeRasterProps.primitiveUnderestimation ? "yes" : "no");
            ui.text("conservativePointAndLineRasterization:       %s", conservativeRasterProps.conservativePointAndLineRasterization ? "yes" : "no");
            ui.text("degenerateTrianglesRasterized:               %s", conservativeRasterProps.degenerateTrianglesRasterized ? "yes" : "no");
            ui.text("degenerateLinesRasterized:                   %s", conservativeRasterProps.degenerateLinesRasterized ? "yes" : "no");
            ui.text("fullyCoveredFragmentShaderInputVariable:     %s", conservativeRasterProps.fullyCoveredFragmentShaderInputVariable ? "yes" : "no");
            ui.text("conservativeRasterizationPostDepthCoverage:  %s", conservativeRasterProps.conservativeRasterizationPostDepthCoverage ? "yes" : "no");
        }
    }
};

VULKAN_EXAMPLE_MAIN()
