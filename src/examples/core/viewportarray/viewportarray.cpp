/*
 * Vulkan Example - Viewport array with single pass rendering using geometry shaders
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/viewportarray/multiview.geom.inl>
#include <shaders/viewportarray/scene.frag.inl>
#include <shaders/viewportarray/scene.vert.inl>

// Vertex layout for the models
static const vks::model::VertexLayout VERTEX_LAYOUT{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_NORMAL,
    vks::model::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    vks::model::Model scene;

    struct UBOGS {
        glm::mat4 projection[2];
        glm::mat4 modelview[2];
        glm::vec4 lightPos = glm::vec4(-2.5f, -3.5f, 0.0f, 1.0f);
    } uboGS;

    vks::Buffer uniformBufferGS;

    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Camera and view properties
    float eyeSeparation = 0.08f;
    const float focalLength = 0.5f;
    const float fov = 90.0f;
    const float zNear = 0.1f;
    const float zFar = 256.0f;

    VulkanExample() {
        title = "Viewport arrays";
        camera.type = Camera::CameraType::firstperson;
        camera.setRotation(glm::vec3(0.0f, 90.0f, 0.0f));
        camera.setTranslation(glm::vec3(7.0f, 3.2f, 0.0f));
        camera.movementSpeed = 5.0f;
        settings.overlay = true;
    }

    ~VulkanExample() {
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        scene.destroy();
        uniformBufferGS.destroy();
    }

    // Enable physical device features required for this example
    void getEnabledFeatures() override {
        vkx::ExampleBase::getEnabledFeatures();
        // Geometry shader support is required for this example
        if (!context.deviceInfo.features.core10.geometryShader) {
            throw std::runtime_error("Selected GPU does not support geometry shaders!");
        }
        // Multiple viewports must be supported
        if (!context.deviceInfo.features.core10.multiViewport) {
            throw std::runtime_error("Selected GPU does not support multi viewports!");
        }
        context.enabledFeatures.core10.geometryShader = context.deviceInfo.features.core10.geometryShader;
        context.enabledFeatures.core10.multiViewport = context.deviceInfo.features.core10.multiViewport;
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) override {
        auto halfWidth = (int32_t)size.width / 2;
        auto windowExtent = vk::Extent2D{ (uint32_t)halfWidth, size.height };
        std::vector<vk::Viewport> viewports{
            // Left
            vk::Viewport{ 0.0f, 0.0f, (float)halfWidth, (float)size.height, 0.0f, 1.0f },
            // Right
            vk::Viewport{ (float)halfWidth, 0.0f, (float)halfWidth, (float)size.height, 0.0f, 1.0f },
        };
        commandBuffer.setViewport(0, viewports);

        std::vector<vk::Rect2D> scissorRects{
            vk::Rect2D{ vk::Offset2D{}, windowExtent },
            vk::Rect2D{ vk::Offset2D{ halfWidth, 0 }, windowExtent },
        };
        commandBuffer.setScissor(0, scissorRects);
        commandBuffer.setLineWidth(1.0f);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        commandBuffer.bindVertexBuffers(0, scene.vertices.buffer, { 0 });
        commandBuffer.bindIndexBuffer(scene.indices.buffer, 0, vk::IndexType::eUint32);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        commandBuffer.drawIndexed(scene.indexCount, 1, 0, 0, 0);
    }

    void loadAssets() override { scene.loadFromFile(getAssetPath() + "models/sampleroom.dae", VERTEX_LAYOUT, 0.25f); }

    void setupDescriptorPool() {
        // Example uses two ubos
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eGeometry },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBufferGS.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        VERTEX_LAYOUT.appendVertexLayout(pipelineBuilder.vertexInputState);
        pipelineBuilder.loadShader(vkx::shaders::viewportarray::scene::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::viewportarray::scene::frag, vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.loadShader(vkx::shaders::viewportarray::multiview::geom, vk::ShaderStageFlagBits::eGeometry);
        pipeline = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Geometry shader uniform buffer block
        uniformBufferGS = loader.createUniformBuffer(uboGS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Geometry shader matrices for the two viewports
        // See http://paulbourke.net/stereographics/stereorender/

        // Calculate some variables
        float aspectRatio = (float)(size.width * 0.5f) / (float)size.height;
        float wd2 = zNear * tan(glm::radians(fov / 2.0f));
        float ndfl = zNear / focalLength;
        float left, right;
        float top = wd2;
        float bottom = -wd2;

        glm::vec3 camFront = camera.getFront();
        glm::vec3 camRight = glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::mat4 rotM = glm::mat4(1.0f);
        glm::mat4 transM;

        rotM = glm::rotate(rotM, glm::radians(camera.rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(camera.rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

        // Left eye
        left = -aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;
        right = aspectRatio * wd2 + 0.5f * eyeSeparation * ndfl;

        transM = glm::translate(glm::mat4(1.0f), camera.position - camRight * (eyeSeparation / 2.0f));

        uboGS.projection[0] = glm::frustum(left, right, bottom, top, zNear, zFar);
        uboGS.modelview[0] = rotM * transM;

        // Right eye
        left = -aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;
        right = aspectRatio * wd2 - 0.5f * eyeSeparation * ndfl;

        transM = glm::translate(glm::mat4(1.0f), camera.position + camRight * (eyeSeparation / 2.0f));

        uboGS.projection[1] = glm::frustum(left, right, bottom, top, zNear, zFar);
        uboGS.modelview[1] = rotM * transM;
        uniformBufferGS.copy(uboGS);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadAssets();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffers(); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.sliderFloat("Eye separation", &eyeSeparation, -1.0f, 1.0f)) {
                updateUniformBuffers();
            }
        }
    }
};

RUN_EXAMPLE(VulkanExample)
