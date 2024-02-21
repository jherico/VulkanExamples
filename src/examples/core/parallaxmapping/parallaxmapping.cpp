/*
 * Vulkan Example - Parallax Mapping
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <shaders/parallaxmapping/normalmap.frag.inl>
#include <shaders/parallaxmapping/normalmap.vert.inl>
#include <shaders/parallaxmapping/parallax.frag.inl>
#include <shaders/parallaxmapping/parallax.vert.inl>

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_TANGENT,
    vks::model::Component::VERTEX_COMPONENT_BITANGENT,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    bool splitScreen = true;

    struct {
        vks::texture::Texture2D colorMap;
        // Normals and height are combined in one texture (height = alpha channel)
        vks::texture::Texture2D normalHeightMap;
    } textures;

    struct {
        vks::model::Model quad;
    } meshes;

    struct {
        vks::Buffer vertexShader;
        vks::Buffer fragmentShader;
    } uniformData;

    struct {
        struct {
            glm::mat4 projection;
            glm::mat4 model;
            glm::mat4 normal;
            glm::vec4 lightPos;
            glm::vec4 cameraPos;
        } vertexShader;

        struct FragmentShader {
            // Scale and bias control the parallax offset effect
            // They need to be tweaked for each material
            // Getting them wrong destroys the depth effect
            float scale = 0.06f;
            float bias = -0.04f;
            float lightRadius = 1.0f;
            int32_t usePom = 1;
            int32_t displayNormalMap = 0;
        } fragmentShader;

    } ubos;

    struct {
        vk::Pipeline parallaxMapping;
        vk::Pipeline normalMapping;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.setRotation({ 0.0, 15.0, 0.0 });
        camera.setRotation({ 0.1, 0.1, -2.5 });
        camera.dolly(-2.25f);
        rotationSpeed = 0.25f;
        paused = true;
        title = "Vulkan Example - Parallax Mapping";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroy(pipelines.parallaxMapping);
        device.destroy(pipelines.normalMapping);

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        meshes.quad.destroy();

        uniformData.vertexShader.destroy();
        uniformData.fragmentShader.destroy();

        textures.colorMap.destroy();
        textures.normalHeightMap.destroy();
    }

    void loadTextures() {
        textures.colorMap.loadFromFile(getAssetPath() + "textures/rocks_color_bc3.dds", vk::Format::eBc3UnormBlock);
        textures.normalHeightMap.loadFromFile(getAssetPath() + "textures/rocks_normal_height_rgba.dds", vk::Format::eR8G8B8A8Unorm);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        vk::Viewport viewport = vks::util::viewport((splitScreen) ? (float)size.width / 2.0f : (float)size.width, (float)size.height, 0.0f, 1.0f);
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

        vk::DeviceSize offsets = 0;
        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);

        // Parallax enabled
        cmdBuffer.setViewport(0, viewport);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.parallaxMapping);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 1);

        // Normal mapping
        if (splitScreen) {
            viewport.x = viewport.width;
            cmdBuffer.setViewport(0, viewport);
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.normalMapping);
            cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 1);
        }
    }

    void loadMeshes() { meshes.quad.loadFromFile(getAssetPath() + "models/plane_z.obj", vertexLayout, 0.1f); }

    void setupDescriptorPool() {
        // Example uses two ubos and two image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes{ { vk::DescriptorType::eUniformBuffer, 2 }, { vk::DescriptorType::eCombinedImageSampler, 2 } };

        descriptorPool = device.createDescriptorPool({ {}, 4, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader color map image sampler
            vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 2 : Fragment combined normal and heightmap
            vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
            // Binding 3 : Fragment shader uniform buffer
            vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        auto colorMapDesc = textures.colorMap.makeDescriptor(defaultSampler);
        auto heightMap = textures.normalHeightMap.makeDescriptor(defaultSampler);

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vertexShader.descriptor },
            // Binding 1 : Fragment shader image sampler
            vk::WriteDescriptorSet{ descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &colorMapDesc },
            // Binding 2 : Combined normal and heightmap
            vk::WriteDescriptorSet{ descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &heightMap },
            // Binding 3 : Fragment shader uniform buffer
            vk::WriteDescriptorSet{ descriptorSet, 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.fragmentShader.descriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Parallax mapping pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout };
        pipelineCreator.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineCreator.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        vertexLayout.appendVertexLayout(pipelineCreator.vertexInputState);
        pipelineCreator.loadShader(vkx::shaders::parallaxmapping::parallax::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::parallaxmapping::parallax::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.parallaxMapping = pipelineCreator.create(context.pipelineCache);
        pipelineCreator.destroyShaderModules();

        // Normal mapping (no parallax effect)
        pipelineCreator.loadShader(vkx::shaders::parallaxmapping::normalmap::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(vkx::shaders::parallaxmapping::normalmap::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.normalMapping = pipelineCreator.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Vertex shader ubo
        uniformData.vertexShader = loader.createUniformBuffer(ubos.vertexShader);
        // Fragment shader ubo
        uniformData.fragmentShader = loader.createUniformBuffer(ubos.fragmentShader);

        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        // Vertex shader
        ubos.vertexShader.projection =
            glm::perspective(glm::radians(45.0f), (float)(size.width * ((splitScreen) ? 0.5f : 1.0f)) / (float)size.height, 0.001f, 256.0f);
        ubos.vertexShader.model = camera.matrices.view;
        ubos.vertexShader.normal = glm::inverseTranspose(ubos.vertexShader.model);

        if (!paused) {
            ubos.vertexShader.lightPos.x = sinf(glm::radians(timer * 360.0f)) * 0.5f;
            ubos.vertexShader.lightPos.y = cosf(glm::radians(timer * 360.0f)) * 0.5f;
        }

        ubos.vertexShader.cameraPos = glm::vec4(0.0, 0.0, camera.position.z, 0.0);
        uniformData.vertexShader.copy(ubos.vertexShader);

        // Fragment shader
        uniformData.fragmentShader.copy(ubos.fragmentShader);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadTextures();
        loadMeshes();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared)
            return;
        draw();
        if (!paused) {
            updateUniformBuffers();
        }
    }

    void viewChanged() override { updateUniformBuffers(); }

    void toggleParallaxOffset() {
        ubos.fragmentShader.usePom = !ubos.fragmentShader.usePom;
        updateUniformBuffers();
    }

    void toggleNormalMapDisplay() {
        ubos.fragmentShader.displayNormalMap = !ubos.fragmentShader.displayNormalMap;
        updateUniformBuffers();
    }

    void toggleSplitScreen() {
        splitScreen = !splitScreen;
        updateUniformBuffers();
        buildCommandBuffers();
    }
};

RUN_EXAMPLE(VulkanExample)
