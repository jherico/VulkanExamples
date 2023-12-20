/*
* Vulkan Example -  Mesh rendering and loading using ASSIMP
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

using namespace vkx;

// Vertex layout used in this example
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color;
};

class VulkanExample : public ExampleBase {
public:
    bool wireframe = false;

    struct {
        vks::texture::Texture2D colorMap;
    } textures;

    // Contains all buffers and information
    // necessary to represent a mesh for rendering purposes
    // This is for demonstration and learning purposes,
    // the other examples use a mesh loader class for easy access
    struct Mesh {
        vks::model::VertexLayout vertexLayout{ {
            vks::model::VERTEX_COMPONENT_POSITION,
            vks::model::VERTEX_COMPONENT_NORMAL,
            vks::model::VERTEX_COMPONENT_UV,
            vks::model::VERTEX_COMPONENT_COLOR,
        } };
        vks::model::Model model;
    } meshes;

    struct {
        vks::Buffer vsScene;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
        glm::vec4 lightPos = glm::vec4(25.0f, 5.0f, 5.0f, 1.0f);
    } uboVS;

    struct {
        vk::Pipeline solid;
        vk::Pipeline wireframe;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        zoomSpeed = 2.5f;
        rotationSpeed = 0.5f;
        camera.setRotation({ -0.5f, -112.75f, 0.0f });
        camera.setTranslation({ -0.1f, 1.1f, -5.5f });
        title = "Vulkan Example - Mesh rendering";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        device.destroyPipeline(pipelines.wireframe);
        device.destroyPipeline(pipelines.solid);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // Destroy and free mesh resources
        meshes.model.destroy();
        textures.colorMap.destroy();
        uniformData.vsScene.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wireframe : pipelines.solid);
        // Bind mesh vertex buffer
        cmdBuffer.bindVertexBuffers(0, meshes.model.vertices.buffer, { 0 });
        // Bind mesh index buffer
        cmdBuffer.bindIndexBuffer(meshes.model.indices.buffer, 0, vk::IndexType::eUint32);
        // Render mesh vertex buffer using it's indices
        cmdBuffer.drawIndexed(meshes.model.indexCount, 1, 0, 0, 0);
    }

    // Load a mesh based on data read via assimp
    // The other example will use the VulkanMesh loader which has some additional functionality for loading meshes
    void loadAssets() override {
        meshes.model.loadFromFile(context, getAssetPath() + "models/voyager/voyager.dae", meshes.vertexLayout);
        textures.colorMap.loadFromFile(context, getAssetPath() + "models/voyager/voyager.ktx", vk::Format::eBc3UnormBlock);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one combined image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1),
        };

        descriptorPool = device.createDescriptorPool({ {}, 1, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader combined sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];
        vk::DescriptorImageInfo texDescriptor{ textures.colorMap.sampler, textures.colorMap.view, vk::ImageLayout::eGeneral };
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vsScene.descriptor },
            // Binding 1 : Color map
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Solid rendering pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;
        pipelineBuilder.vertexInputState.appendVertexLayout(meshes.vertexLayout);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/mesh/mesh.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/mesh/mesh.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);

        // Wire frame rendering pipeline
        pipelineBuilder.rasterizationState.polygonMode = vk::PolygonMode::eLine;
        pipelines.wireframe = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vsScene = context.createUniformBuffer(uboVS);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboVS.projection = getProjection();
        uboVS.model = getView();
        uniformData.vsScene.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();
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
    }

    void viewChanged() override { updateUniformBuffers(); }

    void keyPressed(uint32_t keyCode) override {
        switch (keyCode) {
            case KEY_W:
            case GAMEPAD_BUTTON_A:
                wireframe = !wireframe;
                buildCommandBuffers();
                break;
        }
    }
};

RUN_EXAMPLE(VulkanExample)
