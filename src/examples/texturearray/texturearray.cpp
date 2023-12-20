/*
* Vulkan Example - Texture arrays and instanced rendering
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
};

class VulkanExample : public vkx::ExampleBase {
public:
    // Number of array layers in texture array
    // Also used as instance count
    vks::texture::Texture2DArray textureArray;

    struct {
        vks::model::Model quad;
    } meshes;

    struct {
        vks::Buffer vertexShader;
    } uniformData;

    struct UboInstanceData {
        // Model matrix
        glm::mat4 model;
        // Texture array index
        // Vec4 due to padding
        glm::vec4 arrayIndex;
    };

    struct {
        // Global matrices
        struct {
            glm::mat4 projection;
            glm::mat4 view;
        } matrices;
        // Seperate data for each instance
        UboInstanceData instance[8];
    } uboVS;

    struct {
        vk::Pipeline solid;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        rotationSpeed = 0.25f;
        camera.setRotation({ -15.0f, 35.0f, 0.0f });
        camera.dolly(-15.0f);
        title = "Vulkan Example - Texture arrays";
        srand((uint32_t)time(NULL));
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroyPipeline(pipelines.solid);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        meshes.quad.destroy();

        uniformData.vertexShader.destroy();

        // Clean up texture resources
        textureArray.destroy();
    }

    void loadTextures() {
        const auto& deviceFeatures = context.deviceFeatures;
        vk::Format format;
        std::string filename;
        if (deviceFeatures.textureCompressionBC) {
            filename = "texturearray_bc3_unorm.ktx";
            format = vk::Format::eBc3UnormBlock;
        } else if (deviceFeatures.textureCompressionASTC_LDR) {
            filename = "texturearray_astc_8x8_unorm.ktx";
            format = vk::Format::eAstc8x8UnormBlock;
        } else if (deviceFeatures.textureCompressionETC2) {
            filename = "texturearray_etc2_unorm.ktx";
            format = vk::Format::eEtc2R8G8B8UnormBlock;
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }

        textureArray.loadFromFile(context, getAssetPath() + "textures/" + filename, format);
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, textureArray.layerCount, 0, 0, 0);
    }

    // Setup vertices for a single uv-mapped quad
    void generateQuad() {
#define dim 2.5f
        std::vector<Vertex> vertexBuffer = { { { dim, dim, 0.0f }, { 1.0f, 1.0f } },
                                             { { -dim, dim, 0.0f }, { 0.0f, 1.0f } },
                                             { { -dim, -dim, 0.0f }, { 0.0f, 0.0f } },
                                             { { dim, -dim, 0.0f }, { 1.0f, 0.0f } } };
#undef dim

        meshes.quad.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 1 },
            { vk::DescriptorType::eCombinedImageSampler, 1 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler (texture array)
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];
        // vk::Image descriptor for the texture array
        vk::DescriptorImageInfo texArrayDescriptor = vk::DescriptorImageInfo(textureArray.sampler, textureArray.view, vk::ImageLayout::eGeneral);
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0 : Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.vertexShader.descriptor },
            // Binding 1 : Fragment shader cubemap sampler
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texArrayDescriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.vertexInputState.bindingDescriptions = { { 0, sizeof(Vertex), vk::VertexInputRate::eVertex } };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Location 0 : Position
            { 0, 0, vk::Format::eR32G32B32Sfloat, 0 },
            // Location 1 : Texture coordinates
            { 1, 0, vk::Format::eR32G32Sfloat, sizeof(float) * 3 },
        };
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texturearray/instancing.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texturearray/instancing.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformData.vertexShader = context.createUniformBuffer(uboVS);

        // Array indices and model matrices are fixed
        float offset = -1.5f;
        float center = (textureArray.layerCount * offset) / 2;
        for (uint32_t i = 0; i < textureArray.layerCount; i++) {
            // Instance model matrix
            uboVS.instance[i].model = glm::translate(glm::mat4(), glm::vec3(0.0f, i * offset - center, 0.0f)) *
                                      glm::mat4_cast(glm::angleAxis(glm::radians(60.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
            // Instance texture array index
            uboVS.instance[i].arrayIndex.x = (float)i;
        }
        // Update instanced part of the uniform buffer
        uniformData.vertexShader.copy(uboVS);
        updateUniformBufferMatrices();
    }

    void updateUniformBufferMatrices() {
        // Only updates the uniform buffer block part containing the global matrices
        // Projection
        uboVS.matrices.projection = camera.matrices.perspective;
        // View
        uboVS.matrices.view = camera.matrices.view;

        // Only update the matrices part of the uniform buffer
        uniformData.vertexShader.copy(uboVS.matrices);
    }

    void prepare() override {
        ExampleBase::prepare();
        loadTextures();
        generateQuad();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBufferMatrices(); }
};

RUN_EXAMPLE(VulkanExample)
