/*
* Vulkan Example - Cube map texture loading and displaying
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/
#include <vulkanExampleBase.h>

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    vks::texture::TextureCubeMap cubeMap;

    struct {
        vks::model::Model skybox, object;
    } meshes;

    struct {
        vks::Buffer objectVS;
        vks::Buffer skyboxVS;
    } uniformData;

    struct {
        glm::mat4 projection;
        glm::mat4 model;
    } uboVS;

    struct {
        vk::Pipeline skybox;
        vk::Pipeline reflect;
    } pipelines;

    struct {
        vk::DescriptorSet object;
        vk::DescriptorSet skybox;
    } descriptorSets;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.dolly(-4.0f);
        camera.setRotation({ -2.25f, -35.0f, 0.0f });
        rotationSpeed = 0.25f;
        title = "Vulkan Example - Cube map";
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        // Clean up texture resources
        cubeMap.destroy();

        device.destroyPipeline(pipelines.skybox);
        device.destroyPipeline(pipelines.reflect);

        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        meshes.object.destroy();
        meshes.skybox.destroy();

        uniformData.objectVS.destroy();
        uniformData.skyboxVS.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.setViewport(0, vks::util::viewport(size, 0.0f, 1.0f));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));

        vk::DeviceSize offsets = 0;

        // Skybox
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.skybox, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.skybox.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.skybox.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.skybox);
        cmdBuffer.drawIndexed(meshes.skybox.indexCount, 1, 0, 0, 0);

        // 3D object
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.object, nullptr);
        cmdBuffer.bindVertexBuffers(0, meshes.object.vertices.buffer, offsets);
        cmdBuffer.bindIndexBuffer(meshes.object.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.reflect);
        cmdBuffer.drawIndexed(meshes.object.indexCount, 1, 0, 0, 0);
    }

    void loadMeshes() {
        meshes.object.loadFromFile(context, getAssetPath() + "models/sphere.obj", vertexLayout, 0.05f);
        meshes.skybox.loadFromFile(context, getAssetPath() + "models/cube.obj", vertexLayout, 0.05f);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            { vk::DescriptorType::eUniformBuffer, 2 },
            { vk::DescriptorType::eCombinedImageSampler, 2 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            // Binding 1 : Fragment shader image sampler
            { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSets() {
        // vk::Image descriptor for the cube map texture
        vk::DescriptorImageInfo cubeMapDescriptor{ cubeMap.sampler, cubeMap.view, vk::ImageLayout::eGeneral };
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        // 3D object descriptor set
        descriptorSets.object = device.allocateDescriptorSets(allocInfo)[0];
        device.updateDescriptorSets(
            {
                { descriptorSets.object, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.objectVS.descriptor },
                { descriptorSets.object, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cubeMapDescriptor },
            },
            nullptr);

        // Sky box descriptor set
        descriptorSets.skybox = device.allocateDescriptorSets(allocInfo)[0];
        device.updateDescriptorSets(
            {
                { descriptorSets.skybox, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.skyboxVS.descriptor },
                { descriptorSets.skybox, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &cubeMapDescriptor },
            },
            nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texturecubemap/reflect.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texturecubemap/reflect.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.reflect = pipelineBuilder.create(context.pipelineCache);
        pipelineBuilder.destroyShaderModules();

        pipelineBuilder.loadShader(getAssetPath() + "shaders/texturecubemap/skybox.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/texturecubemap/skybox.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.depthStencilState.depthWriteEnable = VK_FALSE;
        pipelines.skybox = pipelineBuilder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // 3D objact
        uniformData.objectVS = context.createUniformBuffer(uboVS);
        // Skybox
        uniformData.skyboxVS = context.createUniformBuffer(uboVS);
    }

    void updateUniformBuffers() {
        // Common projection
        uboVS.projection = camera.matrices.perspective;

        // Skysphere, rotation only
        uboVS.model = camera.matrices.skyboxView;
        uniformData.skyboxVS.copy(uboVS);

        // 3D object, translation combined with the rotation
        uboVS.model = camera.matrices.view;
        uniformData.objectVS.copy(uboVS);
    }

    void loadTextures() {
        vk::Format format;
        std::string filename;
        const auto& deviceFeatures = context.deviceFeatures;
        if (deviceFeatures.textureCompressionBC) {
            filename = "cubemap_yokohama_bc3_unorm.ktx";
            format = vk::Format::eBc3UnormBlock;
        } else if (deviceFeatures.textureCompressionASTC_LDR) {
            filename = "cubemap_yokohama_astc_8x8_unorm.ktx";
            format = vk::Format::eAstc8x8UnormBlock;
        } else if (deviceFeatures.textureCompressionETC2) {
            filename = "cubemap_yokohama_etc2_unorm.ktx";
            format = vk::Format::eEtc2R8G8B8UnormBlock;
        } else {
            throw std::runtime_error("Device does not support any compressed texture format!");
        }

        cubeMap.loadFromFile(context, getAssetPath() + "textures/" + filename, format);
    }

    void prepare() {
        ExampleBase::prepare();
        loadMeshes();
        prepareUniformBuffers();
        loadTextures();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSets();
        buildCommandBuffers();
        prepared = true;
    }

    virtual void render() {
        if (!prepared)
            return;
        draw();
        updateUniformBuffers();
    }

    virtual void viewChanged() { updateUniformBuffers(); }
};

RUN_EXAMPLE(VulkanExample)
