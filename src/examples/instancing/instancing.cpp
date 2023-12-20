/*
* Vulkan Example - Instanced mesh rendering, uses a separate vertex buffer for instanced data
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

#define INSTANCE_COUNT 2048

// Vertex layout for this example
vks::model::VertexLayout vertexLayout{ {
    vks::model::Component::VERTEX_COMPONENT_POSITION,
    vks::model::Component::VERTEX_COMPONENT_NORMAL,
    vks::model::Component::VERTEX_COMPONENT_UV,
    vks::model::Component::VERTEX_COMPONENT_COLOR,
} };

class VulkanExample : public vkx::ExampleBase {
public:
    struct {
        vks::model::Model rock;
        vks::model::Model planet;
    } models;

    struct {
        vks::texture::Texture2D planet;
        vks::texture::Texture2DArray rocks;
    } textures;

    // Per-instance data block
    struct InstanceData {
        glm::vec3 pos;
        glm::vec3 rot;
        float scale;
        uint32_t texIndex;
    };

    // Contains the instanced data
    vks::Buffer instanceBuffer;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec4 lightPos = glm::vec4(0.0f, -5.0f, 0.0f, 1.0f);
        float locSpeed = 0.0f;
        float globSpeed = 0.0f;
    } uboVS;

    struct {
        vks::Buffer scene;
    } uniformData;

    vk::PipelineLayout pipelineLayout;
    struct {
        vk::Pipeline instancedRocks;
        vk::Pipeline planet;
        vk::Pipeline starfield;
    } pipelines;

    struct {
        vk::DescriptorSet instancedRocks;
        vk::DescriptorSet planet;
    } descriptorSets;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        rotationSpeed = 0.25f;
        camera.dolly(-12.0f);
        title = "Vulkan Example - Instanced mesh rendering";
        srand((uint32_t)time(NULL));
    }

    ~VulkanExample() {
        device.destroy(pipelines.instancedRocks);
        device.destroy(pipelines.planet);
        device.destroy(pipelines.starfield);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);
        instanceBuffer.destroy();
        models.planet.destroy();
        models.rock.destroy();
        uniformData.scene.destroy();
        textures.planet.destroy();
        textures.rocks.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.planet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.starfield);
        cmdBuffer.draw(4, 1, 0, 0);

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.planet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.planet);
        cmdBuffer.bindVertexBuffers(0, models.planet.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(models.planet.indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(models.planet.indexCount, 1, 0, 0, 0);

        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.instancedRocks, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.instancedRocks);

        // Binding point 0 : Mesh vertex buffer
        cmdBuffer.bindVertexBuffers(0, models.rock.vertices.buffer, { 0 });
        // Binding point 1 : Instance data buffer
        cmdBuffer.bindVertexBuffers(1, instanceBuffer.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(models.rock.indices.buffer, 0, vk::IndexType::eUint32);
        // Render instances
        cmdBuffer.drawIndexed(models.rock.indexCount, INSTANCE_COUNT, 0, 0, 0);
    }

    void loadAssets() override {
        models.planet.loadFromFile(context, getAssetPath() + "models/sphere.obj", vertexLayout, 0.2f);
        models.rock.loadFromFile(context, getAssetPath() + "models/rock01.dae", vertexLayout, 0.1f);
        textures.rocks.loadFromFile(context, getAssetPath() + "textures/texturearray_rocks_bc3.ktx", vk::Format::eBc3UnormBlock);
        textures.planet.loadFromFile(context, getAssetPath() + "textures/lavaplanet_bc3_unorm.ktx", vk::Format::eBc3UnormBlock);
    }

    void setupDescriptorPool() {
        // Example uses one ubo
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 2 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 2 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
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
        descriptorSets.instancedRocks = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayout })[0];
        descriptorSets.planet = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, 1, &descriptorSetLayout })[0];

        vk::DescriptorImageInfo texRocksDescriptor = vk::DescriptorImageInfo{ textures.rocks.sampler, textures.rocks.view, vk::ImageLayout::eGeneral };
        vk::DescriptorImageInfo texPlanetDescriptor = vk::DescriptorImageInfo{ textures.planet.sampler, textures.planet.view, vk::ImageLayout::eGeneral };

        device.updateDescriptorSets(
            {
                // Binding 0 : Vertex shader uniform buffer
                { descriptorSets.instancedRocks, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.scene.descriptor },
                // Binding 1 : Color map
                { descriptorSets.instancedRocks, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texRocksDescriptor },
                // Binding 0 : Vertex shader uniform buffer
                { descriptorSets.planet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.scene.descriptor },
                // Binding 1 : Color map
                { descriptorSets.planet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texPlanetDescriptor },
            },
            nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

        // Binding description
        pipelineBuilder.vertexInputState.bindingDescriptions = {
            // Mesh vertex buffer (description) at binding point 0
            // Step for each vertex rendered
            { 0, vertexLayout.stride(), vk::VertexInputRate::eVertex },
            // Step for each instance rendered
            { 1, sizeof(InstanceData), vk::VertexInputRate::eInstance },
        };

        // Attribute descriptions
        // Describes memory layout and shader positions
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Per-Vertex attributes
            // Location 0 : Position
            { 0, 0, vk::Format::eR32G32B32Sfloat, vertexLayout.offset(0) },
            // Location 1 : Normal
            { 1, 0, vk::Format::eR32G32B32Sfloat, vertexLayout.offset(1) },
            // Location 2 : Texture coordinates
            { 2, 0, vk::Format::eR32G32Sfloat, vertexLayout.offset(2) },
            // Location 3 : Color
            { 3, 0, vk::Format::eR32G32B32Sfloat, vertexLayout.offset(3) },

            // Instanced attributes
            // Location 4 : Instance Position
            { 4, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, pos) },
            // Location 5 : Instance Rotation
            { 5, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, rot) },
            // Location 6 : Instance Scale
            { 6, 1, vk::Format::eR32Sfloat, offsetof(InstanceData, scale) },
            // Location 7 : Instance array layer
            { 7, 1, vk::Format::eR32Sint, offsetof(InstanceData, texIndex) },
        };

        // Load shaders
        pipelineBuilder.loadShader(getAssetPath() + "shaders/instancing/instancing.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/instancing/instancing.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Instacing pipeline
        pipelines.instancedRocks = pipelineBuilder.create(context.pipelineCache);


        pipelineBuilder.destroyShaderModules();
        pipelineBuilder.loadShader(getAssetPath() + "shaders/instancing/planet.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/instancing/planet.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelineBuilder.vertexInputState.attributeDescriptions.resize(4);
        pipelineBuilder.vertexInputState.bindingDescriptions.resize(1);
        pipelines.planet = pipelineBuilder.create(context.pipelineCache);

        pipelineBuilder.destroyShaderModules();
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.depthStencilState.depthWriteEnable = VK_FALSE;
        pipelineBuilder.vertexInputState = {};
        pipelineBuilder.loadShader(getAssetPath() + "shaders/instancing/starfield.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/instancing/starfield.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.starfield = pipelineBuilder.create(context.pipelineCache);
    }

    float rnd(float range) { return range * (rand() / float(RAND_MAX)); }
    uint32_t rnd(uint32_t range) { return (uint32_t)rnd((float)range); }

    void prepareInstanceData() {
        std::vector<InstanceData> instanceData;
        instanceData.resize(INSTANCE_COUNT);

        std::default_random_engine rndGenerator(benchmark.active ? 0 : (unsigned)time(nullptr));
        std::uniform_real_distribution<float> uniformDist(0.0, 1.0);
        std::uniform_int_distribution<uint32_t> rndTextureIndex(0, textures.rocks.layerCount);

        // Distribute rocks randomly on two different rings
        for (auto i = 0; i < INSTANCE_COUNT / 2; i++) {
            glm::vec2 ring0{ 7.0f, 11.0f };
            glm::vec2 ring1{ 14.0f, 18.0f };

            float rho, theta;

            // Inner ring
            rho = sqrt((pow(ring0[1], 2.0f) - pow(ring0[0], 2.0f)) * uniformDist(rndGenerator) + pow(ring0[0], 2.0f));
            theta = 2.0 * M_PI * uniformDist(rndGenerator);
            instanceData[i].pos = glm::vec3(rho*cos(theta), uniformDist(rndGenerator) * 0.5f - 0.25f, rho*sin(theta));
            instanceData[i].rot = glm::vec3(M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator));
            instanceData[i].scale = 1.5f + uniformDist(rndGenerator) - uniformDist(rndGenerator);
            instanceData[i].texIndex = rndTextureIndex(rndGenerator);
            instanceData[i].scale *= 0.75f;

            // Outer ring
            rho = sqrt((pow(ring1[1], 2.0f) - pow(ring1[0], 2.0f)) * uniformDist(rndGenerator) + pow(ring1[0], 2.0f));
            theta = 2.0 * M_PI * uniformDist(rndGenerator);
            instanceData[i + INSTANCE_COUNT / 2].pos = glm::vec3(rho*cos(theta), uniformDist(rndGenerator) * 0.5f - 0.25f, rho*sin(theta));
            instanceData[i + INSTANCE_COUNT / 2].rot = glm::vec3(M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator), M_PI * uniformDist(rndGenerator));
            instanceData[i + INSTANCE_COUNT / 2].scale = 1.5f + uniformDist(rndGenerator) - uniformDist(rndGenerator);
            instanceData[i + INSTANCE_COUNT / 2].texIndex = rndTextureIndex(rndGenerator);
            instanceData[i + INSTANCE_COUNT / 2].scale *= 0.75f;
        }

        // Staging
        // Instanced data is static, copy to device local memory
        // This results in better performance
        instanceBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, instanceData);
    }

    void prepareUniformBuffers() {
        uniformData.scene = context.createUniformBuffer(uboVS);
        updateUniformBuffer(true);
    }

    void updateUniformBuffer(bool viewChanged) {
        if (viewChanged) {
            uboVS.projection = getProjection();
            uboVS.view = camera.matrices.view;
        }

        if (!paused) {
            uboVS.locSpeed += frameTimer * 0.35f;
            uboVS.globSpeed += frameTimer * 0.01f;
        }
        uniformData.scene.copy(uboVS);
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareInstanceData();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    void render() override {
        if (!prepared) {
            return;
        }
        draw();
        if (!paused) {
            updateUniformBuffer(false);
        }
    }

    void viewChanged() override { updateUniformBuffer(true); }
};

RUN_EXAMPLE(VulkanExample)
