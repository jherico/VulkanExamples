/*
* Vulkan Example - Dynamic uniform buffers
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
* Summary:
* Demonstrates the use of dynamic uniform buffers.
*
* Instead of using one uniform buffer per-object, this example allocates one big uniform buffer
* with respect to the alignment reported by the device via minUniformBufferOffsetAlignment that
* contains all matrices for the objects in the scene.
*
* The used descriptor type VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC then allows to set a dynamic
* offset used to pass data from the single uniform buffer to the connected shader binding point.
*/

#include <vulkanExampleBase.h>

#define OBJECT_INSTANCES 125

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float color[3];
};

// Wrapper functions for aligned memory allocation
// There is currently no standard for this in C++ that works across all platforms and vendors, so we abstract this
void* alignedAlloc(size_t size, size_t alignment) {
    void* data = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
    data = _aligned_malloc(size, alignment);
#else
    int res = posix_memalign(&data, alignment, size);
    if (res != 0)
        data = nullptr;
#endif
    return data;
}

void alignedFree(void* data) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(data);
#else
    free(data);
#endif
}

class VulkanExample : public vkx::ExampleBase {
public:
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    uint32_t indexCount;

    struct {
        vks::Buffer view;
        vks::Buffer dynamic;
    } uniformBuffers;

    struct {
        glm::mat4 projection;
        glm::mat4 view;
    } uboVS;

    // Store random per-object rotations
    glm::vec3 rotations[OBJECT_INSTANCES];
    glm::vec3 rotationSpeeds[OBJECT_INSTANCES];

    // One big uniform buffer that contains all matrices
    // Note that we need to manually allocate the data to cope for GPU-specific uniform buffer offset alignments
    struct UboDataDynamic {
        glm::mat4* model = nullptr;
    } uboDataDynamic;

    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    float animationTimer = 0.0f;

    size_t dynamicAlignment;

    VulkanExample() {
        title = "Vulkan Example - Dynamic uniform buffers";
        camera.type = Camera::CameraType::lookat;
        camera.setPosition(glm::vec3(0.0f, 0.0f, -30.0f));
        camera.setRotation(glm::vec3(0.0f));
        camera.setPerspective(60.0f, (float)size.width / (float)size.height, 0.1f, 256.0f);
        settings.overlay = true;
    }

    ~VulkanExample() {
        if (uboDataDynamic.model) {
            alignedFree(uboDataDynamic.model);
        }

        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class
        vkDestroyPipeline(device, pipeline, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vertexBuffer.destroy();
        indexBuffer.destroy();

        uniformBuffers.view.destroy();
        uniformBuffers.dynamic.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCommandBuffer) override {
        vk::Viewport viewport;
        viewport.width = (float)size.width;
        viewport.height = (float)size.height;
        viewport.minDepth = 0;
        viewport.maxDepth = 1;
        drawCommandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor;
        scissor.extent = size;
        drawCommandBuffer.setScissor(0, scissor);

        drawCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        VkDeviceSize offsets[1] = { 0 };
        drawCommandBuffer.bindVertexBuffers(0, vertexBuffer.buffer, { 0 });
        drawCommandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);

        // Render multiple objects using different model matrices by dynamically offsetting into one uniform buffer
        for (uint32_t j = 0; j < OBJECT_INSTANCES; j++) {
            // One dynamic offset per dynamic descriptor to offset into the ubo containing all model matrices
            uint32_t dynamicOffset = j * static_cast<uint32_t>(dynamicAlignment);
            // Bind the descriptor set for rendering a mesh using the dynamic offset
            drawCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, dynamicOffset);
            drawCommandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
        }
    }

    void generateCube() {
        // Setup vertices indices for a colored cube
        std::vector<Vertex> vertices = {
            { { -1.0f, -1.0f, 1.0f }, { 1.0f, 0.0f, 0.0f } },  { { 1.0f, -1.0f, 1.0f }, { 0.0f, 1.0f, 0.0f } },
            { { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f } },    { { -1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } },
            { { -1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f, 0.0f } }, { { 1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f, 0.0f } },
            { { 1.0f, 1.0f, -1.0f }, { 0.0f, 0.0f, 1.0f } },   { { -1.0f, 1.0f, -1.0f }, { 0.0f, 0.0f, 0.0f } },
        };

        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3,
        };

        indexCount = static_cast<uint32_t>(indices.size());

        // Create buffers
        // For the sake of simplicity we won't stage the vertex data to the gpu memory

        // Vertex buffer
        vertexBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertices);

        // Index buffer
        indexBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indices);
    }

    void setupDescriptorPool() {
        // Example uses one ubo and one image sampler
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBufferDynamic, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };
        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 2, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
            { 1, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex },
            { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffers.view.descriptor },
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBufferDynamic, nullptr, &uniformBuffers.dynamic.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, {});
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };

        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

        auto& vertexInputState = builder.vertexInputState;
        // Binding description
        vertexInputState.bindingDescriptions = {
            vk::VertexInputBindingDescription{ 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
        };

        // Attribute descriptions
        vertexInputState.attributeDescriptions = {
            vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) },    // Location 0 : Position
            vk::VertexInputAttributeDescription{ 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) },  // Location 1 : Color
        };

        builder.loadShader(getAssetPath() + "shaders/dynamicuniformbuffer/base.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/dynamicuniformbuffer/base.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipeline = builder.create(context.pipelineCache);
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Allocate data for the dynamic uniform buffer object
        // We allocate this manually as the alignment of the offset differs between GPUs

        // Calculate required alignment based on minimum device offset alignment
        size_t minUboAlignment = context.deviceProperties.limits.minUniformBufferOffsetAlignment;
        dynamicAlignment = sizeof(glm::mat4);
        if (minUboAlignment > 0) {
            dynamicAlignment = (dynamicAlignment + minUboAlignment - 1) & ~(minUboAlignment - 1);
        }

        size_t bufferSize = OBJECT_INSTANCES * dynamicAlignment;

        uboDataDynamic.model = (glm::mat4*)alignedAlloc(bufferSize, dynamicAlignment);
        assert(uboDataDynamic.model);

        std::cout << "minUniformBufferOffsetAlignment = " << minUboAlignment << std::endl;
        std::cout << "dynamicAlignment = " << dynamicAlignment << std::endl;

        // Vertex shader uniform buffer block

        // Static shared uniform buffer object with projection and view matrix
        uniformBuffers.view = context.createUniformBuffer(uboVS);

        // Uniform buffer object with per-object matrices
        uniformBuffers.dynamic = context.createBuffer(vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible, bufferSize);
        // Map persistent
        uniformBuffers.dynamic.map();

        // Prepare per-object matrices with offsets and random rotations
        std::mt19937 rndGen(static_cast<uint32_t>(time(0)));
        std::normal_distribution<float> rndDist(-1.0f, 1.0f);
        for (uint32_t i = 0; i < OBJECT_INSTANCES; i++) {
            rotations[i] = glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen)) * 2.0f * (float)M_PI;
            rotationSpeeds[i] = glm::vec3(rndDist(rndGen), rndDist(rndGen), rndDist(rndGen));
        }

        updateUniformBuffers();
        updateDynamicUniformBuffer(true);
    }

    void updateUniformBuffers() {
        // Fixed ubo with projection and view matrices
        uboVS.projection = camera.matrices.perspective;
        uboVS.view = camera.matrices.view;

        memcpy(uniformBuffers.view.mapped, &uboVS, sizeof(uboVS));
    }

    void updateDynamicUniformBuffer(bool force = false) {
        // Update at max. 60 fps
        animationTimer += frameTimer;
        if ((animationTimer <= 1.0f / 60.0f) && (!force)) {
            return;
        }

        // Dynamic ubo with per-object model matrices indexed by offsets in the command buffer
        uint32_t dim = static_cast<uint32_t>(pow(OBJECT_INSTANCES, (1.0f / 3.0f)));
        glm::vec3 offset(5.0f);

        for (uint32_t x = 0; x < dim; x++) {
            for (uint32_t y = 0; y < dim; y++) {
                for (uint32_t z = 0; z < dim; z++) {
                    uint32_t index = x * dim * dim + y * dim + z;

                    // Aligned offset
                    glm::mat4* modelMat = (glm::mat4*)(((uint64_t)uboDataDynamic.model + (index * dynamicAlignment)));

                    // Update rotations
                    rotations[index] += animationTimer * rotationSpeeds[index];

                    // Update matrices
                    glm::vec3 pos =
                        glm::vec3(-((dim * offset.x) / 2.0f) + offset.x / 2.0f + x * offset.x, -((dim * offset.y) / 2.0f) + offset.y / 2.0f + y * offset.y,
                                  -((dim * offset.z) / 2.0f) + offset.z / 2.0f + z * offset.z);
                    *modelMat = glm::translate(glm::mat4(1.0f), pos);
                    *modelMat = glm::rotate(*modelMat, rotations[index].x, glm::vec3(1.0f, 1.0f, 0.0f));
                    *modelMat = glm::rotate(*modelMat, rotations[index].y, glm::vec3(0.0f, 1.0f, 0.0f));
                    *modelMat = glm::rotate(*modelMat, rotations[index].z, glm::vec3(0.0f, 0.0f, 1.0f));
                }
            }
        }

        animationTimer = 0.0f;
        uniformBuffers.dynamic.copy(uniformBuffers.dynamic.size, uboDataDynamic.model);
        // Flush to make changes visible to the host
        uniformBuffers.dynamic.flush();
    }

    void prepare() override {
        ExampleBase::prepare();
        generateCube();
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
        if (!paused)
            updateDynamicUniformBuffer();
    }

    void viewChanged() override { updateUniformBuffers(); }
};

VULKAN_EXAMPLE_MAIN()
