/*
 * Vulkan Example - Basic indexed triangle rendering
 *
 * Note :
 *    This is a "pedal to the metal" example to show off how to get Vulkan up an displaying something
 *    Contrary to the other examples, this one won't make use of helper functions or initializers
 *    Except in a few cases (swap chain setup e.g.)
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

// #include <glad/glad.h>
#include <common/common.hpp>
#include <common/utils.hpp>

#include <rendering/context.hpp>
#include <rendering/shaders.hpp>
#include <rendering/swapchain.hpp>

#include <examples/glfw.hpp>

#if USE_VMA
#include <vk_mem_alloc.h>
#endif

#include <shaders/triangle/triangle.frag.inl>
#include <shaders/triangle/triangle.vert.inl>

#if defined(__ANDROID__)

class TriangleExample {
public:
    void run() {}
};

#else

static vk::Extent2D EMPTY_WINDOW{ 0, 0 };

struct ImageTransitionState {
    vk::ImageLayout layout{ vk::ImageLayout::eUndefined };
    vk::AccessFlags2 accessMask{ vk::AccessFlagBits2::eNone };
    vk::PipelineStageFlags2 stageMask{ vk::PipelineStageFlagBits2::eNone };
    uint32_t queueFamilyIndex{ VK_QUEUE_FAMILY_IGNORED };

    ImageTransitionState() = default;
    ImageTransitionState(vk::ImageLayout layout,
                         vk::AccessFlags2 accessMask = {},
                         vk::PipelineStageFlags2 stageMask = {},
                         uint32_t queueFamilyIndex = VK_QUEUE_FAMILY_IGNORED)
        : layout(layout)
        , accessMask(accessMask)
        , stageMask(stageMask)
        , queueFamilyIndex(queueFamilyIndex) {}
};

struct AllocatedBuffer {
    static VmaAllocator allocator;
    static void init() {
        if (!allocator) {
            const auto& context = vks::SimpleContext::get();
            VmaAllocatorCreateInfo allocatorCreateInfo = {};

            // These flags aren't technically needed because they've both been promoted to core by Vulkan 1.3, which is our minimum target
            allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT | VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
            if (context.deviceInfo.hasExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
                allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
            }
            allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
            allocatorCreateInfo.physicalDevice = context.physicalDevice;
            allocatorCreateInfo.device = context.device;
            allocatorCreateInfo.instance = context.instance;
            vmaCreateAllocator(&allocatorCreateInfo, &allocator);
        }
    }

    static void shutdown() {
        if (allocator) {
            vmaDestroyAllocator(allocator);
        }
    }

    vk::Buffer buffer;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocInfo{};

    void create(const vk::BufferCreateInfo& bufferCreateInfo, VmaAllocationCreateFlags flags = 0, VmaMemoryUsage usage = VMA_MEMORY_USAGE_AUTO) {
        VkBuffer tmp;
        VmaAllocationCreateInfo allocCreateInfo{ .flags = flags, .usage = usage };
        vk::Result result = static_cast<vk::Result>(
            vmaCreateBuffer(allocator, &bufferCreateInfo.operator const VkBufferCreateInfo&(), &allocCreateInfo, &tmp, &allocation, &allocInfo));
        vk::resultCheck(result, "Allocation of a buffer");
        buffer = tmp;
    }

    void create(vk::DeviceSize size,
                const vk::BufferUsageFlags& bufferUsage,
                VmaAllocationCreateFlags flags = 0,
                VmaMemoryUsage usage = VMA_MEMORY_USAGE_AUTO) {
        create(vk::BufferCreateInfo{ {}, size, bufferUsage }, flags, usage);
    }

    void destroy() {
        vmaDestroyBuffer(allocator, buffer, allocation);
        buffer = nullptr;
        allocation = nullptr;
        allocInfo = {};
    }
};

VmaAllocator AllocatedBuffer::allocator = nullptr;

class TriangleExample {
public:
    glfw::Window window;
    float zoom{ -2.5f };
    std::string title{ "Vulkan Example - Basic indexed triangle" };
    vk::Extent2D size{ 1280, 720 };

    vks::Context& context = vks::SimpleContext::get();
    const vk::Device& device{ context.device };
    const vk::Instance& instance{ context.instance };
    const vk::PhysicalDevice& physicalDevice{ context.physicalDevice };
    const vks::QueueFamilyInfo& queueFamilyInfo{ context.queuesInfo.graphics };

    vks::Swapchain swapChain;
    uint32_t swapChainIndex;
    vk::CommandPool commandPool;
    vk::DescriptorPool descriptorPool;

    // List of available frame buffers (same as number of swap chain images)
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Fence> frameSubmissions;

    // Synchronization semaphores
    struct {
        vk::Semaphore imageAcquired;
        vk::Semaphore swapchainFilled;
    } semaphores;

    struct {
        AllocatedBuffer buffer;
        vk::DescriptorBufferInfo descriptor{ nullptr, 0, VK_WHOLE_SIZE };
    } uniformDataVS;

    struct {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } uboVS;

    AllocatedBuffer vertices;
    AllocatedBuffer indices;

    int indexCount;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::Queue queue;
    vk::SurfaceKHR surface;

    void waitIdle() const {
        queue.waitIdle();
        device.waitIdle();
    }

    void run() {
        createWindow();
        initVulkan();
        prepare();
        window.runWindowLoop([&] { draw(); });
        waitIdle();
        destroy();
    }

    void withCommandBuffer(const std::function<void(const vk::CommandBuffer& cmdBuffer)>& f) {
        vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{ commandPool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        commandBuffer.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        f(commandBuffer);
        commandBuffer.end();

        // Submit copies to the graphicsQueue
        vk::SubmitInfo copySubmitInfo;
        copySubmitInfo.commandBufferCount = 1;
        copySubmitInfo.pCommandBuffers = &commandBuffer;
        queue.submit(copySubmitInfo, vk::Fence{});
        waitIdle();
        device.free(commandPool, commandBuffer);
    }

    void createWindow() {
        glfw::Window::init();
        // We'll get into how to resize a window in later examples, for now make a window of a fixed size
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        // We don't want OpenGL, just window management
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window.createWindow(vk::Rect2D{ vk::Offset2D{ 100, 100 }, size }, "Triangle");
    }

    void initVulkan() {
        // Create the Vulkan instance, choose the physical device, and create the logical device after enabling the desired enabledFeatures
        context.setValidationEnabled(true);
        context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
        context.requireDeviceExtensions({ VK_KHR_SWAPCHAIN_EXTENSION_NAME });
        context.createInstance();

        // The `surface` should be created before the Vulkan `device` because the device selection needs to pick a graphicsQueue
        // that will support presentation to the surface
        surface = window.createSurface(context.instance);
        context.pickDevice(surface);
        auto& enabledFeatures = context.enabledFeatures;
        enabledFeatures.core12.timelineSemaphore = VK_TRUE;
        enabledFeatures.core13.dynamicRendering = VK_TRUE;
        enabledFeatures.core13.synchronization2 = VK_TRUE;
        enabledFeatures.core13.maintenance4 = VK_TRUE;
        context.createDevice();

        AllocatedBuffer::init();
    }

    void prepare() {
        // Get a graphicsQueue object for the Graphics graphicsQueue (which by specification will also support all required transfer and compute operations)
        queue = device.getQueue2({ {}, queueFamilyInfo.index, 0 });
        commandPool = device.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyInfo.index });

        // Create the swapchain
        swapChain.create(vks::swapchain::Builder{ size, surface });

        frameSubmissions.resize(swapChain.imageCount);

        // Create semaphores for synchronizing presents and draws
        {
            // This semaphore ensures that the image is complete
            // before starting to submit again
            semaphores.imageAcquired = device.createSemaphore(vk::SemaphoreCreateInfo{});

            // This semaphore ensures that all commands submitted
            // have been finished before submitting the image to the graphicsQueue
            semaphores.swapchainFilled = device.createSemaphore(vk::SemaphoreCreateInfo{});
        }

        prepareVertices();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildDrawCommandBuffers();
    }

    void destroy() {
        size = EMPTY_WINDOW;
        waitIdle();
        for (const auto& fence : frameSubmissions) {
            if (fence) {
                auto result = device.waitForFences(fence, VK_TRUE, UINT64_MAX);
                assert(result == vk::Result::eSuccess);
                device.destroy(fence);
            }
        }
        // Clean up used Vulkan resources
        device.destroy(pipeline);
        pipeline = nullptr;
        device.destroy(pipelineLayout);
        pipelineLayout = nullptr;
        device.destroy(descriptorSetLayout);
        descriptorSetLayout = nullptr;

        vertices.destroy();
        indices.destroy();
        uniformDataVS.buffer.destroy();

        device.destroy(semaphores.imageAcquired);
        semaphores.imageAcquired = nullptr;

        device.destroy(semaphores.swapchainFilled);
        semaphores.swapchainFilled = nullptr;

        device.destroy(descriptorPool);
        descriptorPool = nullptr;
        device.free(commandPool, commandBuffers);
        commandBuffers.clear();
        device.destroy(commandPool);
        commandPool = nullptr;
        swapChain.destroy();
        instance.destroy(surface);
        surface = nullptr;
        AllocatedBuffer::shutdown();
        context.destroy();
    }

    struct Vertex {
        float pos[3];
        float col[3];
    };

    void prepareVertices() {
        // Setup vertices
        std::vector<Vertex> vertexBuffer = { { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
        uint32_t vertexBufferSize = (uint32_t)(vertexBuffer.size() * sizeof(Vertex));

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
        uint32_t indexBufferSize = (uint32_t)(indexBuffer.size() * sizeof(uint32_t));
        indexCount = (uint32_t)indexBuffer.size();

        // Static data like vertex and index buffer should be stored on the device memory
        // for optimal (and fastest) access by the GPU
        //
        // To achieve this we use so-called "staging buffers" :
        // - Create a buffer that's visible to the host (and can be mapped)
        // - Copy the data to this buffer
        // - Create another buffer that's local on the device (VRAM) with the same size
        // - Copy the data from the host to the device using a command buffer
        // - Delete the host visible (staging) buffer
        // - Use the device local buffers for rendering

        // Create destination buffers
        indices.create(indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst);
        vertices.create(vertexBufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);

        // Create temp staging buffers that will let us copy memory to the GPU
        const auto stagingBufferFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        const auto stagingBufferUsage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        struct StagingBuffers {
            AllocatedBuffer vertices;
            AllocatedBuffer indices;
        } stagingBuffers;

        stagingBuffers.vertices.create(vertexBufferSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBufferFlags, stagingBufferUsage);
        memcpy(stagingBuffers.vertices.allocInfo.pMappedData, vertexBuffer.data(), vertexBufferSize);
        stagingBuffers.indices.create(indexBufferSize, vk::BufferUsageFlagBits::eTransferSrc, stagingBufferFlags, stagingBufferUsage);
        memcpy(stagingBuffers.indices.allocInfo.pMappedData, indexBuffer.data(), indexBufferSize);

        // copy
        withCommandBuffer([&](const vk::CommandBuffer& copyCommandBuffer) {
            // Vertex buffer
            copyCommandBuffer.copyBuffer(stagingBuffers.vertices.buffer, vertices.buffer, { vk::BufferCopy{ 0, 0, vertexBufferSize } });
            // Index buffer
            copyCommandBuffer.copyBuffer(stagingBuffers.indices.buffer, indices.buffer, { vk::BufferCopy{ 0, 0, indexBufferSize } });
        });
        // Destroy staging buffers
        stagingBuffers.vertices.destroy();
        stagingBuffers.indices.destroy();
    }

    void prepareUniformBuffers() {
        // Prepare and initialize a uniform buffer block containing shader uniforms
        // In Vulkan there are no more single uniforms like in GL
        // All shader uniforms are passed as uniform buffer blocks

        // Vertex shader uniform buffer block
        uniformDataVS.buffer.create(sizeof(uboVS), vk::BufferUsageFlagBits::eUniformBuffer,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

        // Store information in the uniform's descriptor
        uniformDataVS.descriptor.buffer = uniformDataVS.buffer.buffer;
        uniformDataVS.descriptor.offset = 0;
        uniformDataVS.descriptor.range = VK_WHOLE_SIZE;

        // Update matrices
        uboVS.projectionMatrix = glm::perspective(glm::radians(60.0f), (float)size.width / (float)size.height, 0.1f, 256.0f);
        uboVS.viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));
        uboVS.modelMatrix = glm::mat4();

        // Map uniform buffer and update it
        // If you want to keep a handle to the memory and not unmap it afer updating,
        // create the memory with the vk::MemoryPropertyFlagBits::eHostCoherent
        memcpy(uniformDataVS.buffer.allocInfo.pMappedData, &uboVS, sizeof(uboVS));
    }

    void setupDescriptorPool() {
        // We need to tell the API the number of max. requested descriptors per type
        vk::DescriptorPoolSize typeCounts[1];
        // This example only uses one descriptor type (uniform buffer) and only
        // requests one descriptor of this type
        typeCounts[0].type = vk::DescriptorType::eUniformBuffer;
        typeCounts[0].descriptorCount = 1;
        // For additional types you need to add new entries in the type count list
        // E.g. for two combined image samplers :
        // typeCounts[1].type = vk::DescriptorType::eCombinedImageSampler;
        // typeCounts[1].descriptorCount = 2;

        // Create the global descriptor pool
        // All descriptors used in this example are allocated from this pool
        vk::DescriptorPoolCreateInfo descriptorPoolInfo;
        descriptorPoolInfo.poolSizeCount = 1;
        descriptorPoolInfo.pPoolSizes = typeCounts;
        // Set the max. number of sets that can be requested
        // Requesting descriptors beyond maxSets will result in an error
        descriptorPoolInfo.maxSets = 1;

        descriptorPool = device.createDescriptorPool(descriptorPoolInfo);
    }

    void setupDescriptorSetLayout() {
        // Setup layout of descriptors used in this example
        // Basically connects the different shader stages to descriptors
        // for binding uniform buffers, image samplers, etc.
        // So every shader binding should map to one descriptor set layout
        // binding

        // Binding 0 : Uniform buffer (Vertex shader)
        vk::DescriptorSetLayoutBinding layoutBinding;
        layoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
        layoutBinding.pImmutableSamplers = NULL;

        vk::DescriptorSetLayoutCreateInfo descriptorLayout;
        descriptorLayout.bindingCount = 1;
        descriptorLayout.pBindings = &layoutBinding;

        descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout, nullptr);

        // Create the pipeline layout that is used to generate the rendering pipelines that
        // are based on this descriptor set layout
        // In a more complex scenario you would have different pipeline layouts for different
        // descriptor set layouts that could be reused
        vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo;
        pPipelineLayoutCreateInfo.setLayoutCount = 1;
        pPipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

        pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
    }

    void setupDescriptorSet() {
        // Allocate a new descriptor set from the global descriptor pool
        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        // Update the descriptor set determining the shader binding points
        // For every binding point used in a shader there needs to be one
        // descriptor set matching that binding point

        vk::WriteDescriptorSet writeDescriptorSet;

        // Binding 0 : Uniform buffer
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = vk::DescriptorType::eUniformBuffer;
        writeDescriptorSet.pBufferInfo = &uniformDataVS.descriptor;
        // Binds this uniform buffer to binding point 0
        writeDescriptorSet.dstBinding = 0;

        device.updateDescriptorSets(writeDescriptorSet, nullptr);
    }

    void preparePipelines() {
        // Create our rendering pipeline used in this example
        // Vulkan uses the concept of rendering pipelines to encapsulate fixed states
        //
        // This replaces OpenGL's huge (and cumbersome) state machine. A pipeline is
        // then stored and hashed on the GPU making pipeline changes much faster than
        // having to set dozens of states
        //
        // In a real world application you'd have dozens of pipelines for every shader
        // set used in a scene. Note that there are a few states that are not stored with
        // the pipeline. These are called dynamic states and the pipeline only stores that
        // they are used with this pipeline, but not their states

        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo;
        pipelineRenderingCreateInfo.colorAttachmentCount = 1;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = &swapChain.surfaceFormat.format;

        // Binding description
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions;
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = vk::VertexInputRate::eVertex;

        // Attribute descriptions
        // Describes memory layout and shader attribute locations
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;
        // Location 0 : Position
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = 0;
        // Location 1 : Color
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = sizeof(float) * 3;

        // Assign to vertex input state
        vk::PipelineVertexInputStateCreateInfo inputState;
        inputState.vertexBindingDescriptionCount = (uint32_t)bindingDescriptions.size();
        inputState.pVertexBindingDescriptions = bindingDescriptions.data();
        inputState.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
        inputState.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo;
        pipelineCreateInfo.pNext = &pipelineRenderingCreateInfo;
        // The layout used for this pipeline
        pipelineCreateInfo.layout = pipelineLayout;

        // Vertex input state
        // Describes the topoloy used with this pipeline
        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState;
        // This pipeline renders vertex data as triangle lists
        inputAssemblyState.topology = vk::PrimitiveTopology::eTriangleList;

        // Rasterization state
        vk::PipelineRasterizationStateCreateInfo rasterizationState;
        // Solid polygon mode
        rasterizationState.polygonMode = vk::PolygonMode::eFill;
        // No culling
        rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        rasterizationState.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizationState.depthClampEnable = VK_FALSE;
        rasterizationState.rasterizerDiscardEnable = VK_FALSE;
        rasterizationState.depthBiasEnable = VK_FALSE;
        rasterizationState.lineWidth = 1.0f;

        // Color blend state
        // Describes blend modes and color masks
        vk::PipelineColorBlendStateCreateInfo colorBlendState;
        // One blend attachment state
        // Blending is not used in this example
        vk::PipelineColorBlendAttachmentState blendAttachmentState;
        blendAttachmentState.colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
        blendAttachmentState.blendEnable = VK_FALSE;
        colorBlendState.attachmentCount = 1;
        colorBlendState.pAttachments = &blendAttachmentState;

        // vk::Viewport state
        vk::PipelineViewportStateCreateInfo viewportState;
        // One viewport
        viewportState.viewportCount = 1;
        // One scissor rectangle
        viewportState.scissorCount = 1;

        // Enable dynamic states
        // Describes the dynamic states to be used with this pipeline
        // Dynamic states can be set even after the pipeline has been created
        // So there is no need to create new pipelines just for changing
        // a viewport's dimensions or a scissor box
        vk::PipelineDynamicStateCreateInfo dynamicState;
        // The dynamic state properties themselves are stored in the command buffer
        std::vector<vk::DynamicState> dynamicStateEnables;
        dynamicStateEnables.push_back(vk::DynamicState::eViewport);
        dynamicStateEnables.push_back(vk::DynamicState::eScissor);
        dynamicState.dynamicStateCount = (uint32_t)dynamicStateEnables.size();
        dynamicState.pDynamicStates = dynamicStateEnables.data();

        // Depth and stencil state
        // Describes depth and stenctil test and compare ops
        vk::PipelineDepthStencilStateCreateInfo depthStencilState;
        // No depth or stencil testing enabled
        depthStencilState.depthTestEnable = VK_FALSE;
        depthStencilState.depthWriteEnable = VK_FALSE;
        depthStencilState.stencilTestEnable = VK_FALSE;

        // Multi sampling state
        vk::PipelineMultisampleStateCreateInfo multisampleState;
        multisampleState.pSampleMask = NULL;
        // No multi sampling used in this example
        multisampleState.rasterizationSamples = vk::SampleCountFlagBits::e1;

        // Load shaders
        // Shaders are loaded from the SPIR-V format, which can be generated from glsl
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

        shaderStages[0] = vks::shaders::loadShader(device, vkx::shaders::triangle::triangle::vert, vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = vks::shaders::loadShader(device, vkx::shaders::triangle::triangle::frag, vk::ShaderStageFlagBits::eFragment);

        // Assign states
        // Assign pipeline state create information
        pipelineCreateInfo.stageCount = (uint32_t)shaderStages.size();
        pipelineCreateInfo.pStages = shaderStages.data();
        pipelineCreateInfo.pVertexInputState = &inputState;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;
        pipelineCreateInfo.pMultisampleState = &multisampleState;
        pipelineCreateInfo.pViewportState = &viewportState;
        pipelineCreateInfo.pDepthStencilState = &depthStencilState;
        pipelineCreateInfo.pDynamicState = &dynamicState;

        // Create rendering pipeline
        pipeline = device.createGraphicsPipelines(context.pipelineCache, pipelineCreateInfo, nullptr).value[0];

        for (const auto& shaderStage : shaderStages) {
            device.destroy(shaderStage.module);
        }
    }

    // Fixed sub resource on first mip level and layer
    static vk::ImageMemoryBarrier2 buildImageBarrier(vk::Image image, const ImageTransitionState& srcState, const ImageTransitionState& dstState) {
        vk::ImageMemoryBarrier2 barrier;
        barrier.oldLayout = srcState.layout;
        barrier.srcAccessMask = srcState.accessMask;
        barrier.srcStageMask = srcState.stageMask;
        barrier.srcQueueFamilyIndex = srcState.queueFamilyIndex;
        barrier.newLayout = dstState.layout;
        barrier.dstAccessMask = dstState.accessMask;
        barrier.dstStageMask = dstState.stageMask;
        barrier.dstQueueFamilyIndex = dstState.queueFamilyIndex;
        barrier.image = image;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        return barrier;
    }

    // Fixed sub resource on first mip level and layer
    static void setImageLayout(const vk::CommandBuffer& cmdBuffer,
                               vk::Image image,
                               const ImageTransitionState& srcState,
                               const ImageTransitionState& dstState) {
        vk::ImageMemoryBarrier2 barrier = buildImageBarrier(image, srcState, dstState);
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, barrier });
    }

    void buildDrawCommandBuffers() {
        // Create one command buffer per image in the swap chain.  We also need to recreate the command buffers every time we re-create the swapchain because
        commandBuffers = device.allocateCommandBuffers({ commandPool, vk::CommandBufferLevel::ePrimary, swapChain.imageCount });
        float minDepth = 0;
        float maxDepth = 1;
        vk::Viewport viewport = vk::Viewport{ 0.0f, 0.0f, (float)size.width, (float)size.height, minDepth, maxDepth };
        vk::Rect2D scissor = vk::Rect2D{ vk::Offset2D{}, size };
        vk::DeviceSize offsets = 0;
        vk::RenderingInfo renderingInfo;
        vk::RenderingAttachmentInfo renderingAttachmentInfo;
        renderingAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
        renderingAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        renderingAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        renderingAttachmentInfo.clearValue = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.layerCount = 1;
        renderingInfo.pColorAttachments = &renderingAttachmentInfo;
        renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, size };
        ImageTransitionState undefinedState;
        ImageTransitionState renderState{ vk::ImageLayout::eAttachmentOptimal, vk::AccessFlagBits2::eColorAttachmentWrite,
                                          vk::PipelineStageFlagBits2::eColorAttachmentOutput };
        ImageTransitionState presentState{ vk::ImageLayout::ePresentSrcKHR };

        for (size_t i = 0; i < swapChain.imageCount; ++i) {
            const auto& swapChainImage = swapChain.images[i];
            const auto& cmdBuffer = commandBuffers[i];
            renderingAttachmentInfo.imageView = swapChainImage.view;
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(vk::CommandBufferBeginInfo{});
            setImageLayout(cmdBuffer, swapChainImage.image, undefinedState, renderState);
            cmdBuffer.beginRendering(renderingInfo);
            // Update dynamic viewport state
            cmdBuffer.setViewport(0, viewport);
            // Update dynamic scissor state
            cmdBuffer.setScissor(0, scissor);
            // Bind descriptor sets describing shader binding points
            cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
            // Bind the rendering pipeline (including the shaders)
            cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            // Bind triangle vertices
            cmdBuffer.bindVertexBuffers(0, vertices.buffer, offsets);
            // Bind triangle indices
            cmdBuffer.bindIndexBuffer(indices.buffer, 0, vk::IndexType::eUint32);
            // Draw indexed triangle
            cmdBuffer.drawIndexed(indexCount, 1, 0, 0, 1);
            cmdBuffer.endRendering();
            setImageLayout(cmdBuffer, swapChainImage.image, renderState, presentState);
            cmdBuffer.end();
        }
    }

    void draw() {
        // Don't draw when we're minimized
        if (size == EMPTY_WINDOW) {
            return;
        }

        // Get next image in the swap chain (back/front buffer)
        swapChainIndex = swapChain.acquireNextImage(semaphores.imageAcquired).value;
        // Get the last submission fence if any. If we don't wait on a fence for the last use of this image, the
        // code could easily start to get hundreds of frames ahead of the current one
        auto& fence = frameSubmissions[swapChainIndex];
        if (fence) {
            auto result = device.waitForFences(fence, VK_TRUE, UINT64_MAX);
            if (result != vk::Result::eSuccess) {
                throw std::runtime_error("Error waiting for fence");
            }
            device.destroy(fence);
            fence = nullptr;
        }
        fence = device.createFence(vk::FenceCreateInfo{});
        // Pick the command buffer that we want to use for this frame
        vk::CommandBufferSubmitInfo cmdBufferInfo{ commandBuffers[swapChainIndex] };
        // We need our drawing to wait on the image to actually be acquired
        vk::SemaphoreSubmitInfo wait{ semaphores.imageAcquired, 0, vk::PipelineStageFlagBits2::eNone };
        // After our rendering completes we want to signal that the swapchain image has been filled, so that we can use that
        // semaphore for the present call
        vk::SemaphoreSubmitInfo signal{ semaphores.swapchainFilled, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput };
        queue.submit2(vk::SubmitInfo2{ {}, wait, cmdBufferInfo, signal }, fence);

        // Present the current buffer to the swap chain We pass the signal semaphore from the submit info to ensure that
        // the image is not presented until all rendering has been completed
        auto result = queue.presentKHR(vk::PresentInfoKHR{ semaphores.swapchainFilled, swapChain.handle, swapChainIndex });
    }
};
#endif

RUN_EXAMPLE(TriangleExample)
