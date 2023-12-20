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

#include <glad/glad.h>
#include <glfw/glfw.hpp>
#include <common.hpp>
#include <vks/context.hpp>
#include <vks/swapchain.hpp>
#include <vks/shaders.hpp>
#include <utils.hpp>

#if defined(__ANDROID__)

class TriangleExample {
public:
    void run() {}
};

#else

class TriangleExample : public glfw::Window {
public:
    float zoom{ -2.5f };
    std::string title{ "Vulkan Example - Basic indexed triangle" };
    vk::Extent2D size{ 1280, 720 };
    vks::Context context;
    const vk::Device& device{ context.device };
    const vk::Queue& queue{ context.queue };
    vks::SwapChain swapChain;
    uint32_t currentBuffer;
    vk::CommandPool cmdPool;
    vk::DescriptorPool descriptorPool;
    vk::RenderPass renderPass;
    // List of available frame buffers (same as number of swap chain images)
    std::vector<vk::Framebuffer> framebuffers;
    std::vector<vk::CommandBuffer> commandBuffers;
    // Synchronization semaphores
    struct {
        vk::Semaphore presentComplete;
        vk::Semaphore renderComplete;
    } semaphores;

    struct {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
        vk::DescriptorBufferInfo descriptor;
    } uniformDataVS;

    struct {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } uboVS;

    struct {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } vertices;

    struct {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
    } indices;

    int indexCount;
    vk::PipelineVertexInputStateCreateInfo inputState;
    std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    void onWindowResized(const glm::uvec2& newSize) override {
        queue.waitIdle();
        device.waitIdle();
        size.width = newSize.x;
        size.height = newSize.y;
        swapChain.create(size);
        setupFrameBuffer();
        buildDrawCommandBuffers();
    }

    void run() {
        prepare();
        runWindowLoop([&] { draw(); });
        queue.waitIdle();
        device.waitIdle();
        destroy();
    }

    void prepare() {
        glfw::Window::init();
        // We don't want OpenGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        createWindow({ size.width, size.height }, { 100, 100 });

        context.setValidationEnabled(true);
        context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
        context.requireDeviceExtensions({ VK_KHR_SWAPCHAIN_EXTENSION_NAME });
        context.createInstance();

        // The `surface` should be created before the Vulkan `device` because the device selection needs to pick a queue 
        // that will support presentation to the surface
        auto surface = createSurface(context.instance);

        context.createDevice(surface);

        cmdPool = context.getCommandPool();

        swapChain.setup(context.physicalDevice, context.device, context.queue, context.queueIndices.graphics);
        swapChain.setSurface(surface);
        swapChain.create(size);

        setupRenderPass();
        setupFrameBuffer();

        prepareSemaphore();
        prepareVertices();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildDrawCommandBuffers();
    }

    void destroy() {
        // Clean up used Vulkan resources
        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyBuffer(vertices.buffer);
        device.freeMemory(vertices.memory);

        device.destroyBuffer(indices.buffer);
        device.freeMemory(indices.memory);

        device.destroySemaphore(semaphores.presentComplete);
        device.destroySemaphore(semaphores.renderComplete);

        device.destroyBuffer(uniformDataVS.buffer);
        device.freeMemory(uniformDataVS.memory);

        device.destroyRenderPass(renderPass);
        device.destroyDescriptorPool(descriptorPool);

        for (const auto& framebuffer : framebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        for (const auto& image : swapChain.images) {
            if (image.fence) {
                device.destroyFence(image.fence);
            }
        }
        swapChain.destroy();
        context.destroy();
    }

    void setupRenderPass() {
        if (renderPass) {
            device.destroyRenderPass(renderPass);
        }

        std::array<vk::AttachmentDescription, 1> attachments;
        std::array<vk::AttachmentReference, 1> attachmentReferences;

        // Color attachment
        attachments[0].format = swapChain.colorFormat;
        attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
        attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
        attachments[0].initialLayout = vk::ImageLayout::eUndefined;
        attachments[0].finalLayout = vk::ImageLayout::ePresentSrcKHR;

        // Only one depth attachment, so put it first in the references
        vk::AttachmentReference& colorReference = attachmentReferences[0];
        colorReference.attachment = 0;
        colorReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

        std::array<vk::SubpassDescription, 1> subpasses;
        {
            vk::SubpassDescription& subpass = subpasses[0];
            subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = attachmentReferences.data();
        }

        std::array<vk::SubpassDependency, 1> subpassDependencies;
        {
            vk::SubpassDependency& dependency = subpassDependencies[0];
            dependency.srcSubpass = 0;
            dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
            dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead;
            dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            dependency.srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        }

        vk::RenderPassCreateInfo renderPassInfo;
        renderPassInfo.attachmentCount = (uint32_t)attachments.size();
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = (uint32_t)subpasses.size();
        renderPassInfo.pSubpasses = subpasses.data();
        renderPassInfo.dependencyCount = (uint32_t)subpassDependencies.size();
        renderPassInfo.pDependencies = subpassDependencies.data();
        renderPass = device.createRenderPass(renderPassInfo);
    }

    void setupFrameBuffer() {
        if (!framebuffers.empty()) {
            for (const auto& framebuffer : framebuffers) {
                device.destroyFramebuffer(framebuffer);
            }
            framebuffers.clear();
        }

        std::array<vk::ImageView, 1> attachments;
        vk::FramebufferCreateInfo framebufferCreateInfo;
        framebufferCreateInfo.renderPass = renderPass;
        framebufferCreateInfo.attachmentCount = (uint32_t)attachments.size();
        framebufferCreateInfo.pAttachments = attachments.data();
        framebufferCreateInfo.width = size.width;
        framebufferCreateInfo.height = size.height;
        framebufferCreateInfo.layers = 1;

        // Create frame buffers for every swap chain image
        framebuffers = swapChain.createFramebuffers(framebufferCreateInfo);
    }

    void prepareSemaphore() {
        vk::SemaphoreCreateInfo semaphoreCreateInfo;

        // This semaphore ensures that the image is complete
        // before starting to submit again
        semaphores.presentComplete = device.createSemaphore(semaphoreCreateInfo);

        // This semaphore ensures that all commands submitted
        // have been finished before submitting the image to the queue
        semaphores.renderComplete = device.createSemaphore(semaphoreCreateInfo);
    }

    void prepareVertices() {
        struct Vertex {
            float pos[3];
            float col[3];
        };

        // Setup vertices
        std::vector<Vertex> vertexBuffer = { { { 1.0f, 1.0f, 0.0f },
                                               { 1.0f,
                                                 0.0f,
                                                 0.0f } },
                                             { { -1.0f, 1.0f, 0.0f },
                                               { 0.0f,
                                                 1.0f,
                                                 0.0f } },
                                             { { 0.0f, -1.0f, 0.0f },
                                               { 0.0f,
                                                 0.0f,
                                                 1.0f } } };
        uint32_t vertexBufferSize = (uint32_t)(vertexBuffer.size() * sizeof(Vertex));

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
        uint32_t indexBufferSize = (uint32_t)(indexBuffer.size() * sizeof(uint32_t));
        indexCount = (uint32_t)indexBuffer.size();

        vk::MemoryAllocateInfo memAlloc;
        vk::MemoryRequirements memReqs;

        void* data;

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

        struct StagingBuffer {
            vk::DeviceMemory memory;
            vk::Buffer buffer;
        };

        struct {
            StagingBuffer vertices;
            StagingBuffer indices;
        } stagingBuffers;

        // vk::Buffer copies are done on the queue, so we need a command buffer for them
        vk::CommandBufferAllocateInfo cmdBufInfo;
        cmdBufInfo.commandPool = cmdPool;
        cmdBufInfo.level = vk::CommandBufferLevel::ePrimary;
        cmdBufInfo.commandBufferCount = 1;

        vk::CommandBuffer copyCommandBuffer = device.allocateCommandBuffers(cmdBufInfo)[0];

        // Vertex buffer
        vk::BufferCreateInfo vertexBufferInfo;
        vertexBufferInfo.size = vertexBufferSize;
        // vk::Buffer is used as the copy source
        vertexBufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        // Create a host-visible buffer to copy the vertex data to (staging buffer)
        stagingBuffers.vertices.buffer = device.createBuffer(vertexBufferInfo);
        memReqs = device.getBufferMemoryRequirements(stagingBuffers.vertices.buffer);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        stagingBuffers.vertices.memory = device.allocateMemory(memAlloc);
        // Map and copy
        data = device.mapMemory(stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, vk::MemoryMapFlags());
        memcpy(data, vertexBuffer.data(), vertexBufferSize);
        device.unmapMemory(stagingBuffers.vertices.memory);
        device.bindBufferMemory(stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0);

        // Create the destination buffer with device only visibility
        // vk::Buffer will be used as a vertex buffer and is the copy destination
        vertexBufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
        vertices.buffer = device.createBuffer(vertexBufferInfo);
        memReqs = device.getBufferMemoryRequirements(vertices.buffer);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vertices.memory = device.allocateMemory(memAlloc);
        device.bindBufferMemory(vertices.buffer, vertices.memory, 0);

        // Index buffer
        vk::BufferCreateInfo indexbufferInfo;
        indexbufferInfo.size = indexBufferSize;
        indexbufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
        // Copy index data to a buffer visible to the host (staging buffer)
        stagingBuffers.indices.buffer = device.createBuffer(indexbufferInfo);
        memReqs = device.getBufferMemoryRequirements(stagingBuffers.indices.buffer);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        stagingBuffers.indices.memory = device.allocateMemory(memAlloc);
        data = device.mapMemory(stagingBuffers.indices.memory, 0, indexBufferSize, vk::MemoryMapFlags());
        memcpy(data, indexBuffer.data(), indexBufferSize);
        device.unmapMemory(stagingBuffers.indices.memory);
        device.bindBufferMemory(stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0);

        // Create destination buffer with device only visibility
        indexbufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
        indices.buffer = device.createBuffer(indexbufferInfo);
        memReqs = device.getBufferMemoryRequirements(indices.buffer);
        memAlloc.allocationSize = memReqs.size;
        memAlloc.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        indices.memory = device.allocateMemory(memAlloc);
        device.bindBufferMemory(indices.buffer, indices.memory, 0);

        vk::CommandBufferBeginInfo cmdBufferBeginInfo;
        vk::BufferCopy copyRegion;

        // Put buffer region copies into command buffer
        // Note that the staging buffer must not be deleted before the copies
        // have been submitted and executed
        copyCommandBuffer.begin(cmdBufferBeginInfo);

        // Vertex buffer
        copyRegion.size = vertexBufferSize;
        copyCommandBuffer.copyBuffer(stagingBuffers.vertices.buffer, vertices.buffer, copyRegion);
        // Index buffer
        copyRegion.size = indexBufferSize;
        copyCommandBuffer.copyBuffer(stagingBuffers.indices.buffer, indices.buffer, copyRegion);
        copyCommandBuffer.end();

        // Submit copies to the queue
        vk::SubmitInfo copySubmitInfo;
        copySubmitInfo.commandBufferCount = 1;
        copySubmitInfo.pCommandBuffers = &copyCommandBuffer;

        vk::Fence nullFence;
        queue.submit(copySubmitInfo, nullFence);
        queue.waitIdle();

        device.freeCommandBuffers(cmdPool, copyCommandBuffer);

        // Destroy staging buffers
        device.destroyBuffer(stagingBuffers.vertices.buffer);
        device.freeMemory(stagingBuffers.vertices.memory);
        device.destroyBuffer(stagingBuffers.indices.buffer);
        device.freeMemory(stagingBuffers.indices.memory);

        // Binding description
        bindingDescriptions.resize(1);
        bindingDescriptions[0].binding = 0;
        bindingDescriptions[0].stride = sizeof(Vertex);
        bindingDescriptions[0].inputRate = vk::VertexInputRate::eVertex;

        // Attribute descriptions
        // Describes memory layout and shader attribute locations
        attributeDescriptions.resize(2);
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
        inputState.vertexBindingDescriptionCount = (uint32_t)bindingDescriptions.size();
        inputState.pVertexBindingDescriptions = bindingDescriptions.data();
        inputState.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
        inputState.pVertexAttributeDescriptions = attributeDescriptions.data();
    }

    void prepareUniformBuffers() {
        // Prepare and initialize a uniform buffer block containing shader uniforms
        // In Vulkan there are no more single uniforms like in GL
        // All shader uniforms are passed as uniform buffer blocks
        vk::MemoryRequirements memReqs;

        // Vertex shader uniform buffer block
        vk::BufferCreateInfo bufferInfo;
        vk::MemoryAllocateInfo allocInfo;
        allocInfo.allocationSize = 0;
        allocInfo.memoryTypeIndex = 0;
        bufferInfo.size = sizeof(uboVS);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        // Create a new buffer
        uniformDataVS.buffer = device.createBuffer(bufferInfo);
        // Get memory requirements including size, alignment and memory type
        memReqs = device.getBufferMemoryRequirements(uniformDataVS.buffer);
        allocInfo.allocationSize = memReqs.size;
        // Get the memory type index that supports host visibile memory access
        // Most implementations offer multiple memory tpyes and selecting the
        // correct one to allocate memory from is important
        allocInfo.memoryTypeIndex = context.getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
        // Allocate memory for the uniform buffer
        (uniformDataVS.memory) = device.allocateMemory(allocInfo);
        // Bind memory to buffer
        device.bindBufferMemory(uniformDataVS.buffer, uniformDataVS.memory, 0);

        // Store information in the uniform's descriptor
        uniformDataVS.descriptor.buffer = uniformDataVS.buffer;
        uniformDataVS.descriptor.offset = 0;
        uniformDataVS.descriptor.range = sizeof(uboVS);

        // Update matrices
        uboVS.projectionMatrix = glm::perspective(glm::radians(60.0f), (float)size.width / (float)size.height, 0.1f, 256.0f);
        uboVS.viewMatrix = glm::translate(glm::mat4(), glm::vec3(0.0f, 0.0f, zoom));
        uboVS.modelMatrix = glm::mat4();

        // Map uniform buffer and update it
        // If you want to keep a handle to the memory and not unmap it afer updating,
        // create the memory with the vk::MemoryPropertyFlagBits::eHostCoherent
        void* pData = device.mapMemory(uniformDataVS.memory, 0, sizeof(uboVS), vk::MemoryMapFlags());
        memcpy(pData, &uboVS, sizeof(uboVS));
        device.unmapMemory(uniformDataVS.memory);
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
        // Vulkan uses the concept of rendering pipelines to encapsulate
        // fixed states
        // This replaces OpenGL's huge (and cumbersome) state machine
        // A pipeline is then stored and hashed on the GPU making
        // pipeline changes much faster than having to set dozens of
        // states
        // In a real world application you'd have dozens of pipelines
        // for every shader set used in a scene
        // Note that there are a few states that are not stored with
        // the pipeline. These are called dynamic states and the
        // pipeline only stores that they are used with this pipeline,
        // but not their states

        vk::GraphicsPipelineCreateInfo pipelineCreateInfo;
        // The layout used for this pipeline
        pipelineCreateInfo.layout = pipelineLayout;
        // Renderpass this pipeline is attached to
        pipelineCreateInfo.renderPass = renderPass;

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
        vk::PipelineColorBlendAttachmentState blendAttachmentState[1] = {};
        blendAttachmentState[0].colorWriteMask =
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        blendAttachmentState[0].blendEnable = VK_FALSE;
        colorBlendState.attachmentCount = 1;
        colorBlendState.pAttachments = blendAttachmentState;

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
        shaderStages[0] = vks::shaders::loadShader(device, vkx::getAssetPath() + "shaders/triangle/triangle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        shaderStages[1] = vks::shaders::loadShader(device, vkx::getAssetPath() + "shaders/triangle/triangle.frag.spv", vk::ShaderStageFlagBits::eFragment);

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
        pipelineCreateInfo.renderPass = renderPass;
        pipelineCreateInfo.pDynamicState = &dynamicState;

        // Create rendering pipeline
        pipeline = device.createGraphicsPipelines(context.pipelineCache, pipelineCreateInfo, nullptr).value[0];

        for (const auto& shaderStage : shaderStages) {
            device.destroyShaderModule(shaderStage.module);
        }
    }

    void buildDrawCommandBuffers() {
        // Create one command buffer per image in the swap chain

        // Command buffers store a reference to the
        // frame buffer inside their render pass info
        // so for static usage without having to rebuild
        // them each frame, we use one per frame buffer
        vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
        cmdBufAllocateInfo.commandPool = cmdPool;
        cmdBufAllocateInfo.commandBufferCount = swapChain.imageCount;
        commandBuffers = device.allocateCommandBuffers(cmdBufAllocateInfo);

        vk::CommandBufferBeginInfo cmdBufInfo;
        vk::ClearValue clearValues[2];
        clearValues[0].color = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));
        ;

        vk::RenderPassBeginInfo renderPassBeginInfo;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent = size;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = clearValues;
        glm::vec2 offset;
        float minDepth = 0;
        float maxDepth = 1;
        vk::Viewport viewport = vk::Viewport{ offset.x, offset.y, (float)size.width, (float)size.height, minDepth, maxDepth };
        vk::Rect2D scissor = vk::Rect2D{ vk::Offset2D(), size };
        vk::DeviceSize offsets = 0;
        for (size_t i = 0; i < swapChain.imageCount; ++i) {
            const auto& cmdBuffer = commandBuffers[i];
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(cmdBufInfo);
            renderPassBeginInfo.framebuffer = framebuffers[i];
            cmdBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
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
            cmdBuffer.endRenderPass();
            cmdBuffer.end();
        }
    }

    void draw() {
        // Get next image in the swap chain (back/front buffer)
        currentBuffer = swapChain.acquireNextImage(semaphores.presentComplete).value;

        // The submit infor strcuture contains a list of
        // command buffers and semaphores to be submitted to a queue
        // If you want to submit multiple command buffers, pass an array
        vk::PipelineStageFlags pipelineStages = vk::PipelineStageFlagBits::eBottomOfPipe;
        vk::SubmitInfo submitInfo;
        submitInfo.pWaitDstStageMask = &pipelineStages;
        // The wait semaphore ensures that the image is presented
        // before we start submitting command buffers agein
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &semaphores.presentComplete;
        // Submit the currently active command buffer
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentBuffer];
        // The signal semaphore is used during queue presentation
        // to ensure that the image is not rendered before all
        // commands have been submitted
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &semaphores.renderComplete;

        // Submit to the graphics queue
        // TODO explain submit fence
        queue.submit(submitInfo, swapChain.getSubmitFence(true));

        // Present the current buffer to the swap chain
        // We pass the signal semaphore from the submit info
        // to ensure that the image is not rendered until
        // all commands have been submitted
        swapChain.queuePresent(semaphores.renderComplete);
    }
};
#endif

RUN_EXAMPLE(TriangleExample)
