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

#include <vulkanExampleBase.h>

class VulkanExample : public vkx::ExampleBase {
    // Moved to base class
    /*
    float zoom{ -2.5f };
    std::string title{ "Vulkan Example - Basic indexed triangle" };
    vk::Extent2D size{ 1280, 720 };
    vkx::SwapChain swapChain;
    uint32_t currentBuffer;
    vk::CommandPool cmdPool;
    vk::DescriptorPool descriptorPool;
    vk::RenderPass renderPass;
    // List of shader modules created (stored for cleanup)
    std::vector<vk::ShaderModule> shaderModules;
    // List of available frame buffers (same as number of swap chain images)
    std::vector<vk::Framebuffer> framebuffers;
    std::vector<vk::CommandBuffer> commandBuffers;
    */

    // Moved to base class
    /*
    void run()
    void render()
    void prepare() 
    void setupRenderPass()
    void setupFrameBuffer()
    void prepareSemaphore()
    */

public:
    // vks::Buffer is a helper structure to encapsulate a buffer,
    // the memory for that buffer, and a descriptor for the buffer (if necessary)
    // We'll see more of what it does when we start using it
    //
    // We need one each for vertex, index and uniform data
    vks::Buffer vertices;
    vks::Buffer indices;
    vks::Buffer uniformDataVS;
    uint32_t indexCount{ 0 };

    // As before
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::Pipeline pipeline;
    vk::PipelineLayout pipelineLayout;

    struct Vertex {
        float pos[3];
        float col[3];
    };

    // As before
    struct UboVS {
        glm::mat4 projectionMatrix;
        glm::mat4 modelMatrix;
        glm::mat4 viewMatrix;
    } uboVS;

    VulkanExample() {
        size.width = 1280;
        size.height = 720;
        camera.dolly(-2.5f);
        title = "Vulkan Example - triangle revisited";
    }

    ~VulkanExample() {
        // The helper class we use for encapsulating buffer has a destroy method
        // that cleans up all the resources it owns.
        vertices.destroy();
        indices.destroy();
        uniformDataVS.destroy();

        // As before
        device.destroyPipeline(pipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);
    }

    void prepare() override {
        // Even though we moved some of the preparations to the base class, we still have more to do locally
        // so we call be base class prepare method and then our preparation methods.  The base class sets up
        // the swapchain, renderpass, framebuffers, command pool and debugging.  It also creates some
        // helper classes for loading textures and for rendering text overlays, but we will not use them yet
        ExampleBase::prepare();
        prepareVertices();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();

        // Update the drawCmdBuffers with the required drawing commands
        buildCommandBuffers();
        prepared = true;
    }

    // In our previous example, we created a function buildCommandBuffers that did two jobs.  First, it allocated a
    // command buffer for each swapChain image, and then it populated those command buffers with the commands required
    // to render our triangle.
    //
    // Some of this is now done by the base class, which calls this method to populate the actual commands for each
    // swapChain image specific CommandBuffer
    //
    // Note that this method only works if we have a single renderpass, since the parent class calls beginRenderPass
    // and endRenderPass around this method.  If we have multiple render passes then we'd need to override the
    // parent class buildCommandBuffers to do the appropriate work
    //
    // For now, that is left for us to do is to set viewport & scissor regions, bind pipelines, and draw geometry.
    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, viewport());
        cmdBuffer.setScissor(0, scissor());
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        cmdBuffer.bindVertexBuffers(0, vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(indices.buffer, 0, vk::IndexType::eUint32);
        cmdBuffer.drawIndexed(indexCount, 1, 0, 0, 1);
    }

    // The prepareVertices method has changed from the previous implementation.  All of the logic that was done to
    // populate the device local buffers has been moved into a helper function, "stageToDeviceBuffer"
    //
    // context.stageToDeviceBuffer takes care of all of the work of creating a temporary host visible buffer, copying the
    // data to it, creating the actual device local buffer, copying the contents from one buffer to another,
    // and destroying the temporary buffer.
    //
    // Additionally, the staging function is templated, so you don't need to pass it a void pointer and a size,
    // but instead can pass it a std::vector containing an array of data, and it will automatically calculate
    // the required size.
    void prepareVertices() {
        // Setup vertices
        std::vector<Vertex> vertexBuffer{
            { { 1.0f, 1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
            { { -1.0f, 1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
            { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
        };
        vertices = context.stageToDeviceBuffer<Vertex>(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
        indexCount = (uint32_t)indexBuffer.size();
        indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    ////////////////////////////////////////
    //
    // All as before
    //
    void prepareUniformBuffers() {
        uboVS.projectionMatrix = getProjection();
        uboVS.viewMatrix = glm::translate(glm::mat4(), camera.position);
        uboVS.modelMatrix = glm::inverse(camera.matrices.skyboxView);
        uniformDataVS = context.createUniformBuffer(uboVS);
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

        // Vertex input state
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        pipelineBuilder.vertexInputState.bindingDescriptions = {
            { 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
        };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            { 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) },
            { 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, col) },
        };

        // No culling
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        // Depth and stencil state
        pipelineBuilder.depthStencilState = { false };
        // Load shaders
        // Shaders are loaded from the SPIR-V format, which can be generated from glsl
        pipelineBuilder.loadShader(getAssetPath() + "shaders/triangle/triangle.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/triangle/triangle.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Create rendering pipeline
        pipeline = pipelineBuilder.create(context.pipelineCache);
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

    void setupDescriptorSet() {
        // Allocate a new descriptor set from the global descriptor pool
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Update the descriptor set determining the shader binding points
        // For every binding point used in a shader there needs to be one
        // descriptor set matching that binding point

        // Binding 0 : Uniform buffer
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataVS.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }
};

RUN_EXAMPLE(VulkanExample)
