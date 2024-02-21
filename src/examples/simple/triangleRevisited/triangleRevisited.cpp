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

#include <examples/example.hpp>
#include <shaders/triangle/triangle.frag.inl>
#include <shaders/triangle/triangle.vert.inl>
#include <vks/descriptorsets.hpp>

class VulkanExample : public vkx::ExampleBase {
public:
    // vks::Buffer is a helper structure to encapsulate a buffer,
    // the memory for that buffer, and a descriptor for the buffer (if necessary)
    // We'll see more of what it does when we start using it
    //
    // We need one each for vertex, index and uniform data
    vks::Buffer vertices;
    vks::Buffer indices;
    vks::Buffer uniformDataVS;

    // As before
    // vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::Pipeline pipeline;

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
        device.destroy(pipeline);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
    }

    void prepare() override {
        // Even though we moved some of the preparations to the base class, we still have more to do locally
        // so we call be base class prepare method and then our preparation methods.  The base class sets up
        // the swapchain, renderpass, framebuffers, command pool and debugging.  It also creates some
        // helper classes for loading textures and for rendering text overlays, but we will not use them yet
        ExampleBase::prepare();
        prepareVertices();
        prepareUniformBuffers();
        prepareDescriptors();
        preparePipelines();
        // Update the drawCmdBuffers with the required drawing commands
        buildCommandBuffers();
        prepared = true;
    }

    // In our previous example, we created a function buildCommandBuffers that did two jobs.  First, it allocated a
    // command buffer for each handle image, and then it populated those command buffers with the commands required
    // to render our triangle.
    //
    // Some of this is now done by the base class, which calls this method to populate the actual commands for each
    // handle image specific CommandBuffer
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
        cmdBuffer.drawIndexed(3, 1, 0, 0, 1);
    }

    // The prepareVertices method has changed from the previous implementation.  All of the logic that was done to
    // populate the device local buffers has been moved into a helper function, "stageToDeviceBuffer"
    //
    // loader.stageToDeviceBuffer takes care of all of the work of creating a temporary host visible buffer, copying the
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
        vertices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);

        // Setup indices
        std::vector<uint32_t> indexBuffer{ { 0, 1, 2 } };
        indices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    ////////////////////////////////////////
    //
    // All as before
    //
    void prepareUniformBuffers() {
        uboVS.projectionMatrix = getProjection();
        uboVS.viewMatrix = glm::translate(glm::mat4(), camera.position);
        uboVS.modelMatrix = glm::inverse(camera.matrices.skyboxView);
        uniformDataVS = loader.createUniformBuffer(uboVS);
    }

    void prepareDescriptors() {
        // Create descriptor pool
        {
            vk::DescriptorPoolSize typeCounts{ vk::DescriptorType::eUniformBuffer, 1 };
            descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 1, typeCounts });
        }

        const std::vector<vk::DescriptorSetLayoutBinding> layoutBinding{
            // Binding 0 : Uniform buffer (Vertex shader)
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };
        // Create descriptor set layout
        descriptorSetLayout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{ {}, layoutBinding });
        // Allocate a new descriptor set from the global descriptor pool
        descriptorSet = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{ descriptorPool, descriptorSetLayout })[0];

        vks::descriptor::Writer writer;
        writer.parse(layoutBinding);
        writer.setUniformBuffer(descriptorSet, 0, uniformDataVS.buffer);
        assert(writer.valid());
        device.updateDescriptorSets(writer.writes, nullptr);
    }

    void preparePipelines() {
        // Create the pipeline layout that is used to generate the rendering pipelines that
        // are based on this descriptor set layout
        // In a more complex scenario you would have different pipeline layouts for different
        // descriptor set layouts that could be reused
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, descriptorSetLayout });

        // Vertex input state
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.vertexInputState.bindingDescriptions = {
            { 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
        };
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            { 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos) },
            { 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, col) },
        };
        pipelineBuilder.dynamicRendering(swapChain.surfaceFormat.format, deviceInfo.supportedDepthFormat);
        pipelineBuilder.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineBuilder.depthStencilState = false;
        // Load shaders
        // Shaders are loaded from the SPIR-V format, which can be generated from glsl
        pipelineBuilder.loadShader(vkx::shaders::triangle::triangle::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::triangle::triangle::frag, vk::ShaderStageFlagBits::eFragment);
        // Create rendering pipeline
        pipeline = pipelineBuilder.create(pipelineCache);
    }
};

RUN_EXAMPLE(VulkanExample)
