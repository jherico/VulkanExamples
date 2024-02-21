/*
 * Vulkan Example - Compute shader ray tracing
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/compute.hpp>
#include <examples/example.hpp>
#include <shaders/raytracing/raytracing.comp.inl>
#include <shaders/raytracing/texture.frag.inl>
#include <shaders/raytracing/texture.vert.inl>

#define TEX_DIM 2048

// Vertex layout for this example
struct Vertex {
    float pos[3];
    float uv[2];
};

vks::model::VertexLayout vertexLayout{ {
    vks::model::VERTEX_COMPONENT_POSITION,
    vks::model::VERTEX_COMPONENT_UV,
} };

struct RenderTarget {
    vks::Image image;
    vk::ImageView view;

    void destroy() {
        const auto& device = vks::Context::get().device;
        if (view) {
            device.destroy(view);
            view = nullptr;
        }

        image.destroy();
    }
} renderTarget;

struct RaytracingCompute : public vkx::Compute {
    struct Ubo {
        glm::vec3 lightPos;
        // Aspect ratio of the viewport
        float aspectRatio;
        glm::vec4 fogColor = glm::vec4(0.0f);
        struct Camera {
            glm::vec3 pos = glm::vec3(0.0f, 1.5f, 4.0f);
            glm::vec3 lookat = glm::vec3(0.0f, 0.5f, 0.0f);
            float fov = 10.0f;
        } camera;
    } ubo;

    vks::Buffer uniformBuffer;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::DescriptorPool descriptorPool;

    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::ImageMemoryBarrier2 acquireBarrier, releaseBarrier;

    void prepare(uint32_t swapchainImageCount) override {
        vkx::Compute::prepare(swapchainImageCount);

        acquireBarrier.oldLayout = vk::ImageLayout::eGeneral;
        acquireBarrier.newLayout = vk::ImageLayout::eGeneral;
        acquireBarrier.image = renderTarget.image.image;
        acquireBarrier.subresourceRange = renderTarget.image.getWholeRange();
        acquireBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;
        acquireBarrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        acquireBarrier.srcQueueFamilyIndex = context.queuesInfo.graphics.index;
        acquireBarrier.dstQueueFamilyIndex = context.queuesInfo.compute.index;

        releaseBarrier.oldLayout = vk::ImageLayout::eGeneral;
        releaseBarrier.newLayout = vk::ImageLayout::eGeneral;
        releaseBarrier.image = renderTarget.image.image;
        releaseBarrier.subresourceRange = renderTarget.image.getWholeRange();
        releaseBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        releaseBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        releaseBarrier.srcQueueFamilyIndex = context.queuesInfo.compute.index;
        releaseBarrier.dstQueueFamilyIndex = context.queuesInfo.graphics.index;

        const auto& loader = vks::Loader::get();
        loader.withPrimaryCommandBuffer(computeQueue, [&](const vk::CommandBuffer& setupCmdBuffer) {
            using namespace vks::util;
            // Create image view
            setImageLayout(setupCmdBuffer, renderTarget.image, ImageTransitionState::UNDEFINED, ImageTransitionState::GENERAL);
            setupCmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, releaseBarrier });
        });

        // Vertex shader uniform buffer block
        uniformBuffer = loader.createUniformBuffer(ubo);
        updateUniformBuffers(0);

        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 1 },
            // Compute pipeline uses storage images image loads and stores
            { vk::DescriptorType::eStorageImage, 1 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Sampled image (write)
            { 0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 1 : Uniform buffer block
            { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });

        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        std::vector<vk::DescriptorImageInfo> computeTexDescriptors{
            { nullptr, renderTarget.view, vk::ImageLayout::eGeneral },
        };
        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            // Binding 0 : Output storage image
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &computeTexDescriptors[0] },
            // Binding 1 : Uniform buffer block
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformBuffer.descriptor },
        };
        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);

        // Create compute shader pipelines
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = pipelineLayout;
        using namespace vkx::shaders::raytracing;
        computePipelineCreateInfo.stage = vks::shaders::loadShader(device, raytracing::comp, vk::ShaderStageFlagBits::eCompute);
        pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;

        // Prepare and initialize uniform buffer containing shader uniforms
    }

    void destroy() override {
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        uniformBuffer.destroy();

        device.destroy(pipeline);
        device.destroy(descriptorPool);
        vkx::Compute::destroy();
    }

    void updateUniformBuffers(float timer) {
        ubo.lightPos.x = 0.0f + sin(glm::radians(timer * 360.0f)) * 2.0f;
        ubo.lightPos.y = 5.0f;
        ubo.lightPos.z = 1.0f;
        ubo.lightPos.z = 0.0f + cos(glm::radians(timer * 360.0f)) * 2.0f;
        uniformBuffer.copy(ubo);
    }

    void buildCommandBuffers() override {
        for (const auto& computeCmdBuffer : commandBuffers) {
            // Transfer the image from the graphics graphicsQueue to the compute graphicsQueue
            computeCmdBuffer.begin(vk::CommandBufferBeginInfo{});
            // Acquire from the graphics graphicsQueue
            computeCmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, 0, nullptr, 0, nullptr, 1, &acquireBarrier });
            // Execute compute pipeline
            computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);
            computeCmdBuffer.dispatch(renderTarget.image.createInfo.extent.width / 16, renderTarget.image.createInfo.extent.height / 16, 1);
            // Release to the graphics graphicsQueue
            computeCmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, 0, nullptr, 0, nullptr, 1, &releaseBarrier });
            computeCmdBuffer.end();
        }
    }
};

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

public:
    RaytracingCompute compute;
    struct {
        vks::model::Model quad;
    } meshes;

    struct {
        vk::Pipeline display;
        vk::Pipeline compute;
    } pipelines;

    int vertexBufferSize;
    vk::ImageMemoryBarrier2 acquireBarrier, releaseBarrier;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSetPostCompute;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.dolly(-2.0f);
        title = "Vulkan Example - Compute shader ray tracing";
        compute.ubo.aspectRatio = (float)size.width / (float)size.height;
        paused = true;
        timerSpeed *= 0.5f;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroy(pipelines.display);
        device.destroy(pipelines.compute);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        compute.destroy();

        meshes.quad.destroy();

        renderTarget.destroy();
    }

    // Prepare a texture target that is used to store compute shader calculations
    void prepareTextureTarget(RenderTarget& tex, uint32_t width, uint32_t height, vk::Format format) {
        // Get device properties for the requested texture format
        vk::FormatProperties formatProperties;
        formatProperties = physicalDevice.getFormatProperties(format);
        // Check if requested image format supports image storage operations
        assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);

        // Prepare blit target texture
        using namespace vks::util;

        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.format = format;
        imageCreateInfo.extent = vk::Extent3D{ width, height, 1 };
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.initialLayout = vk::ImageLayout::ePreinitialized;
        // vk::Image will be sampled in the fragment shader and used as storage target in the compute shader
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
        tex.image.create(imageCreateInfo);
        tex.view = tex.image.createView(vk::ImageViewType::e2D);

        // transition the image from the compute graphicsQueue
        acquireBarrier.oldLayout = vk::ImageLayout::eGeneral;
        acquireBarrier.newLayout = vk::ImageLayout::eGeneral;
        acquireBarrier.image = renderTarget.image.image;
        acquireBarrier.subresourceRange = renderTarget.image.getWholeRange();
        acquireBarrier.srcAccessMask = vk::AccessFlagBits2::eNone;
        acquireBarrier.srcStageMask = vk::PipelineStageFlagBits2::eNone;
        acquireBarrier.dstAccessMask = vk::AccessFlagBits2::eInputAttachmentRead;
        acquireBarrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        acquireBarrier.srcQueueFamilyIndex = computeQueue.familyInfo.index;
        acquireBarrier.dstQueueFamilyIndex = graphicsQueue.familyInfo.index;
        // transition the image from the compute graphicsQueue
        releaseBarrier.oldLayout = vk::ImageLayout::eGeneral;
        releaseBarrier.newLayout = vk::ImageLayout::eGeneral;
        releaseBarrier.image = renderTarget.image.image;
        releaseBarrier.subresourceRange = renderTarget.image.getWholeRange();
        releaseBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        releaseBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        releaseBarrier.dstAccessMask = vk::AccessFlagBits2::eNone;
        releaseBarrier.dstStageMask = vk::PipelineStageFlagBits2::eNone;
        releaseBarrier.srcQueueFamilyIndex = graphicsQueue.familyInfo.index;
        releaseBarrier.dstQueueFamilyIndex = computeQueue.familyInfo.index;
    }

    void updateCommandBufferPreDraw(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, nullptr, nullptr, acquireBarrier });
        Parent::updateCommandBufferPreDraw(cmdBuffer);
    }

    void updateCommandBufferPostDraw(const vk::CommandBuffer& cmdBuffer) override {
        Parent::updateCommandBufferPostDraw(cmdBuffer);
        // Send it back to the compute graphicsQueue
        cmdBuffer.pipelineBarrier2(vk::DependencyInfo{ {}, 0, nullptr, 0, nullptr, 1, &releaseBarrier });
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        // Display ray traced image generated by compute shader as a full screen quad
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSetPostCompute, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.display);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
    }

    // Setup vertices for a single uv-mapped quad
    void generateQuad() {
#define dim 1.0f
        std::vector<Vertex> vertexBuffer = { { { dim, dim, 0.0f }, { 1.0f, 1.0f } },
                                             { { -dim, dim, 0.0f }, { 0.0f, 1.0f } },
                                             { { -dim, -dim, 0.0f }, { 0.0f, 0.0f } },
                                             { { dim, -dim, 0.0f }, { 1.0f, 0.0f } } };
#undef dim

        meshes.quad.vertices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
    }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 2 },
            // Graphics pipeline uses image samplers for display
            { vk::DescriptorType::eCombinedImageSampler, 4 },
            // Compute pipeline uses storage images image loads and stores
            { vk::DescriptorType::eStorageImage, 1 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 3, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Fragment shader image sampler
            { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSetPostCompute = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // vk::Image descriptor for the color map texture
        vk::DescriptorImageInfo texDescriptor{ defaultSampler, renderTarget.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Fragment shader texture sampler
            { descriptorSetPostCompute, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        // Display pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout };
        pipelineCreator.depthStencilState = true;
        pipelineCreator.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        pipelineCreator.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        vertexLayout.appendVertexLayout(pipelineCreator.vertexInputState);
        using namespace vkx::shaders::raytracing;
        pipelineCreator.loadShader(texture::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(texture::frag, vk::ShaderStageFlagBits::eFragment);
        pipelines.display = pipelineCreator.create(context.pipelineCache);
    }

    // Prepare the compute pipeline that generates the ray traced image

    void prepare() {
        ExampleBase::prepare();
        prepareTextureTarget(renderTarget, TEX_DIM, TEX_DIM, vk::Format::eR8G8B8A8Unorm);
        compute.prepare(swapChain.imageCount);
        generateQuad();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        compute.buildCommandBuffers();
        prepared = true;
    }

    void postRender() {
        auto& cmdBuffer = compute.commandBuffers[currentIndex];
        using namespace vks::frame;
        QueuedCommandBuilder builder{ cmdBuffer, vkx::RenderStates::COMPUTE_POST, vk::PipelineStageFlagBits2::eComputeShader };
        builder.withQueueFamilyIndex(computeQueue.familyInfo.index);
        queueCommandBuffer(builder);
    }

    void update(float timeDelta) override {
        Parent::update(timeDelta);
        if (!paused) {
            compute.updateUniformBuffers(timer);
        }
    }

    virtual void viewChanged() { compute.updateUniformBuffers(timer); }
};

RUN_EXAMPLE(VulkanExample)
