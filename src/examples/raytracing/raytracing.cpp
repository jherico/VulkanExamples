/*
* Vulkan Example - Compute shader ray tracing
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>

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

class VulkanExample : public vkx::ExampleBase {
private:
    vks::Image textureComputeTarget;

public:
    struct {
        vks::model::Model quad;
    } meshes;

    vks::Buffer uniformDataCompute;

    struct UboCompute {
        glm::vec3 lightPos;
        // Aspect ratio of the viewport
        float aspectRatio;
        glm::vec4 fogColor = glm::vec4(0.0f);
        struct Camera {
            glm::vec3 pos = glm::vec3(0.0f, 1.5f, 4.0f);
            glm::vec3 lookat = glm::vec3(0.0f, 0.5f, 0.0f);
            float fov = 10.0f;
        } camera;
    } uboCompute;

    struct {
        vk::Pipeline display;
        vk::Pipeline compute;
    } pipelines;

    int vertexBufferSize;

    vk::Queue computeQueue;
    vk::CommandBuffer computeCmdBuffer;
    vk::PipelineLayout computePipelineLayout;
    vk::DescriptorSet computeDescriptorSet;
    vk::DescriptorSetLayout computeDescriptorSetLayout;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSetPostCompute;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        camera.dolly(-2.0f);
        title = "Vulkan Example - Compute shader ray tracing";
        uboCompute.aspectRatio = (float)size.width / (float)size.height;
        paused = true;
        timerSpeed *= 0.5f;
    }

    ~VulkanExample() {
        // Clean up used Vulkan resources
        // Note : Inherited destructor cleans up resources stored in base class

        device.destroyPipeline(pipelines.display);
        device.destroyPipeline(pipelines.compute);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyPipelineLayout(computePipelineLayout);
        device.destroyDescriptorSetLayout(computeDescriptorSetLayout);

        meshes.quad.destroy();
        uniformDataCompute.destroy();

        device.freeCommandBuffers(cmdPool, computeCmdBuffer);

        textureComputeTarget.destroy();
    }

    // Prepare a texture target that is used to store compute shader calculations
    void prepareTextureTarget(vks::Image& tex, uint32_t width, uint32_t height, vk::Format format) {
        context.withPrimaryCommandBuffer([&](const vk::CommandBuffer& setupCmdBuffer) {
            // Get device properties for the requested texture format
            vk::FormatProperties formatProperties;
            formatProperties = physicalDevice.getFormatProperties(format);
            // Check if requested image format supports image storage operations
            assert(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);

            // Prepare blit target texture
            tex.extent.width = width;
            tex.extent.height = height;

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
            tex = context.createImage(imageCreateInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
            context.setImageLayout(setupCmdBuffer, tex.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::ePreinitialized, vk::ImageLayout::eGeneral);

            // Create sampler
            vk::SamplerCreateInfo sampler;
            sampler.magFilter = vk::Filter::eLinear;
            sampler.minFilter = vk::Filter::eLinear;
            sampler.mipmapMode = vk::SamplerMipmapMode::eLinear;
            sampler.addressModeU = vk::SamplerAddressMode::eRepeat;
            sampler.addressModeV = sampler.addressModeU;
            sampler.addressModeW = sampler.addressModeU;
            sampler.mipLodBias = 0.0f;
            sampler.maxAnisotropy = 0;
            sampler.compareOp = vk::CompareOp::eNever;
            sampler.minLod = 0.0f;
            sampler.maxLod = 0.0f;
            sampler.borderColor = vk::BorderColor::eFloatOpaqueWhite;
            tex.sampler = device.createSampler(sampler);

            // Create image view
            vk::ImageViewCreateInfo view;
            view.viewType = vk::ImageViewType::e2D;
            view.format = format;
            view.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
            view.image = tex.image;
            tex.view = device.createImageView(view);
        });
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) {
        // vk::Image memory barrier to make sure that compute
        // shader writes are finished before sampling
        // from the texture
        vk::ImageMemoryBarrier imageMemoryBarrier;
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eGeneral;
        imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
        imageMemoryBarrier.image = textureComputeTarget.image;
        imageMemoryBarrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eInputAttachmentRead;
        cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, vk::DependencyFlags(), nullptr, nullptr,
                                  imageMemoryBarrier);
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindVertexBuffers(0, meshes.quad.vertices.buffer, { 0 });
        cmdBuffer.bindIndexBuffer(meshes.quad.indices.buffer, 0, vk::IndexType::eUint32);
        // Display ray traced image generated by compute shader as a full screen quad
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSetPostCompute, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.display);
        cmdBuffer.drawIndexed(meshes.quad.indexCount, 1, 0, 0, 0);
    }

    void buildComputeCommandBuffer() {
        vk::CommandBufferBeginInfo cmdBufInfo;
        computeCmdBuffer.begin(cmdBufInfo);
        computeCmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines.compute);
        computeCmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, computeDescriptorSet, nullptr);
        computeCmdBuffer.dispatch(textureComputeTarget.extent.width / 16, textureComputeTarget.extent.height / 16, 1);
        computeCmdBuffer.end();
    }

    void compute() {
        // Compute
        vk::SubmitInfo computeSubmitInfo;
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &computeCmdBuffer;
        computeQueue.submit(computeSubmitInfo, nullptr);
        computeQueue.waitIdle();
    }

    // Setup vertices for a single uv-mapped quad
    void generateQuad() {
#define dim 1.0f
        std::vector<Vertex> vertexBuffer = { { { dim, dim, 0.0f }, { 1.0f, 1.0f } },
                                             { { -dim, dim, 0.0f }, { 0.0f, 1.0f } },
                                             { { -dim, -dim, 0.0f }, { 0.0f, 0.0f } },
                                             { { dim, -dim, 0.0f }, { 1.0f, 0.0f } } };
#undef dim

        meshes.quad.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexBuffer);
        std::vector<uint32_t> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        meshes.quad.indexCount = (uint32_t)indexBuffer.size();
        meshes.quad.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, indexBuffer);
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
        vk::DescriptorImageInfo texDescriptor{ textureComputeTarget.sampler, textureComputeTarget.view, vk::ImageLayout::eGeneral };

        std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
            // Binding 0 : Fragment shader texture sampler
            { descriptorSetPostCompute, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texDescriptor },
        };

        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    // Create a separate command buffer for compute commands
    void createComputeCommandBuffer() { computeCmdBuffer = device.allocateCommandBuffers({ cmdPool, vk::CommandBufferLevel::ePrimary, 1 })[0]; }

    void preparePipelines() {
        // Display pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineCreator{ device, pipelineLayout, renderPass };
        pipelineCreator.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        pipelineCreator.vertexInputState.appendVertexLayout(vertexLayout);
        pipelineCreator.loadShader(getAssetPath() + "shaders/raytracing/texture.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineCreator.loadShader(getAssetPath() + "shaders/raytracing/texture.frag.spv", vk::ShaderStageFlagBits::eFragment);
        pipelines.display = pipelineCreator.create(context.pipelineCache);
    }

    // Prepare the compute pipeline that generates the ray traced image
    void prepareCompute() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0 : Sampled image (write)
            { 0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 1 : Uniform buffer block
            { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };

        computeDescriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        computePipelineLayout = device.createPipelineLayout({ {}, 1, &computeDescriptorSetLayout });

        computeDescriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &computeDescriptorSetLayout })[0];

        std::vector<vk::DescriptorImageInfo> computeTexDescriptors{
            { nullptr, textureComputeTarget.view, vk::ImageLayout::eGeneral },
        };
        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            // Binding 0 : Output storage image
            { computeDescriptorSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &computeTexDescriptors[0] },
            // Binding 1 : Uniform buffer block
            { computeDescriptorSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformDataCompute.descriptor },
        };
        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);

        // Create compute shader pipelines
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = computePipelineLayout;
        computePipelineCreateInfo.stage =
            vks::shaders::loadShader(device, getAssetPath() + "shaders/raytracing/raytracing.comp.spv", vk::ShaderStageFlagBits::eCompute);
        pipelines.compute = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
    }

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers() {
        // Vertex shader uniform buffer block
        uniformDataCompute = context.createUniformBuffer(uboCompute);
        updateUniformBuffers();
    }

    void updateUniformBuffers() {
        uboCompute.lightPos.x = 0.0f + sin(glm::radians(timer * 360.0f)) * 2.0f;
        uboCompute.lightPos.y = 5.0f;
        uboCompute.lightPos.z = 1.0f;
        uboCompute.lightPos.z = 0.0f + cos(glm::radians(timer * 360.0f)) * 2.0f;
        uniformDataCompute.copy(uboCompute);
    }

    // Find and create a compute capable device queue
    void getComputeQueue() {
        uint32_t queueIndex = 0;

        std::vector<vk::QueueFamilyProperties> queueProps = physicalDevice.getQueueFamilyProperties();
        uint32_t queueCount = (uint32_t)queueProps.size();

        for (queueIndex = 0; queueIndex < queueCount; queueIndex++) {
            if (queueProps[queueIndex].queueFlags & vk::QueueFlagBits::eCompute)
                break;
        }
        assert(queueIndex < queueCount);

        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.queueFamilyIndex = queueIndex;
        queueCreateInfo.queueCount = 1;
        computeQueue = device.getQueue(queueIndex, 0);
    }

    void prepare() {
        ExampleBase::prepare();
        generateQuad();
        getComputeQueue();
        createComputeCommandBuffer();
        prepareUniformBuffers();
        prepareTextureTarget(textureComputeTarget, TEX_DIM, TEX_DIM, vk::Format::eR8G8B8A8Unorm);
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        prepareCompute();
        buildCommandBuffers();
        buildComputeCommandBuffer();
        prepared = true;
    }

    virtual void render() {
        if (!prepared)
            return;
        draw();
        compute();
        if (!paused) {
            updateUniformBuffers();
        }
    }

    virtual void viewChanged() { updateUniformBuffers(); }
};

RUN_EXAMPLE(VulkanExample)
