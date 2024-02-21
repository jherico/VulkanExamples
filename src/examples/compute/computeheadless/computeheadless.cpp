/*
 * Vulkan Example - Minimal headless compute example
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

// TODO: separate transfer graphicsQueue (if not supported by compute graphicsQueue) including buffer ownership transfer

#include <common/common.hpp>
#include <common/utils.hpp>
#include <rendering/context.hpp>
#include <rendering/loader.hpp>
#include <rendering/shaders.hpp>
#include <vks/buffer.hpp>

#include <shaders/computeheadless/headless.comp.inl>

#define BUFFER_ELEMENTS 32

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#define LOG(...) ((void)__android_log_print(ANDROID_LOG_INFO, "vulkanExample", __VA_ARGS__))
#else
#define LOG(...) printf(__VA_ARGS__)
#endif

class VulkanExample {
public:
    vks::Loader& loader{ vks::Loader::get() };
    vks::Context& context{ vks::Context::get() };
    vks::DeviceInfo& deviceInfo{ context.deviceInfo };
    vks::QueuesInfo& queuesInfo{ context.queuesInfo };
    vk::Device& device{ context.device };
    vks::QueueManager queueManager;
    vk::CommandBuffer commandBuffer;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    vk::ShaderModule shaderModule;
    /*
     * Prepare storage buffers
     */
    std::vector<uint32_t> computeInput;
    std::vector<uint32_t> computeOutput;
    vks::Buffer deviceBuffer, hostBuffer;
    size_t bufferSize = sizeof(uint32_t) * BUFFER_ELEMENTS;

    VulkanExample() {}

    void prepare() {
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
        LOG("loading vulkan lib");
        vks::android::loadVulkanLibrary();
#endif
        context.createInstance();
        context.pickDevice();
        context.createDevice();
        LOG("GPU: %s\n", deviceInfo.properties.core10.deviceName.data());

        queueManager = vks::QueueManager{ device, context.queuesInfo.compute };
        setupBuffers();
        setupDescriptors();
        setupPipeline();
        setupCommandBuffer();
    }

    void setupBuffers() {
        computeInput.resize(BUFFER_ELEMENTS);
        computeOutput.resize(BUFFER_ELEMENTS);
        // Fill input data
        uint32_t n = 0;
        std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });
        // Copy input data to VRAM using a staging buffer
        deviceBuffer =
            loader.stageToDeviceBuffer<uint32_t>(queueManager, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, computeInput);

        hostBuffer = vks::Buffer::Builder{ bufferSize }
                         .withBufferUsage(vk::BufferUsageFlagBits::eTransferDst)
                         .withAllocCreateFlags(VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
                         .build();
    }

    void setupDescriptors() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1 },
        };

        descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{ {}, 1, static_cast<uint32_t>(poolSizes.size()), poolSizes.data() });

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };
        descriptorSetLayout = device.createDescriptorSetLayout({ {}, 1, setLayoutBindings.data() });

        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });

        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        vk::DescriptorBufferInfo bufferDescriptor{ deviceBuffer.buffer, 0, VK_WHOLE_SIZE };
        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets = {
            vk::WriteDescriptorSet{ descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferDescriptor },
        };
        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);
    }

    void setupPipeline() {
        // Create pipeline
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = pipelineLayout;
        computePipelineCreateInfo.stage =
            vks::shaders::loadShader(context.device, vkx::shaders::computeheadless::headless::comp, vk::ShaderStageFlagBits::eCompute);

        // Pass SSBO size via specialization constant
        struct SpecializationData {
            uint32_t BUFFER_ELEMENT_COUNT = BUFFER_ELEMENTS;
        } specializationData;
        vk::SpecializationMapEntry specializationMapEntry{ 0, 0, sizeof(uint32_t) };
        vk::SpecializationInfo specializationInfo{ 1, &specializationMapEntry, sizeof(SpecializationData), &specializationData };
        computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;
        pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroy(computePipelineCreateInfo.stage.module);
    }

    void setupCommandBuffer() {
        if (commandBuffer) {
            queueManager.freeCommandBuffer(commandBuffer);
        }
        // Create a command buffer for compute operations
        commandBuffer = queueManager.createCommandBuffer();
        commandBuffer.begin(vk::CommandBufferBeginInfo{});
        // Barrier to ensure that input buffer transfer is finished before compute shader reads from it
        vk::BufferMemoryBarrier2 bufferBarrier;
        vk::DependencyInfo dependencyInfo{ {}, nullptr, bufferBarrier, nullptr };
        bufferBarrier.buffer = deviceBuffer.buffer;
        bufferBarrier.size = VK_WHOLE_SIZE;
        bufferBarrier.srcAccessMask = vk::AccessFlagBits2::eHostWrite;
        bufferBarrier.srcStageMask = vk::PipelineStageFlagBits2::eHost;
        bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bufferBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        bufferBarrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        commandBuffer.pipelineBarrier2(dependencyInfo);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);
        commandBuffer.dispatch(BUFFER_ELEMENTS, 1, 1);

        // Barrier to ensure that shader writes are finished before buffer is read back from GPU
        bufferBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        bufferBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        bufferBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        bufferBarrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
        commandBuffer.pipelineBarrier2(dependencyInfo);

        // Read back to host visible buffer
        vk::BufferCopy copyRegion{ 0, 0, bufferSize };
        commandBuffer.copyBuffer(deviceBuffer.buffer, hostBuffer.buffer, copyRegion);

        // Barrier to ensure that buffer copy is finished before host reading from it
        bufferBarrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        bufferBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        bufferBarrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;
        bufferBarrier.dstStageMask = vk::PipelineStageFlagBits2::eHost;
        bufferBarrier.buffer = hostBuffer.buffer;
        commandBuffer.pipelineBarrier2(dependencyInfo);
        commandBuffer.end();
    }

    void run() {
        LOG("Running headless compute example\n");
        prepare();

        queueManager.submitAndWait(commandBuffer);
        // Make device writes visible to the host
        hostBuffer.copyOut(bufferSize, computeOutput.data(), 0);

        // Output buffer contents
        LOG("Compute input:\n");
        for (auto v : computeInput) {
            LOG("%d \t", v);
        }
        std::cout << std::endl;

        LOG("Compute output:\n");
        for (auto v : computeOutput) {
            LOG("%d \t", v);
        }
        std::cout << std::endl;
    }

    ~VulkanExample() {
        deviceBuffer.destroy();
        hostBuffer.destroy();
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(descriptorPool);
        device.destroy(pipeline);
        queueManager.destroy();
        context.destroy();
        std::cout << "Finished. Press enter to terminate...";
        getchar();
    }
};

VULKAN_EXAMPLE_MAIN()
