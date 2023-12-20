#include "vks/context.hpp"

namespace vkx {

// Resources for the compute part of the example
struct Compute {
    Compute(const vks::Context& context)
        : context(context) {}

    const vks::Context& context;
    const vk::Device& device{ context.device };
    vk::Queue queue;
    vk::CommandPool commandPool;

    struct Semaphores {
        vk::Semaphore ready;
        vk::Semaphore complete;
    } semaphores;

    virtual void prepare() {
        // Create a compute capable device queue
        queue = context.device.getQueue(context.queueIndices.compute, 0);
        semaphores.ready = device.createSemaphore({});
        semaphores.complete = device.createSemaphore({});
        // Separate command pool as queue family for compute may be different than graphics
        commandPool = device.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, context.queueIndices.compute });
    }

    virtual void destroy() {
        context.device.destroy(semaphores.complete);
        context.device.destroy(semaphores.ready);
        context.device.destroy(commandPool);
    }

    void submit(const vk::ArrayProxy<const vk::CommandBuffer>& commandBuffers) {
        static const std::vector<vk::PipelineStageFlags> waitStages{ vk::PipelineStageFlagBits::eComputeShader };
        // Submit compute commands
        vk::SubmitInfo computeSubmitInfo;
        computeSubmitInfo.commandBufferCount = commandBuffers.size();
        computeSubmitInfo.pCommandBuffers = commandBuffers.data();
        computeSubmitInfo.waitSemaphoreCount = 1;
        computeSubmitInfo.pWaitSemaphores = &semaphores.ready;
        computeSubmitInfo.pWaitDstStageMask = waitStages.data();
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &semaphores.complete;
        queue.submit(computeSubmitInfo, {});
    }
};

}  // namespace vkx
