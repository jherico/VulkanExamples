#include <rendering/context.hpp>

namespace vkx {

struct Compute {
    const vks::Context& context{ vks::Context::get() };
    const vk::Device& device{ context.device };
    vks::QueueManager computeQueue;
    std::vector<vk::CommandBuffer> commandBuffers;

    virtual void buildCommandBuffers() = 0;

    virtual void prepare(uint32_t swapchainImageCount) {
        computeQueue = vks::QueueManager(device, context.queuesInfo.compute);
        commandBuffers = computeQueue.allocateCommandBuffers(swapchainImageCount);
    }

    virtual void destroy() {
        computeQueue.freeCommandBuffers(commandBuffers);
        computeQueue.destroy();
    }
};

}  // namespace vkx
