//
//  Copyright � 2023 Bradley Austin Davis
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#include "framequeue.hpp"

namespace vks { namespace frame {

void QueuedCommands::queueCommandBuffer(const vk::CommandBuffer& commandBuffer, uint64_t timelineValue, const vk::PipelineStageFlags2& flags) {
    queueCommandBuffer(QueuedCommand::Builder{ commandBuffer, timelineValue, flags });
}

void QueuedCommands::queueCommandBuffer(const QueuedCommand::Builder& queuedCommand) {
    if (queuedCommand.requiresSwapchainImage) {
        lastSwapchainAccess = queuedCommands.size();
    }
    queuedCommands.emplace_back(queuedCommand);
}

bool QueuedCommands::valid() const {
    if (lastSwapchainAccess == INVALID_VECTOR_INDEX) {
        return false;
    }

    uint64_t lastTimelineValue = 0;
    for (const auto& command : queuedCommands) {
        if (command.timelineValue <= lastTimelineValue) {
            return false;
        }
        lastTimelineValue = command.timelineValue;
    }
    return true;
}

void QueuedCommands::reset() {
    lastSwapchainAccess = INVALID_VECTOR_INDEX;
    queuedCommands.clear();
}

}}  // namespace vks::frame

#if 0
    struct SubmitInfo {
    vk::CommandBuffer cmdBuffer;
    uint64_t timelineValue;
    vk::PipelineStageFlags2 pipelineStages;
    bool requiresSwapchainImage;
    std::vector<vk::SemaphoreSubmitInfo> additionalWaits;
    std::vector<vk::SemaphoreSubmitInfo> additionalSignals;

    SubmitInfo() = default;
    SubmitInfo(const SubmissionBuilder& builder);
    SubmitInfo(const vk::CommandBuffer& cmdBuffer,
               uint64_t signalValue,
               const vk::PipelineStageFlags2& stages,
               bool requiresSwapchainImage,
               const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalWaitSemaphores,
               const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalSignalSemaphores);
};

struct SubmissionBuilder {
    vk::CommandBuffer cmdBuffer;
    uint64_t timelineValue;
    vk::PipelineStageFlags2 pipelineStages;
    bool requiresSwapchainImage{ false };
    std::vector<vk::SemaphoreSubmitInfo> additionalWaits;
    std::vector<vk::SemaphoreSubmitInfo> additionalSignals;

    SubmissionBuilder(const vk::CommandBuffer& cmdBuffer, uint64_t signalValue, const vk::PipelineStageFlags2& stages)
        : cmdBuffer(cmdBuffer)
        , timelineValue(signalValue)
        , pipelineStages(stages) {}

    SubmissionBuilder& withWaits(const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& waits) {
        additionalWaits.reserve(additionalWaits.size() + waits.size());
        for (const auto& wait : waits) {
            additionalWaits.emplace_back(wait);
        }
    }

    SubmissionBuilder& withSignals(const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& signals) {
        additionalSignals.reserve(additionalWaits.size() + signals.size());
        for (const auto& signal : signals) {
            additionalSignals.emplace_back(signal);
        }
    }
};
ExampleBase::SubmitInfo::SubmitInfo(const vk::CommandBuffer& cmdBuffer,
                                    uint64_t timelineValue,
                                    const vk::PipelineStageFlags2& pipelineStages,
                                    bool requiresSwapchainImage,
                                    const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalWaitSemaphores,
                                    const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalSignalSemaphores)
    : cmdBuffer(cmdBuffer)
    , timelineValue(timelineValue)
    , pipelineStages(pipelineStages)
    , requiresSwapchainImage(requiresSwapchainImage) {
    if (!additionalWaitSemaphores.empty()) {
        this->additionalWaits.insert(this->additionalWaits.begin(), additionalWaitSemaphores.begin(), additionalWaitSemaphores.end());
    }
    if (!additionalSignalSemaphores.empty()) {
        this->additionalSignals.insert(this->additionalSignals.begin(), additionalSignalSemaphores.begin(), additionalSignalSemaphores.end());
    }
}


// A sequence of command buffers intended to be executed in order with a timeline semaphore
struct FrameQueue {
    struct QueuedCommandInfo {
        vk::CommandBuffer cmdBuffer;
        uint64_t timelineValue;
        vk::PipelineStageFlags2 pipelineStages;
        uint32_t queueFamilyIndex{ VK_QUEUE_FAMILY_IGNORED };
        bool requiresSwapchainImage{ false };
        std::vector<vk::SemaphoreSubmitInfo> additionalWaits;
        std::vector<vk::SemaphoreSubmitInfo> additionalSignals;
    };

    struct QueuedCommand : public QueuedCommandInfo {
        struct Builder : public QueuedCommandInfo {
            Builder(const vk::CommandBuffer& cmdBuffer, uint64_t signalValue, const vk::PipelineStageFlags2& stages) {
                this->cmdBuffer = cmdBuffer;
                this->timelineValue = signalValue;
                this->pipelineStages = stages;
            }

            Builder& withWaits(const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& waits) {
                additionalWaits.reserve(additionalWaits.size() + waits.size());
                for (const auto& wait : waits) {
                    additionalWaits.emplace_back(wait);
                }
            }

            QueuedCommandBuilder& withSignals(const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& signals) {
                additionalSignals.reserve(additionalWaits.size() + signals.size());
                for (const auto& signal : signals) {
                    additionalSignals.emplace_back(signal);
                }
            }
        };


        vk::CommandBuffer cmdBuffer;
        uint64_t timelineValue;
        vk::PipelineStageFlags2 pipelineStages;
        bool requiresSwapchainImage;
        std::vector<vk::SemaphoreSubmitInfo> additionalWaits;
        std::vector<vk::SemaphoreSubmitInfo> additionalSignals;

        SubmitInfo() = default;
        SubmitInfo(const SubmissionBuilder& builder);
        SubmitInfo(const vk::CommandBuffer& cmdBuffer,
                   uint64_t signalValue,
                   const vk::PipelineStageFlags2& stages,
                   bool requiresSwapchainImage,
                   const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalWaitSemaphores,
                   const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalSignalSemaphores);
    };

    std::vector<QueuedCommand> commandBuffers;
};
#endif
