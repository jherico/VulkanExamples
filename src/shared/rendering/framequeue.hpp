//
//  Copyright � 2023 Bradley Austin Davis
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#pragma once

#include <vulkan/vulkan.hpp>

namespace vks { namespace frame {

constexpr size_t INVALID_VECTOR_INDEX = static_cast<size_t>(-1);

struct QueuedCommand {
    vk::CommandBuffer cmdBuffer;
    uint64_t timelineValue;
    vk::PipelineStageFlags2 pipelineStages;
    uint32_t queueFamilyIndex{ VK_QUEUE_FAMILY_IGNORED };
    bool requiresSwapchainImage{ false };
    std::vector<vk::SemaphoreSubmitInfo> additionalWaits;
    std::vector<vk::SemaphoreSubmitInfo> additionalSignals;
};

struct QueuedCommandBuilder : public QueuedCommand {
    QueuedCommandBuilder(const vk::CommandBuffer& cmdBuffer, uint64_t signalValue, const vk::PipelineStageFlags2& stages) {
        this->cmdBuffer = cmdBuffer;
        this->timelineValue = signalValue;
        this->pipelineStages = stages;
    }

    QueuedCommandBuilder& withWaits(const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& waits) {
        additionalWaits.reserve(additionalWaits.size() + waits.size());
        for (const auto& wait : waits) {
            additionalWaits.emplace_back(wait);
        }
        return *this;
    }

    QueuedCommandBuilder& withSignals(const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& signals) {
        additionalSignals.reserve(additionalWaits.size() + signals.size());
        for (const auto& signal : signals) {
            additionalSignals.emplace_back(signal);
        }
        return *this;
    }

    QueuedCommandBuilder& withSwapchainImageRequired(bool required = true) {
        this->requiresSwapchainImage = required;
        return *this;
    }
    QueuedCommandBuilder& withQueueFamilyIndex(uint32_t queueFamilyIndex) {
        this->queueFamilyIndex = queueFamilyIndex;
        return *this;
    }
};

// A sequence of command buffers intended to be executed in order with a timeline semaphore
struct QueuedCommands {
    struct QueuedCommand : public vks::frame::QueuedCommand {
        using Parent = vks::frame::QueuedCommand;
        using Builder = vks::frame::QueuedCommandBuilder;
        QueuedCommand() = default;
        QueuedCommand(const Builder& builder) { static_cast<Parent&>(*this) = builder; }
    };

    std::vector<QueuedCommand> queuedCommands;
    size_t lastSwapchainAccess{ INVALID_VECTOR_INDEX };

    void queueCommandBuffer(const vk::CommandBuffer& commandBuffer, uint64_t timelineValue, const vk::PipelineStageFlags2& flags);
    void queueCommandBuffer(const QueuedCommand::Builder& queuedCommand);

    bool valid() const;

    void reset();
};

}}  // namespace vks::frame
