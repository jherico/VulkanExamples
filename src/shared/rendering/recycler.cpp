//
//  Copyright � 2023 Bradley Austin Davis
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#include "recycler.hpp"
#include "context.hpp"

namespace vks {

Recycler& Recycler::get() {
    static Recycler instance;
    return instance;
}

void Recycler::trashCommandBuffers(const vk::CommandPool& commandPool, vk::CommandBuffer& cmdBuffer) const {
    TypedItem<vk::CommandBuffer>::Lambda destroyer = [=](const vk::CommandBuffer& cmdBuffer) {
        const auto& device = vks::Context::get().device;
        device.free(commandPool, cmdBuffer);
    };
    trash(std::move(cmdBuffer), std::move(destroyer));
    cmdBuffer = nullptr;
}

void Recycler::trashCommandBuffers(const vk::CommandPool& commandPool, std::vector<vk::CommandBuffer>& cmdBuffers) const {
    std::vector<vk::CommandBuffer> trashedBuffers;
    trashedBuffers.swap(cmdBuffers);
    TypedItemVector<vk::CommandBuffer>::Lambda destroyer = [=](const std::vector<vk::CommandBuffer>& cmdBuffers) {
        const auto& device = vks::Context::get().device;
        device.free(commandPool, cmdBuffers);
    };
    trashAll(std::move(trashedBuffers), std::move(destroyer));
}

void Recycler::emptyDumpster(vk::Fence fence) {
    // FIXME we should check the size of the dumpster and keep track of the max size, so we can reserve that amount in newDumpster and minimize allocations
    ItemList newDumpster;
    newDumpster.swap(dumpster);
    TypedItem<ItemList>::Lambda destroyer = [](const ItemList& items) {
        for (const auto& item : items) {
            item->destroy();
        }
    };

    Item::Ptr item = std::make_unique<TypedItem<ItemList>>(std::move(newDumpster), destroyer);
    recycler.push(FencedItem{ fence, std::move(item) });
}

void Recycler::recycle() {
    const auto& device = vks::Context::get().device;
    while (!recycler.empty() && vk::Result::eSuccess == device.getFenceStatus(recycler.front().first)) {
        const auto fence = recycler.front().first;
        const auto destroyer = std::move(recycler.front().second);
        recycler.pop();
        destroyer->destroy();
        if (recycler.empty() || fence != recycler.front().first) {
            device.destroy(fence);
        }
    }
}

void Recycler::flush() {
    for (const auto& trash : dumpster) {
        trash->destroy();
    }

    const auto& device = vks::Context::get().device;
    while (!recycler.empty()) {
        device.waitIdle();
        recycle();
    }
}

}  // namespace vks
