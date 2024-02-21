//
//  Created by Bradley Austin Davis
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#pragma once

#include <examples/example.hpp>
#include <rendering/offscreen.hpp>

namespace vkx {

struct OffscreenRenderer : public vkx::offscreen::Renderer {
private:
    using Parent = vkx::offscreen::Renderer;

public:
    vks::QueueManager queue;
    vk::CommandBuffer cmdBuffer;
    bool active{ true };

    void prepare(const vkx::offscreen::Builder& builder) override {
        vkx::offscreen::Renderer::prepare(builder);
        queue = vks::QueueManager{ device, context.queuesInfo.graphics };
        cmdBuffer = queue.createCommandBuffer();
    }

    void destroy() override {
        if (cmdBuffer) {
            queue.freeCommandBuffer(cmdBuffer);
            cmdBuffer = nullptr;
        }
        queue.destroy();
        Parent::destroy();
    }

    void waitIdle() const {
        queue.handle.waitIdle();
        device.waitIdle();
    }
};

class OffscreenExampleBase : public ExampleBase {
protected:
    virtual ~OffscreenExampleBase() { 
        offscreen.destroy(); 
    }

    virtual void buildOffscreenCommandBuffer() = 0;
    virtual void prepareOffscreen() = 0;

    void preRender() override {
        if (offscreen.active) {
            queueCommandBuffer(offscreen.cmdBuffer, RenderStates::OFFSCREEN_PRERENDER, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
        }
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareOffscreen();
    }

    OffscreenRenderer offscreen;
};

}  // namespace vkx
