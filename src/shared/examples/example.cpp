/*
 * Vulkan Example base class
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */
#include "example.hpp"

#include <common/filesystem.hpp>
#include <common/storage.hpp>

#ifdef ENABLE_UI
#include <imgui.h>
#endif
using namespace vkx;

vk::Extent2D ExampleBase::EMPTY_RECT;

void PerFrameData::reset() {
    static const auto& context = vks::Context::get();
    wait();
    vk::SemaphoreTypeCreateInfo semaphoreTypeCreateInfo{ vk::SemaphoreType::eTimeline, 0 };
    semaphore = context.device.createSemaphore({ {}, &semaphoreTypeCreateInfo });
    finalState = 0;
}

void PerFrameData::destroy() {
    static const auto& context = vks::Context::get();
    if (semaphore) {
        context.device.destroy(semaphore);
        semaphore = nullptr;
    }
    finalState = 0;
}

void PerFrameData::wait() {
    static const auto& context = vks::Context::get();
    if (!semaphore || finalState == 0) {
        return;
    }
    auto result = context.device.waitSemaphores(vk::SemaphoreWaitInfo{ {}, semaphore, finalState }, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for semaphore final state");
    }
    destroy();
}

// Avoid doing work in the ctor as it can't make use of overridden virtual functions
// Instead, use the `prepare` and `run` methods
ExampleBase::ExampleBase() {
#if defined(__ANDROID__)
    vks::storage::setAssetManager(vkx::android::androidApp->activity->assetManager);
    vkx::android::androidApp->userData = this;
    vkx::android::androidApp->onInputEvent = ExampleBase::handle_input_event;
    vkx::android::androidApp->onAppCmd = ExampleBase::handle_app_cmd;
#endif
    camera.setPerspective(60.0f, size, 0.1f, 256.0f);
}

ExampleBase::~ExampleBase() {
    // Wait till there's no work to be done
    waitIdle();

    // Clean up Vulkan resources
    swapChain.destroy();
    if (defaultSampler) {
        device.destroy(defaultSampler);
        defaultSampler = nullptr;
    }

    // FIXME destroy surface
    if (descriptorPool) {
        device.destroy(descriptorPool);
        descriptorPool = nullptr;
    }
    clearFrameData();

    device.destroy(depthStencilView);
    depthStencilView = nullptr;
    depthStencil.destroy();

    if (semaphores.swapchainAcquire.semaphore) {
        device.destroy(semaphores.swapchainAcquire.semaphore);
        semaphores.swapchainAcquire.semaphore = nullptr;
    }

    if (semaphores.swapchainFilled.semaphore) {
        device.destroy(semaphores.swapchainFilled.semaphore);
        semaphores.swapchainFilled.semaphore = nullptr;
    }
#if ENABLE_UI
    ui.destroy();
#endif

    if (graphicsQueue) {
        graphicsQueue.destroy(true);
    }
    if (computeQueue) {
        computeQueue.destroy(true);
    }
    if (transferQueue) {
        transferQueue.destroy(true);
    }
    device.waitIdle();
    device.destroy(pipelineCache);
    vks::Allocation::shutdown();

    instance.destroy(surface);
    surface = nullptr;
    context.destroy();
    window.destroyWindow();
    glfw::Window::terminate();
}

void ExampleBase::queueCommandBuffer(const vk::CommandBuffer& cmdBuffer,
                                     uint64_t signalValue,
                                     const vk::PipelineStageFlags2& stages,
                                     bool requiresSwapchainImage,
                                     const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalWaitSemaphores,
                                     const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalSignalSemaphores) {
    vks::frame::QueuedCommandBuilder builder{ cmdBuffer, signalValue, stages };
    if (!additionalWaitSemaphores.empty()) {
        builder.withWaits(additionalWaitSemaphores);
    }
    if (!additionalSignalSemaphores.empty()) {
        builder.withSignals(additionalSignalSemaphores);
    }
    builder.withSwapchainImageRequired(requiresSwapchainImage);
    queueCommandBuffer(builder);
}

void ExampleBase::queueCommandBuffer(const vks::frame::QueuedCommandBuilder& builder) {
    currentFrameQueue.queueCommandBuffer(builder);
}

void ExampleBase::run() {
    try {
        glfw::Window::init();
        setupWindow();
        initVulkan();
        prepare();
        // Lock the loader
        vks::Loader::get().lock();
        renderLoop();
        waitIdle();
    } catch (const std::system_error& err) {
        std::cerr << err.what() << std::endl;
    }
}

void ExampleBase::getEnabledFeatures() {
    auto& enabledFeatures = context.enabledFeatures;
    assert(context.deviceInfo.features.core12.timelineSemaphore == VK_TRUE);
    enabledFeatures.core12.timelineSemaphore = VK_TRUE;
    assert(context.deviceInfo.features.core13.dynamicRendering == VK_TRUE);
    enabledFeatures.core13.dynamicRendering = VK_TRUE;
    assert(context.deviceInfo.features.core13.maintenance4 == VK_TRUE);
    enabledFeatures.core13.maintenance4 = VK_TRUE;
    assert(context.deviceInfo.features.core13.synchronization2 == VK_TRUE);
    enabledFeatures.core13.synchronization2 = VK_TRUE;
    enabledFeatures.core13.pipelineCreationCacheControl = VK_TRUE;
    enabledFeatures.descriptorBufferEXT.descriptorBuffer = VK_TRUE;

    enabledFeatures.core10.samplerAnisotropy = context.deviceInfo.features.core10.samplerAnisotropy;
    enabledFeatures.core10.textureCompressionBC = deviceInfo.features.core10.textureCompressionBC;
    enabledFeatures.core10.textureCompressionASTC_LDR = deviceInfo.features.core10.textureCompressionASTC_LDR;
    enabledFeatures.core10.textureCompressionETC2 = deviceInfo.features.core10.textureCompressionETC2;
    enabledFeatures.core10.samplerAnisotropy = deviceInfo.features.core10.samplerAnisotropy;
}

void ExampleBase::waitIdle() {
    if (graphicsQueue) {
        graphicsQueue.handle.waitIdle();
    }
    if (computeQueue) {
        computeQueue.handle.waitIdle();
    }
    if (transferQueue) {
        graphicsQueue.handle.waitIdle();
    }
    device.waitIdle();
}

void ExampleBase::initVulkan() {
    context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
    context.requireDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    context.createInstance(version);
    surface = window.createSurface(context.instance);
    context.pickDevice(surface);

    getEnabledFeatures();
    context.createDevice();
    pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());
    if (!context.queuesInfo.graphics) {
        throw new std::runtime_error("Unable to find required graphics queue");
    }

    // Build the queues
    graphicsQueue = vks::QueueManager{ device, context.queuesInfo.graphics };
    if (context.queuesInfo.compute) {
        computeQueue = vks::QueueManager{ device, context.queuesInfo.compute };
    }
    if (context.queuesInfo.transfer) {
        transferQueue = vks::QueueManager{ device, context.queuesInfo.transfer };
    }

    // Create synchronization objects, we need binary semaphores to determine
    // when the swapchain image is available for rendering and when it has completed
    semaphores.swapchainAcquire.semaphore = device.createSemaphore({});
    semaphores.swapchainFilled.semaphore = device.createSemaphore({});
}

bool ExampleBase::platformLoopCondition() {
    if (window.shouldClose()) {
        return false;
    }
    window.pollEvents();
    return true;
}

void ExampleBase::renderLoop() {
    auto tStart = std::chrono::high_resolution_clock::now();

    while (platformLoopCondition()) {
        auto tEnd = std::chrono::high_resolution_clock::now();
        auto tDiff = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        auto tDiffSeconds = tDiff / 1000.0f;
        tStart = tEnd;

        // Render frame
        if (prepared) {
            render();
            update(tDiffSeconds);
        }
    }
}

std::string ExampleBase::getWindowTitle() {
    static const std::string deviceName = context.deviceInfo.properties.core10.deviceName;
    std::string windowTitle;
    windowTitle = title + " - " + deviceName + " - " + std::to_string(frameCounter) + " fps";
    return windowTitle;
}

#if ENABLE_UI
void ExampleBase::setupUi() {
    settings.overlay = settings.overlay && (!benchmark.active);
    if (!settings.overlay) {
        return;
    }

    struct vkx::ui::UIOverlayCreateInfo overlayCreateInfo;
    // Setup default overlay creation info
    overlayCreateInfo.colorFormat = swapChain.surfaceFormat.format;
    overlayCreateInfo.colorAttachmentViews = swapChain.getViews();
    // overlayCreateInfo.depthFormat = deviceInfo.supportedDepthFormat;
    // overlayCreateInfo.depthStencilView = depthStencilView;
    overlayCreateInfo.size = size;
    ImGui::SetCurrentContext(ImGui::CreateContext());
    // Virtual function call for example to customize overlay creation
    OnSetupUIOverlay(overlayCreateInfo);
    ui.create(graphicsQueue, overlayCreateInfo);
    // for (auto& shader : overlayCreateInfo.shaders) {
    //     device.destroy(shader.module);
    //     shader.module = vk::ShaderModule{};
    // }
    updateOverlay();
}
#endif

void ExampleBase::prepare() {
    setupSwapChainAndImages();
    allocateFrameData();

    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.magFilter = vk::Filter::eLinear;
    samplerCreateInfo.minFilter = vk::Filter::eLinear;
    samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    samplerCreateInfo.anisotropyEnable = context.enabledFeatures.core10.samplerAnisotropy;
    samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;
    defaultSampler = device.createSampler(samplerCreateInfo);

#if ENABLE_UI
    setupUi();
#endif
    loadAssets();
}

void ExampleBase::allocateFrameData() {
    clearFrameData();
    perFrameData.resize(swapChain.imageCount);
    // Create one command buffer per image in the swap chain
    // Command buffers store a reference to the
    // frame buffer inside their render pass info
    // so for static usage without having to rebuild
    // them each frame, we use one per frame buffer
    auto commandBuffers = graphicsQueue.allocateCommandBuffers(swapChain.imageCount);
    vk::SemaphoreTypeCreateInfo semaphoreTypeCreateInfo{ vk::SemaphoreType::eTimeline, 0 };
    for (size_t i = 0; i < swapChain.imageCount; i++) {
        auto& frameData = perFrameData[i];
        //// A semaphore used to synchronize command submission
        // frameData.semaphore = device.createSemaphore({ {}, &semaphoreTypeCreateInfo });
    }
}

void ExampleBase::clearFrameData() {
    perImageData.clear();
    for (auto& frameData : perFrameData) {
        frameData.destroy();
    }
    std::vector<vk::CommandBuffer> commandBuffers;
    for (auto& imageData : perImageData) {
        if (imageData.commandBuffer) {
            commandBuffers.push_back(imageData.commandBuffer);
            imageData.commandBuffer = nullptr;
        }
    }
    if (!commandBuffers.empty()) {
        recycler.trashCommandBuffers(graphicsQueue.pool, commandBuffers);
    }
}

void ExampleBase::updateCommandBufferPreDraw(const vk::CommandBuffer& commandBuffer) {
    const auto& swapChainImage = swapChain.images[currentIndex];

    vk::RenderingAttachmentInfo colorAttachmentInfo;
    colorAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.clearValue = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));
    colorAttachmentInfo.imageView = swapChainImage.view;

    vk::RenderingAttachmentInfo depthAttachmentInfo;
    depthAttachmentInfo.imageLayout = vk::ImageLayout::eAttachmentOptimal;
    depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachmentInfo.clearValue = vk::ClearDepthStencilValue{ 1.0, 0 };
    depthAttachmentInfo.imageView = depthStencilView;

    vk::RenderingInfo renderingInfo;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.layerCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;
    renderingInfo.pStencilAttachment = &depthAttachmentInfo;
    renderingInfo.renderArea = vk::Rect2D{ vk::Offset2D{}, size };

    // Because we're getting the image from our swapchain wrapper using a fence, we don't need to care about the prior layout.
    // Our swapchain mechanism prevents us from using an image until after it's been presented.
    swapChainImage.setLayout(commandBuffer, vks::util::ImageTransitionState::COLOR_ATTACHMENT);
    commandBuffer.beginRendering(renderingInfo);
}

void ExampleBase::updateCommandBufferPostDraw(const vk::CommandBuffer& commandBuffer) {
    commandBuffer.endRendering();
    const auto& swapChainImage = swapChain.images[currentIndex];
    swapChainImage.setLayout(commandBuffer, vks::util::ImageTransitionState::PRESENT);
}

void ExampleBase::buildCommandBuffers() {
    perImageData.resize(swapChain.imageCount);
    auto commandBuffers = graphicsQueue.allocateCommandBuffers(swapChain.imageCount);
    // Destroy and recreate command buffers if already present
    vk::CommandBufferBeginInfo cmdBufInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse };
    for (currentIndex = 0; currentIndex < swapChain.imageCount; ++currentIndex) {
        perImageData[currentIndex].commandBuffer = commandBuffers[currentIndex];
        auto& cmdBuffer = perImageData[currentIndex].commandBuffer;
        cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
        cmdBuffer.begin(cmdBufInfo);
        updateCommandBufferPreDraw(cmdBuffer);
        updateDrawCommandBuffer(cmdBuffer);
        updateCommandBufferPostDraw(cmdBuffer);
        cmdBuffer.end();
    }
}

void ExampleBase::prepareFrame() {
    // Acquire the next image from the swap chaing
    auto resultValue = swapChain.acquireNextImage(semaphores.swapchainAcquire.semaphore);
    if (resultValue.result == vk::Result::eSuboptimalKHR) {
        auto& loader = vks::Loader::get();
        windowResize(window.getSize());
        resultValue = swapChain.acquireNextImage(semaphores.swapchainAcquire.semaphore);
    }
    currentIndex = resultValue.value;
}

void ExampleBase::drawCurrentCommandBuffer() {
    queueCommandBuffer(perImageData[currentIndex].commandBuffer, RenderStates::RENDER_SCENE, vk::PipelineStageFlagBits2::eColorAttachmentOutput, true);
}

#if ENABLE_UI
void ExampleBase::drawCurrentUiBuffer() {
    if (ui.cmdBuffers.size() > currentIndex) {
        queueCommandBuffer(ui.cmdBuffers[currentIndex], RenderStates::RENDER_UI, vk::PipelineStageFlagBits2::eColorAttachmentOutput, true);
    }
}
#endif

const vks::QueueManager& ExampleBase::getQueue(uint32_t queueFamilyIndex) const {
    if (queueFamilyIndex == graphicsQueue.familyInfo.index || queueFamilyIndex == VK_QUEUE_FAMILY_IGNORED) {
        return graphicsQueue;
    } else if (queueFamilyIndex == computeQueue.familyInfo.index) {
        return computeQueue;
    } else if (queueFamilyIndex == transferQueue.familyInfo.index) {
        return transferQueue;
    }
    throw std::runtime_error("Invalid queue family index requested");
}

void ExampleBase::submitFrame() {
    auto& frameData = perFrameData[currentIndex];
    // Wait for the previous frame state (is a no-op on the first frame when the value will be 0)
    frameData.wait();
    frameData.reset();
    assert(currentFrameQueue.valid());

    // Number of command buffers for this frame
    auto size = currentFrameQueue.queuedCommands.size();
    const auto& semaphore = frameData.semaphore;

    static std::vector<vk::SemaphoreSubmitInfo> waitInfos;
    static std::vector<vk::SemaphoreSubmitInfo> signalInfos;
    bool acquired = false;

    vk::Fence fence = device.createFence(vk::FenceCreateInfo{});
    uint32_t lastQueueFamily = graphicsQueue.familyInfo.index;

    vk::SemaphoreSubmitInfo waitInfo{ semaphore, RenderStates::NONE, vk::PipelineStageFlagBits2::eNone };
    for (size_t i = 0; i < size; ++i) {
        auto& submissionInfo = currentFrameQueue.queuedCommands[i];
        waitInfos = submissionInfo.additionalWaits;
        signalInfos = submissionInfo.additionalSignals;
        if (!acquired && submissionInfo.requiresSwapchainImage) {
            assert(submissionInfo.queueFamilyIndex == VK_QUEUE_FAMILY_IGNORED || submissionInfo.queueFamilyIndex == graphicsQueue.familyInfo.index);
            waitInfos.push_back(semaphores.swapchainAcquire);
            acquired = true;
        }

        vk::SemaphoreSubmitInfo signalInfo{ semaphore, submissionInfo.timelineValue, submissionInfo.pipelineStages };
        signalInfos.push_back(signalInfo);
        if (currentFrameQueue.lastSwapchainAccess == i) {
            signalInfos.push_back(semaphores.swapchainFilled);
        }

        auto& queue = getQueue(submissionInfo.queueFamilyIndex);
        if (queue.familyInfo.index != lastQueueFamily) {
            waitInfo.stageMask = vk::PipelineStageFlagBits2::eNone;
        }
        waitInfos.push_back(waitInfo);
        if (i == size - 1) {
            queue.submit2(submissionInfo.cmdBuffer, waitInfos, signalInfos, fence);
        } else {
            queue.submit2(submissionInfo.cmdBuffer, waitInfos, signalInfos);
        }
        // The next bit will wait on the previous signal
        waitInfo = signalInfo;
    }

    frameData.finalState = waitInfo.value;
    currentFrameQueue.reset();
    auto presentResult = graphicsQueue.present(swapChain.handle, currentIndex, semaphores.swapchainFilled.semaphore);
    if (presentResult == vk::Result::eSuboptimalKHR || presentResult == vk::Result::eErrorOutOfDateKHR) {
        windowResize(window.getSize());
    }
    recycler.emptyDumpster(fence);
    recycler.recycle();
}

void ExampleBase::draw() {
    // Get next image in the swap chain (back/front buffer)
    prepareFrame();
    preRender();
    // Execute the compiled command buffer for the current swap chain image
    drawCurrentCommandBuffer();
#ifdef ENABLE_UI
    drawCurrentUiBuffer();
#endif
    postRender();
    // Push the rendered frame to the surface
    submitFrame();
}

void ExampleBase::render() {
    if (!prepared || size == EMPTY_RECT) {
        return;
    }
    draw();
}

void ExampleBase::update(float deltaTime) {
    frameTimer = deltaTime;
    ++frameCounter;

    camera.update(deltaTime);
    if (camera.moving()) {
        viewUpdated = true;
    }

    // Convert to clamped timer value
    if (!paused) {
        timer += timerSpeed * frameTimer;
        if (timer > 1.0) {
            timer -= 1.0f;
        }
    }
    fpsTimer += frameTimer;
    if (fpsTimer > 1.0f) {
#if !defined(__ANDROID__)
        window.setTitle(getWindowTitle());
#endif
        lastFPS = frameCounter;
        fpsTimer = 0.0f;
        frameCounter = 0;
    }

#if ENABLE_UI
    updateOverlay();
#endif

    // Check gamepad state
    const float deadZone = 0.0015f;
    // todo : check if gamepad is present
    // todo : time based and relative axis positions
    if (camera.type != Camera::CameraType::firstperson) {
        // Rotate
        if (std::abs(gamePadState.axisLeft.x) > deadZone) {
            camera.rotate(glm::vec3(0.0f, gamePadState.axisLeft.x * 0.5f, 0.0f));
            viewUpdated = true;
        }
        if (std::abs(gamePadState.axisLeft.y) > deadZone) {
            camera.rotate(glm::vec3(gamePadState.axisLeft.y * 0.5f, 0.0f, 0.0f));
            viewUpdated = true;
        }
        // Zoom
        if (std::abs(gamePadState.axisRight.y) > deadZone) {
            camera.dolly(gamePadState.axisRight.y * 0.01f * zoomSpeed);
            viewUpdated = true;
        }
    } else {
        viewUpdated |= camera.updatePad(gamePadState.axisLeft, gamePadState.axisRight, frameTimer);
    }

    if (viewUpdated) {
        viewUpdated = false;
        viewChanged();
    }
}

void ExampleBase::setupSwapChainAndImages() {
    vks::swapchain::Builder builder{ size, surface };
    builder.vsync = enableVsync;
    swapChain.create(builder);

    depthStencil.destroy();
    if (depthStencilView) {
        device.destroyImageView(depthStencilView);
        depthStencilView = nullptr;
    }

    // Create the depth stencil view, try to get a dedicated memory allocation for things like attachments.
    if (defaultDepthStencilFormat != vk::Format::eUndefined) {
        vks::Image::Builder depthStencilBuilder{ size };
        depthStencilBuilder.withFormat(defaultDepthStencilFormat);
        depthStencilBuilder.withUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc);
        depthStencil = depthStencilBuilder.build();
        depthStencilView = depthStencil.createView();
        loader.withPrimaryCommandBuffer([&](const vk::CommandBuffer& commandBuffer) {
            using namespace vks::util;
            setImageLayout(commandBuffer, depthStencil, ImageTransitionState::UNDEFINED, ImageTransitionState::DEPTH_ATTACHMENT);
        });
    }
}

void ExampleBase::windowResize(const vk::Extent2D& newSize) {
    if (!prepared) {
        return;
    }
    vks::Loader::get().lock(false);
    prepared = false;

    waitIdle();

    // Recreate swap chain
    size = newSize;
    if (size == vk::Extent2D{ 0, 0 }) {
        return;
    }

    setupSwapChainAndImages();

#if ENABLE_UI
    if (settings.overlay) {
        ui.resize(size, swapChain.getViews());
    }
#endif

    // Notify derived class
    windowResized();

    // Command buffers need to be recreated as they may store
    // references to the recreated frame buffer
    allocateFrameData();
    buildCommandBuffers();

    viewChanged();

    prepared = true;
    vks::Loader::get().lock(true);
}

#if ENABLE_UI
void ExampleBase::updateOverlay() {
    if (!settings.overlay) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();

    io.DisplaySize = ImVec2((float)size.width, (float)size.height);
    io.DeltaTime = frameTimer;

    io.MousePos = ImVec2(mousePos.x, mousePos.y);
    io.MouseDown[0] = mouseButtons.left;
    io.MouseDown[1] = mouseButtons.right;

    ImGui::NewFrame();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Vulkan Example", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::TextUnformatted(title.c_str());
    ImGui::TextUnformatted(context.deviceInfo.properties.core10.deviceName);
    ImGui::Text("%.2f ms/frame (%.1d fps)", (1000.0f / lastFPS), lastFPS);

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 5.0f * ui.scale));
#endif
    ImGui::PushItemWidth(110.0f * ui.scale);
    OnUpdateUIOverlay();
    ImGui::PopItemWidth();
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    ImGui::PopStyleVar();
#endif

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::Render();

    ui.update();

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    if (mouseButtons.left) {
        mouseButtons.left = false;
    }
#endif
}

#endif

void ExampleBase::mouseMoved(const glm::vec2& newPos) {
#if ENABLE_UI
    if (settings.overlay) {
        auto imgui = ImGui::GetIO();
        if (imgui.WantCaptureMouse) {
            mousePos = newPos;
            return;
        }
    }
#endif
    glm::vec2 deltaPos = mousePos - newPos;
    if (deltaPos == vec2()) {
        return;
    }

    const auto& dx = deltaPos.x;
    const auto& dy = deltaPos.y;

    if (mouseButtons.left) {
        camera.rotate(glm::vec3(dy * camera.rotationSpeed, -dx * camera.rotationSpeed, 0.0f));
        viewUpdated = true;
    }
    if (mouseButtons.right) {
        camera.dolly(dy * .005f * zoomSpeed);
        viewUpdated = true;
    }
    if (mouseButtons.middle) {
        camera.translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.0f));
        viewUpdated = true;
    }
    mousePos = newPos;
}

void ExampleBase::mouseScrolled(float delta) {
    camera.translate(glm::vec3(0.0f, 0.0f, (float)delta * 0.005f * zoomSpeed));
    viewUpdated = true;
}

void ExampleBase::keyPressed(uint32_t key) {
    if (camera.firstperson) {
        switch (key) {
            case KEY_W:
                camera.keys.up = true;
                break;
            case KEY_S:
                camera.keys.down = true;
                break;
            case KEY_A:
                camera.keys.left = true;
                break;
            case KEY_D:
                camera.keys.right = true;
                break;
        }
    }

    switch (key) {
        case KEY_P:
            paused = !paused;
            break;

#if ENABLE_UI
        case KEY_F1:
            ui.visible = !ui.visible;
            break;
#endif

        case KEY_ESCAPE:
#if defined(__ANDROID__)
#else
            window.close();
#endif
            break;

        default:
            break;
    }
}

void ExampleBase::keyReleased(uint32_t key) {
    if (camera.firstperson) {
        switch (key) {
            case KEY_W:
                camera.keys.up = false;
                break;
            case KEY_S:
                camera.keys.down = false;
                break;
            case KEY_A:
                camera.keys.left = false;
                break;
            case KEY_D:
                camera.keys.right = false;
                break;
        }
    }
}

#if defined(__ANDROID__)

int32_t ExampleBase::handle_input_event(android_app* app, AInputEvent* event) {
    ExampleBase* exampleBase = reinterpret_cast<ExampleBase*>(app->userData);
    return exampleBase->onInput(event);
}

void ExampleBase::handle_app_cmd(android_app* app, int32_t cmd) {
    ExampleBase* exampleBase = reinterpret_cast<ExampleBase*>(app->userData);
    exampleBase->onAppCmd(cmd);
}

int32_t ExampleBase::onInput(AInputEvent* event) {
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        bool handled = false;
        ivec2 touchPoint;
        int32_t eventSource = AInputEvent_getSource(event);
        switch (eventSource) {
            case AINPUT_SOURCE_TOUCHSCREEN: {
                int32_t action = AMotionEvent_getAction(event);

                switch (action) {
                    case AMOTION_EVENT_ACTION_UP:
                        mouseButtons.left = false;
                        break;

                    case AMOTION_EVENT_ACTION_DOWN:
                        // Detect double tap
                        mouseButtons.left = true;
                        mousePos.x = AMotionEvent_getX(event, 0);
                        mousePos.y = AMotionEvent_getY(event, 0);
                        break;

                    case AMOTION_EVENT_ACTION_MOVE:
                        touchPoint.x = AMotionEvent_getX(event, 0);
                        touchPoint.y = AMotionEvent_getY(event, 0);
                        mouseMoved(vec2{ touchPoint });
                        break;

                    default:
                        break;
                }
            }
                return 1;
        }
    }

    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY) {
        int32_t keyCode = AKeyEvent_getKeyCode((const AInputEvent*)event);
        int32_t action = AKeyEvent_getAction((const AInputEvent*)event);
        int32_t button = 0;

        if (action == AKEY_EVENT_ACTION_UP)
            return 0;

        switch (keyCode) {
            case AKEYCODE_BUTTON_A:
                keyPressed(GAMEPAD_BUTTON_A);
                break;
            case AKEYCODE_BUTTON_B:
                keyPressed(GAMEPAD_BUTTON_B);
                break;
            case AKEYCODE_BUTTON_X:
                keyPressed(GAMEPAD_BUTTON_X);
                break;
            case AKEYCODE_BUTTON_Y:
                keyPressed(GAMEPAD_BUTTON_Y);
                break;
            case AKEYCODE_BUTTON_L1:
                keyPressed(GAMEPAD_BUTTON_L1);
                break;
            case AKEYCODE_BUTTON_R1:
                keyPressed(GAMEPAD_BUTTON_R1);
                break;
            case AKEYCODE_BUTTON_START:
                paused = !paused;
                break;
        };
    }
    return 0;
}

void ExampleBase::onAppCmd(int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (vkx::android::androidApp->window != nullptr) {
                setupWindow();
                initVulkan();
                setupSwapchain();
                prepare();
            }
            break;
        case APP_CMD_LOST_FOCUS:
            focused = false;
            break;
        case APP_CMD_GAINED_FOCUS:
            focused = true;
            break;
        default:
            break;
    }
}

void ExampleBase::setupWindow() {
    window = vkx::android::androidApp->window;
    size.width = ANativeWindow_getWidth(window);
    size.height = ANativeWindow_getHeight(window);
    camera.updateAspectRatio(size);
}

#else

void ExampleBase::setupWindow() {
    bool fullscreen = false;

    window.setMouseEventHandler([&](int button, int action, int mods) {
        switch (button) {
            case GLFW_MOUSE_BUTTON_LEFT:
                mouseButtons.left = action == GLFW_PRESS;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                mouseButtons.right = action == GLFW_PRESS;
                break;
            case GLFW_MOUSE_BUTTON_MIDDLE:
                mouseButtons.middle = action == GLFW_PRESS;
                break;
        }
    });
    window.setKeyPressedHandler([&](int key, int mods) { keyPressed(key); });
    window.setKeyReleasedHandler([&](int key, int mods) { keyReleased(key); });
    window.setMouseMovedHandler([&](float x, float y) { mouseMoved({ x, y }); });
    window.setMouseScrolledHandler([&](float scrolly) { mouseScrolled(scrolly); });
    window.setResizeHandler([&](const vk::Extent2D& size) { windowResize(size); });
    window.setCloseHandler([&]() { window.close(); });

#ifdef _WIN32
    // Check command line arguments
    for (int32_t i = 0; i < __argc; i++) {
        if (__argv[i] == std::string("-fullscreen")) {
            fullscreen = true;
        }
    }
#endif

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto monitor = glfwGetPrimaryMonitor();
    auto mode = glfwGetVideoMode(monitor);
    size.width = mode->width;
    size.height = mode->height;
    if (fullscreen) {
        window.createWindow(size, "My Title", monitor);
    } else {
        size.width /= 2;
        size.height /= 2;
        window.createWindow(size, "My Title", nullptr);
    }
    if (!window) {
        throw std::runtime_error("Could not create window");
    }
}
#endif

#if 0
#if defined(__ANDROID__)
    bool destroy = false;
    focused = true;
    int ident, events;
    struct android_poll_source* source;
    while (!destroy && (ident = ALooper_pollAll(focused ? 0 : -1, NULL, &events, (void**)&source)) >= 0) {
        if (source != NULL) {
            source->process(vkx::android::androidApp, source);
        }
        destroy = vkx::android::androidApp->destroyRequested != 0;
    }

    // App destruction requested
    // Exit loop, example will be destroyed in application main
    return !destroy;
#else

    if (0 != glfwWindowShouldClose(window)) {
        return false;
    }

    glfwPollEvents();

    if (0 != glfwJoystickPresent(0)) {
        // FIXME implement joystick handling
        int axisCount{ 0 };
        const float* axes = glfwGetJoystickAxes(0, &axisCount);
        if (axisCount >= 2) {
            gamePadState.axisLeft.x = axes[0] * 0.01f;
            gamePadState.axisLeft.y = axes[1] * -0.01f;
        }
        if (axisCount >= 4) {
            gamePadState.axisRight.x = axes[0] * 0.01f;
            gamePadState.axisRight.y = axes[1] * -0.01f;
        }
        if (axisCount >= 6) {
            float lt = (axes[4] + 1.0f) / 2.0f;
            float rt = (axes[5] + 1.0f) / 2.0f;
            gamePadState.rz = (rt - lt);
        }
        uint32_t newButtons{ 0 };
        static uint32_t oldButtons{ 0 };
        {
            int buttonCount{ 0 };
            const uint8_t* buttons = glfwGetJoystickButtons(0, &buttonCount);
            for (uint8_t i = 0; i < buttonCount && i < 64; ++i) {
                if (0 != buttons[i]) {
                    newButtons |= (1 << i);
                }
            }
        }
        auto changedButtons = newButtons & ~oldButtons;
        if (changedButtons & 0x01) {
            keyPressed(GAMEPAD_BUTTON_A);
        }
        if (changedButtons & 0x02) {
            keyPressed(GAMEPAD_BUTTON_B);
        }
        if (changedButtons & 0x04) {
            keyPressed(GAMEPAD_BUTTON_X);
        }
        if (changedButtons & 0x08) {
            keyPressed(GAMEPAD_BUTTON_Y);
        }
        if (changedButtons & 0x10) {
            keyPressed(GAMEPAD_BUTTON_L1);
        }
        if (changedButtons & 0x20) {
            keyPressed(GAMEPAD_BUTTON_R1);
        }
        oldButtons = newButtons;
    } else {
        memset(&gamePadState, 0, sizeof(gamePadState));
    }
    return true;
#endif
#endif