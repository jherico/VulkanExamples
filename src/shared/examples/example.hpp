/*
* Vulkan Example base class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <common/common.hpp>
#include <common/utils.hpp>

#include <vks/vks.hpp>
#include <vks/pipelines.hpp>
#include <vks/helpers.hpp>

#include <rendering/recycler.hpp>
#include <rendering/model.hpp>
#include <rendering/camera.hpp>
#include <rendering/swapchain.hpp>
#include <rendering/framequeue.hpp>

#include <examples/glfw.hpp>
#include <examples/keycodes.hpp>
#ifdef ENABLE_UI
#include <examples/ui.hpp>
#endif

namespace vkx {

constexpr size_t INVALID_VECTOR_INDEX = static_cast<size_t>(-1);

struct UpdateOperation {
    const vk::Buffer buffer;
    const vk::DeviceSize size;
    const vk::DeviceSize offset;
    const uint32_t* data;

    template <typename T>
    UpdateOperation(const vk::Buffer& buffer, const T& data, vk::DeviceSize offset = 0)
        : buffer(buffer)
        , size(sizeof(T))
        , offset(offset)
        , data((uint32_t*)&data) {
        assert(0 == (sizeof(T) % 4));
        assert(0 == (offset % 4));
    }
};

enum RenderStates : uint64_t
{
    NONE = 0x0,
    COMPUTE_PRERENDER = 0x0000000000000200,
    OFFSCREEN_PRERENDER = 0x0000000000000300,
    RENDER_SCENE = 0x0000000000000400,
    RENDER_UI = 0x0000000000000500,
    COMPUTE_POST = 0x0000000000000600,
    OFFSCREEN_POST = 0x0000000000000700,
    COMPOSITE = 0x0000000000000800,
};

struct PerImageData {
    vk::CommandBuffer commandBuffer;
};

struct PerFrameData {
    vk::Semaphore semaphore;
    uint32_t imageIndex{ 0 };
    uint64_t finalState{ 0 };

    void wait();
    void reset();
    void destroy();
};

class ExampleBase {
    static vk::Extent2D EMPTY_RECT;

protected:
    ExampleBase();
    ~ExampleBase();

    using vAF = vk::AccessFlagBits;
    using vBU = vk::BufferUsageFlagBits;
    using vDT = vk::DescriptorType;
    using vF = vk::Format;
    using vIL = vk::ImageLayout;
    using vIT = vk::ImageType;
    using vIVT = vk::ImageViewType;
    using vIU = vk::ImageUsageFlagBits;
    using vIA = vk::ImageAspectFlagBits;
    using vMP = vk::MemoryPropertyFlagBits;
    using vPS = vk::PipelineStageFlagBits;
    using vSS = vk::ShaderStageFlagBits;

public:
    void run();
    // Called if the window is resized and some resources have to be recreatesd
    void windowResize(const vk::Extent2D& newSize);

private:
    // Set to true when the debug marker extension is detected
    bool enableDebugMarkers{ false };
    // fps timer (one second interval)
    float fpsTimer = 0.0f;
    // Get window title with example name, device, et.
    std::string getWindowTitle();

protected:
    bool enableVsync{ false };

    // Command buffers used for rendering
    std::vector<PerFrameData> perFrameData;
    std::vector<PerImageData> perImageData;
    vks::frame::QueuedCommands currentFrameQueue;

    vk::Viewport viewport() { return vks::util::viewport(size); }
    vk::Rect2D scissor() { return vks::util::rect2D(size); }

    const vks::QueueManager& getQueue(uint32_t queueFamilyIndex) const;
    virtual void clearFrameData() final;
    virtual void allocateFrameData() final;

    virtual void buildCommandBuffers();

protected:
    // Last frame time, measured using a high performance timer (if available)
    float frameTimer{ 0.0015f };
    // Frame counter to display fps
    uint32_t frameCounter{ 0 };
    uint32_t lastFPS{ 0 };
    // Active frame buffer index
    uint32_t currentIndex = 0;
    // Descriptor set pool
    vk::DescriptorPool descriptorPool;

    vks::Context& context{ vks::Context::get() };
    vks::Loader& loader{ vks::Loader::get() };
    vks::Recycler& recycler{ vks::Recycler::get() };

    // The Vulkan instance
    const vk::Instance& instance{ context.instance };
    const vks::DeviceInfo& deviceInfo{ context.deviceInfo };
    // The physical device handle, providing access to information about the device capabilities and features
    const vk::PhysicalDevice& physicalDevice{ context.physicalDevice };
    // The logical device handle, providing access to runtime functionality
    const vk::Device& device{ context.device };
    const vk::Format& defaultColorFormat{ swapChain.surfaceFormat.format };
    const vk::Format& defaultDepthStencilFormat{ deviceInfo.supportedDepthFormat };
    vk::SurfaceKHR surface;
    vk::PipelineCache& pipelineCache{ context.pipelineCache };
    vks::QueueManager graphicsQueue;
    vks::QueueManager computeQueue;
    vks::QueueManager transferQueue;
    vk::Sampler defaultSampler;

#if ENABLE_UI
    vkx::ui::UIOverlay ui;
#endif

    // Wraps the swap chain to present images (framebuffers) to the windowing system
    vks::swapchain::Swapchain swapChain;

    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        vk::SemaphoreSubmitInfo swapchainAcquire;
        vk::SemaphoreSubmitInfo swapchainFilled;
    } semaphores;

    // Returns the base asset path (for shaders, models, textures) depending on the os
    const std::string& getAssetPath() { return ::vkx::getAssetPath(); }

protected:
    void queueCommandBuffer(const vks::frame::QueuedCommandBuilder& builder);
    void queueCommandBuffer(const vk::CommandBuffer& cmdBuffer,
                            uint64_t value,
                            const vk::PipelineStageFlags2& pipelineStages,
                            bool requiresSwapchainImage = false,
                            const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalWaitSemaphores = nullptr,
                            const vk::ArrayProxy<const vk::SemaphoreSubmitInfo>& additionalSignalSemaphores = nullptr);

    /** @brief Activates validation layers (and message output) when set to true */
    bool& validation{ context.enableValidation };

    /** @brief Example settings that can be changed e.g. by command line arguments */
    struct Settings {
        /** @brief Set to true if fullscreen mode has been requested via command line */
        bool fullscreen = false;
        /** @brief Set to true if v-sync will be forced for the swapchain */
        bool vsync = false;
#if ENABLE_UI
        /** @brief Enable UI overlay */
        bool overlay = true;
#endif
    } settings;

    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    struct {
        bool active = false;
    } benchmark;

    bool prepared = false;
    vk::Result lastPresent{ vk::Result::eSuccess };
    uint32_t version{ VK_MAKE_VERSION(1, 3, 0) };
    vk::Extent2D size{ 1280, 720 };
    uint32_t& width{ size.width };
    uint32_t& height{ size.height };

    vk::ClearColorValue defaultClearColor = vks::util::clearColor(glm::vec4({ 0.025f, 0.025f, 0.025f, 1.0f }));
    vk::ClearDepthStencilValue defaultClearDepth{ 1.0f, 0 };

    // Defines a frame rate independent timer value clamped from -1.0...1.0
    // For use in animations, rotations, etc.
    float timer = 0.0f;

    // Multiplier for speeding up (or slowing down) the global timer
    float timerSpeed = 0.25f;

    bool paused = false;

    // Use to adjust mouse rotation speed
    float rotationSpeed = 1.0f;
    // Use to adjust mouse zoom speed
    float zoomSpeed = 1.0f;

    Camera camera;
    glm::vec2 mousePos;
    bool viewUpdated{ false };

    std::string title = "Vulkan Example";
    std::string name = "vulkanExample";
    vks::Image depthStencil;
    vk::ImageView depthStencilView;

    // Gamepad state (only one pad supported)
    struct {
        glm::vec2 axisLeft = glm::vec2(0.0f);
        glm::vec2 axisRight = glm::vec2(0.0f);
        float rz{ 0.0f };
    } gamePadState;

#if ENABLE_UI
    void updateOverlay();
    virtual void OnUpdateUIOverlay() {}
    virtual void OnSetupUIOverlay(vkx::ui::UIOverlayCreateInfo& uiCreateInfo) {}
#endif

    // Setup the vulkan instance, enable required extensions and connect to the physical device (GPU)
    virtual void initVulkan();
    virtual void waitIdle();
    virtual void setupWindow();
    virtual void getEnabledFeatures();
    // A default draw implementation
    virtual void draw();
    // Basic render function
    virtual void render();
    virtual void update(float deltaTime);
    // Called when view change occurs
    // Can be overriden in derived class to e.g. update uniform buffers
    // Containing view dependant matrices
    virtual void viewChanged() {}

    // Called when the window has been resized
    // Can be overriden in derived class to recreate or rebuild resources attached to the frame buffer / swapchain
    virtual void windowResized() {}

    // Setup default depth and stencil views
    virtual void setupSwapChainAndImages();
    //void setupDepthStencil();
    //virtual void createSwapchain();

#if ENABLE_UI
    void setupUi();
    void drawCurrentUiBuffer();
#endif

    // Placeholders for things to happen before and after rendering the frame, such as for external graphicsQueue or API operations
    virtual void preRender() {}
    virtual void postRender() {}

    virtual void updateCommandBufferPreDraw(const vk::CommandBuffer& commandBuffer);

    virtual void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) {}

    virtual void updateCommandBufferPostDraw(const vk::CommandBuffer& commandBuffer);

    virtual void drawCurrentCommandBuffer();

    // Prepare commonly used Vulkan functions
    virtual void prepare();

    virtual void loadAssets() {}

    bool platformLoopCondition();

    // Start the main render loop
    void renderLoop();

    // Prepare the frame for workload submission
    // - Acquires the next image from the swap chain
    // - Submits a post present barrier
    // - Sets the default wait and signal semaphores
    void prepareFrame();

    // Submit the frames' workload
    // - Submits the text overlay (if enabled)
    // -
    void submitFrame();

    virtual const glm::mat4& getProjection() const { return camera.matrices.perspective; }

    virtual const glm::mat4& getView() const { return camera.matrices.view; }

    // Called if a key is pressed
    // Can be overriden in derived class to do custom key handling
    virtual void keyPressed(uint32_t key);
    virtual void keyReleased(uint32_t key);

    virtual void mouseMoved(const glm::vec2& newPos);
    virtual void mouseScrolled(float delta);

private:
    // OS specific
#if defined(__ANDROID__)
    // true if application has focused, false if moved to background
    ANativeWindow* window{ nullptr };
    bool focused = false;
    static int32_t handle_input_event(android_app* app, AInputEvent* event);
    int32_t onInput(AInputEvent* event);
    static void handle_app_cmd(android_app* app, int32_t cmd);
    void onAppCmd(int32_t cmd);
#else
    glfw::Window window;
#endif
};
}  // namespace vkx

// Image loading
