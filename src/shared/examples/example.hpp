/*
* Vulkan Example base class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "common.hpp"

#include "vks/vks.hpp"
#include "vks/helpers.hpp"
#include "vks/filesystem.hpp"
#include "vks/model.hpp"
#include "vks/shaders.hpp"
#include "vks/pipelines.hpp"
#include "vks/texture.hpp"

#include "ui.hpp"
#include "utils.hpp"
#include "camera.hpp"
#include "compute.hpp"

#if defined(__ANDROID__)
#include "AndroidNativeApp.hpp"
#endif

#define GAMEPAD_BUTTON_A 0x1000
#define GAMEPAD_BUTTON_B 0x1001
#define GAMEPAD_BUTTON_X 0x1002
#define GAMEPAD_BUTTON_Y 0x1003
#define GAMEPAD_BUTTON_L1 0x1004
#define GAMEPAD_BUTTON_R1 0x1005
#define GAMEPAD_BUTTON_START 0x1006

namespace vkx {

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

class ExampleBase {
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
    void windowResize(const glm::uvec2& newSize);

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
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::ClearValue> clearValues;
    vk::RenderPassBeginInfo renderPassBeginInfo;
    vk::Viewport viewport() { return vks::util::viewport(size); }
    vk::Rect2D scissor() { return vks::util::rect2D(size); }

    virtual void clearCommandBuffers() final;
    virtual void allocateCommandBuffers() final;
    virtual void setupRenderPassBeginInfo();
    virtual void buildCommandBuffers();

protected:
    // Last frame time, measured using a high performance timer (if available)
    float frameTimer{ 0.0015f };
    // Frame counter to display fps
    uint32_t frameCounter{ 0 };
    uint32_t lastFPS{ 0 };

    // Color buffer format
    vk::Format colorformat{ vk::Format::eB8G8R8A8Unorm };

    // Depth buffer format...  selected during Vulkan initialization
    vk::Format depthFormat{ vk::Format::eUndefined };

    // Global render pass for frame buffer writes
    vk::RenderPass renderPass;

    // List of available frame buffers (same as number of swap chain images)
    std::vector<vk::Framebuffer> framebuffers;
    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    vk::DescriptorPool descriptorPool;

    void addRenderWaitSemaphore(const vk::Semaphore& semaphore, const vk::PipelineStageFlags& waitStages = vk::PipelineStageFlagBits::eBottomOfPipe);

    std::vector<vk::Semaphore> renderWaitSemaphores;
    std::vector<vk::PipelineStageFlags> renderWaitStages;
    std::vector<vk::Semaphore> renderSignalSemaphores;

    vks::Context context;
    const vk::PhysicalDevice& physicalDevice{ context.physicalDevice };
    const vk::Device& device{ context.device };
    const vk::Queue& queue{ context.queue };
    const vk::PhysicalDeviceFeatures& deviceFeatures{ context.deviceFeatures };
    vk::PhysicalDeviceFeatures& enabledFeatures{ context.enabledFeatures };
    vkx::ui::UIOverlay ui{ context };

    vk::SurfaceKHR surface;
    // Wraps the swap chain to present images (framebuffers) to the windowing system
    vks::SwapChain swapChain;

    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        vk::Semaphore acquireComplete;
        // Command buffer submission and execution
        vk::Semaphore renderComplete;
        // UI buffer submission and execution
        vk::Semaphore overlayComplete;
#if 0
        vk::Semaphore transferComplete;
#endif
    } semaphores;

    // Returns the base asset path (for shaders, models, textures) depending on the os
    const std::string& getAssetPath() { return ::vkx::getAssetPath(); }

protected:
    /** @brief Example settings that can be changed e.g. by command line arguments */
    struct Settings {
        /** @brief Activates validation layers (and message output) when set to true */
        bool validation = false;
        /** @brief Set to true if fullscreen mode has been requested via command line */
        bool fullscreen = false;
        /** @brief Set to true if v-sync will be forced for the swapchain */
        bool vsync = false;
        /** @brief Enable UI overlay */
        bool overlay = true;
    } settings;

    struct {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    struct {
        bool active = false;
    } benchmark;

    // Command buffer pool
    vk::CommandPool cmdPool;

    bool prepared = false;
    uint32_t version = VK_MAKE_VERSION(1, 1, 0);
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

    // Gamepad state (only one pad supported)
    struct {
        glm::vec2 axisLeft = glm::vec2(0.0f);
        glm::vec2 axisRight = glm::vec2(0.0f);
        float rz{ 0.0f };
    } gamePadState;

    void updateOverlay();

    virtual void OnUpdateUIOverlay() {}
    virtual void OnSetupUIOverlay(vkx::ui::UIOverlayCreateInfo& uiCreateInfo) {}

    // Setup the vulkan instance, enable required extensions and connect to the physical device (GPU)
    virtual void initVulkan();
    virtual void setupSwapchain();
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
    void setupDepthStencil();
    // Create framebuffers for all requested swap chain images
    // Can be overriden in derived class to setup a custom framebuffer (e.g. for MSAA)
    virtual void setupFrameBuffer();

    // Setup a default render pass
    // Can be overriden in derived class to setup a custom render pass (e.g. for MSAA)
    virtual void setupRenderPass();

    void setupUi();

    virtual void updateCommandBufferPreDraw(const vk::CommandBuffer& commandBuffer) {}

    virtual void updateDrawCommandBuffer(const vk::CommandBuffer& commandBuffer) {}

    virtual void updateCommandBufferPostDraw(const vk::CommandBuffer& commandBuffer) {}

    void drawCurrentCommandBuffer();

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
    ANativeWindow* window{ nullptr};
    bool focused = false;
    static int32_t handle_input_event(android_app* app, AInputEvent* event);
    int32_t onInput(AInputEvent* event);
    static void handle_app_cmd(android_app* app, int32_t cmd);
    void onAppCmd(int32_t cmd);
#else
    GLFWwindow* window{ nullptr };
    // Keyboard movement handler
    virtual void mouseAction(int buttons, int action, int mods);
    static void KeyboardHandler(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void MouseHandler(GLFWwindow* window, int button, int action, int mods);
    static void MouseMoveHandler(GLFWwindow* window, double posx, double posy);
    static void MouseScrollHandler(GLFWwindow* window, double xoffset, double yoffset);
    static void FramebufferSizeHandler(GLFWwindow* window, int width, int height);
    static void CloseHandler(GLFWwindow* window);
#endif
};
}  // namespace vkx
