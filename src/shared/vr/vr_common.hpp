#include "../common.hpp"
#include "../vks/context.hpp"
#include "../vks/swapchain.hpp"
#include "../shapes.h"
#include "../shapesRenderer.hpp"

class VrExample : glfw::Window {
    using Parent = glfw::Window;

public:
    vks::Context context;
    vks::SwapChain swapchain;
    vk::SurfaceKHR surface;
    std::shared_ptr<vkx::ShapesRenderer> shapesRenderer{ std::make_shared<vkx::ShapesRenderer>(context, true) };
    double fpsTimer{ 0 };
    float lastFPS{ 0 };
    uint32_t frameCounter{ 0 };
    glm::uvec2 size{ 1280, 720 };
    glm::uvec2 renderTargetSize;
    std::array<glm::mat4, 2> eyeViews;
    std::array<glm::mat4, 2> eyeProjections;

    ~VrExample() {
        shapesRenderer.reset();
        // Shut down Vulkan
        context.destroy();
    }

    typedef void (*GLFWkeyfun)(GLFWwindow*, int, int, int, int);

    void prepareWindow() {
        // Make the on screen window 1/4 the resolution of the render target
        size = renderTargetSize;
        size /= 4;

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        createWindow(size);
        context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
    }

    void prepareVulkan() {
        context.createInstance();
        surface = createSurface(context.instance);
        context.createDevice(surface);
    }

    void prepareSwapchain() {
        swapchain.setup(context.physicalDevice, context.device, context.queue, context.queueIndices.graphics);
        swapchain.setSurface(surface);
        swapchain.create(vk::Extent2D{ size.x, size.y });
    }

    void prepareRenderer() {
        shapesRenderer->framebufferSize = renderTargetSize;
        shapesRenderer->colorFormats = { vk::Format::eR8G8B8A8Srgb };
        shapesRenderer->prepare();
    }

    virtual void recenter() = 0;

    void onKeyEvent(int key, int scancode, int action, int mods) override {
        switch (key) {
            case GLFW_KEY_R:
                recenter();
            default:
                break;
        }
    }

    virtual void prepare() {
        prepareWindow();
        prepareVulkan();
        prepareSwapchain();
        prepareRenderer();
    }

    virtual void update(float delta) { shapesRenderer->update(delta, eyeProjections, eyeViews); }

    virtual void render() = 0;

    virtual std::string getWindowTitle() = 0;

    void run() {
        prepare();
        auto tStart = std::chrono::high_resolution_clock::now();
        static auto lastFrameCounter = frameCounter;
        runWindowLoop([&] {
            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            update((float)tDiff / 1000.0f);
            render();
            fpsTimer += (float)tDiff;
            if (fpsTimer > 1000.0f) {
                setTitle(getWindowTitle());
                lastFPS = (float)(frameCounter - lastFrameCounter);
                lastFPS *= 1000.0f;
                lastFPS /= (float)fpsTimer;
                fpsTimer = 0.0f;
                lastFrameCounter = frameCounter;
            }
            tStart = tEnd;
            ++frameCounter;
        });
    }
};
