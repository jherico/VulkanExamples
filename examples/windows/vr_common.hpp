#include <common.hpp>
#include <vulkanContext.hpp>
#include <vulkanSwapChain.hpp>
#include <vulkanShapes.hpp>

class VrExample : glfw::Window {
    using Parent = glfw::Window;
public:
    vkx::Context context;
    vkx::SwapChain swapChain { context };
    std::shared_ptr<vkx::ShapesRenderer> shapesRenderer { std::make_shared<vkx::ShapesRenderer>(context, true) };
    double fpsTimer { 0 };
    float lastFPS { 0 };
    uint32_t frameCounter { 0 };
    glm::uvec2 size { 1280, 720 };
    glm::uvec2 renderTargetSize;
    std::array<glm::mat4, 2> eyeViews;
    std::array<glm::mat4, 2> eyeProjections;

    ~VrExample() {
        shapesRenderer.reset();

        // Shut down Vulkan 
        context.destroyContext();

        // Shut down GLFW
        if (nullptr != window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }

    typedef void(*GLFWkeyfun)(GLFWwindow*, int, int, int, int);

    void prepareWindow() {
        // Make the on screen window 1/4 the resolution of the render target
        size = renderTargetSize;
        size /= 4;

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(size.x, size.y, "glfw", nullptr, nullptr);
        if (!window) {
            throw std::runtime_error("Unable to create rendering window");
        }
        context.addInstanceExtensionPicker([]()->std::set<std::string> {
            return glfw::getRequiredInstanceExtensions();
        });

        Parent::prepareWindow();
    }

    void prepareVulkan() {
        //context.setValidationEnabled(true);
        context.createContext();
    }

    void prepareSwapchain() {
        swapChain.createSurface(window);
        auto extent2d = vk::Extent2D { size.x, size.y };
        swapChain.create(extent2d);
    }

    void prepareRenderer() {
        shapesRenderer->framebufferSize = renderTargetSize;
        shapesRenderer->colorFormats = { vk::Format::eR8G8B8A8Srgb };
        shapesRenderer->prepare();
    }

    virtual void recenter() = 0;

    void keyEvent(int key, int scancode, int action, int mods) override {
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

    virtual void update(float delta) {
        shapesRenderer->update(delta, eyeProjections, eyeViews);
    }

    virtual void render() = 0;

    virtual std::string getWindowTitle() = 0;

    void run() {
        prepare();
        auto tStart = std::chrono::high_resolution_clock::now();
        static auto lastFrameCounter = frameCounter;
        while (!glfwWindowShouldClose(window)) {
            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            glfwPollEvents();
            update((float)tDiff / 1000.0f);
            render();
            ++frameCounter;
            fpsTimer += (float)tDiff;
            if (fpsTimer > 1000.0f) {
                std::string windowTitle = getWindowTitle();
                glfwSetWindowTitle(window, windowTitle.c_str());
                lastFPS = (float)(frameCounter - lastFrameCounter);
                lastFPS *= 1000.0f;
                lastFPS /= (float)fpsTimer;
                fpsTimer = 0.0f;
                lastFrameCounter = frameCounter;
            }
            tStart = tEnd;
        }
    }
};


