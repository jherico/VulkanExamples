#pragma once

#if !defined(ANDROID)
#include <string>
#include <vector>
#include <functional>
#include <set>
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

namespace glfw {

class Monitor {
private:
    GLFWmonitor* monitor{ nullptr };
};

class Window {
public:
    Window();
    static bool init();
    static void terminate();
    static std::vector<std::string> getRequiredInstanceExtensions();
    static vk::SurfaceKHR createWindowSurface(GLFWwindow* window, const vk::Instance& instance, const vk::AllocationCallbacks* pAllocator = nullptr);

    vk::SurfaceKHR createSurface(const vk::Instance& instance, const vk::AllocationCallbacks* pAllocator = nullptr) {
        return createWindowSurface(window, instance, pAllocator);
    }

    void swapBuffers() const { glfwSwapBuffers(window); }

    vk::Extent2D getSize() {
        vk::Extent2D result;
        glfwGetWindowSize(window, &(int&)(result.width), &(int&)(result.height));
        return result;
    }
    void createWindow(const vk::Extent2D& size, const std::string& title, GLFWmonitor* monitor = nullptr) {
        createWindow(vk::Rect2D{ vk::Offset2D{}, size }, title, monitor);
    }

    void createWindow(const vk::Rect2D& rect, const std::string& title, GLFWmonitor* monitor = nullptr) {
        // Disable window resize
        window = glfwCreateWindow(rect.extent.width, rect.extent.height, "Window Title", monitor, nullptr);
        if (rect.offset != vk::Offset2D{}) {
            glfwSetWindowPos(window, rect.offset.x, rect.offset.y);
        }
        glfwSetWindowUserPointer(window, this);
        glfwSetKeyCallback(window, KeyboardHandler);
        glfwSetMouseButtonCallback(window, MouseButtonHandler);
        glfwSetCursorPosCallback(window, MouseMoveHandler);
        glfwSetWindowCloseCallback(window, WindowCloseHandler);
        glfwSetFramebufferSizeCallback(window, FramebufferSizeHandler);
        glfwSetScrollCallback(window, MouseScrollHandler);
    }

    void close() { glfwSetWindowShouldClose(window, 1); }
    void destroyWindow() {
        glfwDestroyWindow(window);
        window = nullptr;
    }

    void makeCurrent() const { glfwMakeContextCurrent(window); }

    void present() const { glfwSwapBuffers(window); }

    void showWindow(bool show = true) {
        if (show) {
            glfwShowWindow(window);
        } else {
            glfwHideWindow(window);
        }
    }

    void setTitle(const std::string& title) { glfwSetWindowTitle(window, title.c_str()); }

    void setSizeLimits(const vk::Extent2D& minSize, const vk::Extent2D& maxSize = {}) {
        glfwSetWindowSizeLimits(window, minSize.width, minSize.height, (maxSize.width != 0) ? maxSize.width : minSize.width,
                                (maxSize.height != 0) ? maxSize.height : minSize.height);
    }

    bool shouldClose() const { return 0 != glfwWindowShouldClose(window); }

    static void pollEvents() { glfwPollEvents(); }

    operator bool() const { return window != nullptr; }

    void runWindowLoop(const std::function<void()>& frameHandler) {
        while (!shouldClose()) {
            pollEvents();
            if (frameHandler) {
                frameHandler();
            }
        }
    }

private:
    using ResizeHandler = std::function<void(const vk::Extent2D&)>;
    using CloseHandler = std::function<void()>;
    using KeyEventHandler = std::function<void(int, int, int, int)>;
    using KeyPressedHandler = std::function<void(int, int)>;
    using KeyReleasedHandler = std::function<void(int, int)>;
    using MouseEventHandler = std::function<void(int, int, int)>;
    using MousePressedHandler = std::function<void(int, int)>;
    using MouseReleasedHandler = std::function<void(int, int)>;
    using MouseMovedHandler = std::function<void(float, float)>;
    using MouseScrolledHandler = std::function<void(float)>;

    ResizeHandler resizeHandler;
    CloseHandler closeHandler;
    KeyEventHandler keyHandler;
    KeyPressedHandler keyPressHandler;
    KeyReleasedHandler keyReleaseHandler;
    MouseEventHandler mouseHandler;
    MousePressedHandler mousePressHandler;
    MouseReleasedHandler mouseReleaseHandler;
    MouseMovedHandler mouseMoveHandler;
    MouseScrolledHandler mouseScrollHandler;

public:
    void setResizeHandler(const ResizeHandler& handler) { resizeHandler = handler; }
    void setCloseHandler(const CloseHandler& handler) { closeHandler = handler; }
    void setKeyEventHandler(const KeyEventHandler& handler) { keyHandler = handler; }
    void setKeyPressedHandler(const KeyPressedHandler& handler) { keyPressHandler = handler; }
    void setKeyReleasedHandler(const KeyReleasedHandler& handler) { keyReleaseHandler = handler; }
    void setMouseEventHandler(const MouseEventHandler& handler) { mouseHandler = handler; }
    void setMousePressedHandler(const MousePressedHandler& handler) { mousePressHandler = handler; }
    void setMouseReleasedHandler(const MouseReleasedHandler& handler) { mouseReleaseHandler = handler; }
    void setMouseMovedHandler(const MouseMovedHandler& handler) { mouseMoveHandler = handler; }
    void setMouseScrolledHandler(const MouseScrolledHandler& handler) { mouseScrollHandler = handler; }

    //
    // Event handlers are called by the GLFW callback mechanism and should not be called directly
    //

private:
    static void KeyboardHandler(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void MouseButtonHandler(GLFWwindow* window, int button, int action, int mods);
    static void MouseMoveHandler(GLFWwindow* window, double posx, double posy);
    static void MouseScrollHandler(GLFWwindow* window, double xoffset, double yoffset);
    static void WindowCloseHandler(GLFWwindow* window);
    static void FramebufferSizeHandler(GLFWwindow* window, int width, int height);

    GLFWwindow* window{ nullptr };
};
}  // namespace glfw

#endif
