#include "glfw.hpp"
#include <vulkan/vulkan.hpp>
#if !defined(ANDROID)
#include <mutex>

namespace glfw {

bool Window::init() {
    return GLFW_TRUE == glfwInit();
}

void Window::terminate() {
    glfwTerminate();
}

#if defined(VULKAN_HPP)
std::vector<std::string> Window::getRequiredInstanceExtensions() {
    std::vector<std::string> result;
    uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    if (names && count) {
        for (uint32_t i = 0; i < count; ++i) {
            result.emplace_back(names[i]);
        }
    }
    return result;
}

vk::SurfaceKHR Window::createWindowSurface(GLFWwindow* window, const vk::Instance& instance, const vk::AllocationCallbacks* pAllocator) {
    VkSurfaceKHR rawSurface;
    vk::Result result =
        static_cast<vk::Result>(glfwCreateWindowSurface((VkInstance)instance, window, reinterpret_cast<const VkAllocationCallbacks*>(pAllocator), &rawSurface));
    vk::resultCheck(result, "vk::CommandBuffer::begin");
    return rawSurface;
}
#endif

void Window::KeyboardHandler(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    if (example->keyHandler) {
        example->keyHandler(key, scancode, action, mods);
    }
}

void Window::MouseButtonHandler(GLFWwindow* window, int button, int action, int mods) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    if (example->mouseHandler) {
        example->mouseHandler(button, action, mods);
    }
}

void Window::MouseMoveHandler(GLFWwindow* window, double posx, double posy) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    if (example->mouseMoveHandler) {
        example->mouseMoveHandler((float)posx, (float)posy);
    }
}

void Window::MouseScrollHandler(GLFWwindow* window, double xoffset, double yoffset) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    if (example->mouseScrollHandler && yoffset != 0.0) {
        example->mouseScrollHandler((float)yoffset);
    }
}

void Window::WindowCloseHandler(GLFWwindow* window) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    if (example->closeHandler) {
        example->closeHandler();
    }
}

void Window::FramebufferSizeHandler(GLFWwindow* window, int width, int height) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    if (example->resizeHandler) {
        example->resizeHandler({ (uint32_t)width, (uint32_t)height });
    }
}

Window::Window() {
    mouseHandler = [&](int button, int action, int mods) {
        switch (action) {
            case GLFW_PRESS:
                if (mousePressHandler) {
                    mousePressHandler(button, mods);
                }
                break;

            case GLFW_RELEASE:
                if (mouseReleaseHandler) {
                    mouseReleaseHandler(button, mods);
                }
                break;

            default:
                break;
        }
    };
    keyHandler = [&](int key, int scancode, int action, int mods) {
        switch (action) {
            case GLFW_PRESS:
                if (keyPressHandler) {
                    keyPressHandler(key, mods);
                }
                break;

            case GLFW_RELEASE:
                if (keyReleaseHandler) {
                    keyReleaseHandler(key, mods);
                }
                break;

            default:
                break;
        }
    };
}

}  // namespace glfw
#endif
