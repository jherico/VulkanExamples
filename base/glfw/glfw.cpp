#include "glfw.hpp"
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
    return {vk::ResultValue<VkSurfaceKHR>(result, rawSurface).value};
}
#endif

void Window::KeyboardHandler(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    example->onKeyEvent(key, scancode, action, mods);
}

void Window::MouseButtonHandler(GLFWwindow* window, int button, int action, int mods) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    example->onMouseButtonEvent(button, action, mods);
}

void Window::MouseMoveHandler(GLFWwindow* window, double posx, double posy) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    example->onMouseMoved(glm::vec2(posx, posy));
}

void Window::MouseScrollHandler(GLFWwindow* window, double xoffset, double yoffset) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    example->onMouseScrolled((float)yoffset);
}

void Window::CloseHandler(GLFWwindow* window) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    example->onWindowClosed();
}

void Window::FramebufferSizeHandler(GLFWwindow* window, int width, int height) {
    Window* example = (Window*)glfwGetWindowUserPointer(window);
    example->onWindowResized(glm::uvec2(width, height));
}
}  // namespace glfw
#endif
