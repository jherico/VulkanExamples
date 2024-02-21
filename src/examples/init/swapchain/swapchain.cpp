#include <common/common.hpp>
#include <examples/glfw.hpp>
#include <rendering/context.hpp>

#if !defined(__ANDROID__)

struct SwapchainImage {
    vk::Image image;
    vk::ImageView view;
    vk::Fence fence;
    vk::ImageSubresourceRange subresourceRange;
};

class Swapchain {
private:
    const vks::Context& context;

public:
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapchain;
    std::vector<SwapchainImage> images;
    vk::Extent2D swapchainExtent;
    vk::Format colorFormat;
    vk::ColorSpaceKHR colorSpace;
    uint32_t imageCount{ 0 };
    uint32_t currentImage{ 0 };

    Swapchain(const vks::Context& context)
        : context(context) {}

    void setWindowSurface(const vk::SurfaceKHR& surface) {
        this->surface = surface;
        // Get list of supported surface formats
        std::vector<vk::SurfaceFormatKHR> surfaceFormats = context.physicalDevice.getSurfaceFormatsKHR(surface);
        auto formatCount = surfaceFormats.size();

        // If the surface format list only includes one entry with  vk::Format::eUndefined,
        // there is no preferered format, so we assume  vk::Format::eB8G8R8A8Unorm
        if ((formatCount == 1) && (surfaceFormats[0].format == vk::Format::eUndefined)) {
            colorFormat = vk::Format::eB8G8R8A8Unorm;
        } else {
            // Always select the first available color format
            // If you need a specific format (e.g. SRGB) you'd need to
            // iterate over the list of available surface format and
            // check for it's presence
            colorFormat = surfaceFormats[0].format;
        }
        colorSpace = surfaceFormats[0].colorSpace;
    }

    // Creates an os specific surface
    // Tries to find a graphics and a present graphicsQueue
    void create(const vk::Extent2D& size, int layers = 1) {
        vk::SwapchainKHR oldSwapchain = swapchain;
        // Get physical device surface properties and formats
        vk::SurfaceCapabilitiesKHR surfCaps = context.physicalDevice.getSurfaceCapabilitiesKHR(surface);
        // Get available present modes
        std::vector<vk::PresentModeKHR> presentModes = context.physicalDevice.getSurfacePresentModesKHR(surface);
        auto presentModeCount = presentModes.size();

        // width and height are either both -1, or both not -1.
        if (surfCaps.currentExtent.width == -1) {
            swapchainExtent = size;
        } else {
            // If the surface size is defined, the swap chain size must match
            swapchainExtent = surfCaps.currentExtent;
        }

        // Prefer mailbox mode if present, it's the lowest latency non-tearing present  mode
        vk::PresentModeKHR swapchainPresentMode = vk::PresentModeKHR::eFifo;
        for (size_t i = 0; i < presentModeCount; i++) {
            if (presentModes[i] == vk::PresentModeKHR::eMailbox) {
                swapchainPresentMode = vk::PresentModeKHR::eMailbox;
                break;
            }
            if ((swapchainPresentMode != vk::PresentModeKHR::eMailbox) && (presentModes[i] == vk::PresentModeKHR::eImmediate)) {
                swapchainPresentMode = vk::PresentModeKHR::eImmediate;
            }
        }

        // Determine the number of images
        uint32_t desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1;
        if ((surfCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfCaps.maxImageCount)) {
            desiredNumberOfSwapchainImages = surfCaps.maxImageCount;
        }

        vk::SurfaceTransformFlagBitsKHR preTransform;
        if (surfCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
            preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
        } else {
            preTransform = surfCaps.currentTransform;
        }

        auto imageFormat = context.physicalDevice.getImageFormatProperties(colorFormat, vk::ImageType::e2D, vk::ImageTiling::eOptimal,
                                                                           vk::ImageUsageFlagBits::eColorAttachment, vk::ImageCreateFlags());

        vk::SwapchainCreateInfoKHR swapchainCI;
        swapchainCI.surface = surface;
        swapchainCI.minImageCount = desiredNumberOfSwapchainImages;
        swapchainCI.imageFormat = colorFormat;
        swapchainCI.imageColorSpace = colorSpace;
        swapchainCI.imageExtent = swapchainExtent;
        swapchainCI.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        swapchainCI.preTransform = preTransform;
        swapchainCI.imageArrayLayers = layers;
        swapchainCI.imageSharingMode = vk::SharingMode::eExclusive;
        swapchainCI.queueFamilyIndexCount = 0;
        swapchainCI.pQueueFamilyIndices = NULL;
        swapchainCI.presentMode = swapchainPresentMode;
        swapchainCI.oldSwapchain = oldSwapchain;
        swapchainCI.clipped = true;
        swapchainCI.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

        swapchain = context.device.createSwapchainKHR(swapchainCI);

        // If an existing sawp chain is re-created, destroy the old swap chain
        // This also cleans up all the presentable images
        if (oldSwapchain) {
            for (uint32_t i = 0; i < imageCount; i++) {
                context.device.destroy(images[i].view);
            }
            context.device.destroy(oldSwapchain);
        }

        vk::ImageViewCreateInfo colorAttachmentView;
        colorAttachmentView.format = colorFormat;
        colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.layerCount = layers;
        colorAttachmentView.viewType = vk::ImageViewType::e2D;

        // Get the swap chain images
        auto swapChainImages = context.device.getSwapchainImagesKHR(swapchain);
        imageCount = (uint32_t)swapChainImages.size();

        // Get the swap chain buffers containing the image and imageview
        images.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; i++) {
            auto& image = images[i];
            image.image = swapChainImages[i];
            colorAttachmentView.image = image.image;
            image.view = context.device.createImageView(colorAttachmentView);
            image.subresourceRange = colorAttachmentView.subresourceRange;
        }
    }

    // Acquires the next image in the swap chain
    uint32_t acquireNextImage(const vk::Semaphore& presentCompleteSemaphore) {
        vk::AcquireNextImageInfoKHR acquireInfo{ swapchain, UINT64_MAX, presentCompleteSemaphore, nullptr, 0x01 };
        auto resultValue = context.device.acquireNextImage2KHR(acquireInfo);
        vk::Result result = resultValue.result;
        if (result != vk::Result::eSuccess) {
            // TODO handle eSuboptimalKHR
            std::cerr << "Invalid acquire result: " << vk::to_string(result);
            throw std::error_code(result);
        }

        currentImage = resultValue.value;
        return currentImage;
    }

    // This function serves two purposes.  The first is to provide a fence associated with a given swap chain
    // image.  If this fence is submitted to a graphicsQueue along with the command buffer(s) that write to that image
    // then if that fence is signaled, you can rely on the fact that those command buffers
    // (and any other per-swapchain-image resoures) are no longer in use.
    //
    // The second purpose is to actually perform a blocking wait on any previous fence that was associated with
    // that image before returning.  By doing so, it can ensure that we do not attempt to submit a command
    // buffer that may already be exeucting for a previous frame using this image.
    //
    // Note that the fence
    const vk::Fence& getSubmitFence() {
        auto& image = images[currentImage];
        if (image.fence) {
            vk::Result fenceResult = vk::Result::eTimeout;
            while (vk::Result::eTimeout == fenceResult) {
                fenceResult = context.device.waitForFences(image.fence, VK_TRUE, UINT64_MAX);
            }
            context.device.resetFences(image.fence);
        } else {
            image.fence = context.device.createFence(vk::FenceCreateFlags());
        }
        return image.fence;
    }

    // Free all Vulkan resources used by the swap chain
    void cleanup() {
        for (uint32_t i = 0; i < imageCount; i++) {
            auto& image = images[i];
            if (image.fence) {
                auto result = context.device.waitForFences(image.fence, VK_TRUE, UINT64_MAX);
                assert(result != vk::Result::eTimeout);
                context.device.destroy(image.fence);
            }
            context.device.destroy(image.view);
            // Note, we do not destroy the vk::Image itself  because we are not responsible for it. It is
            // owned by the underlying swap chain implementation and will be handled by destroySwapchainKHR
        }
        images.clear();
        context.device.destroy(swapchain);
        context.instance.destroySurfaceKHR(surface);
    }
};

class SwapChainExample {
    vks::Context& context{ vks::Context::get() };
    glfw::Window window;
    Swapchain swapchain{ context };
    vk::Extent2D windowSize;
    vk::SurfaceKHR surface;
    vks::QueueManager queue;

    // List of command buffers (same as number of swap chain images)
    std::vector<vk::CommandBuffer> commandBuffers;

    // Syncronization primitices
    struct {
        vk::SemaphoreSubmitInfo acquireComplete;
        vk::SemaphoreSubmitInfo renderComplete;
    } semaphores;

public:
    SwapChainExample() = default;

    void createWindow() {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto monitor = glfwGetPrimaryMonitor();
        auto mode = glfwGetVideoMode(monitor);
        windowSize = vk::Extent2D{ static_cast<uint32_t>(mode->width / 2), static_cast<uint32_t>(mode->height / 2) };
        window.setResizeHandler([this](const vk::Extent2D& extent) { onWindowResized(extent); });
        window.createWindow(windowSize, { 100, 100 });
        window.showWindow();
        surface = window.createSurface(context.instance);
    }

    void createCommandBuffers() {
        // Allocate command buffers, 1 for each swap chain image
        if (commandBuffers.empty()) {
            commandBuffers = queue.allocateCommandBuffers(swapchain.imageCount);
        }

        static const std::vector<vk::ClearColorValue> CLEAR_COLORS{
            vks::util::clearColor({ 1, 0, 0, 0 }), vks::util::clearColor({ 0, 1, 0, 0 }), vks::util::clearColor({ 0, 0, 1, 0 }),
            vks::util::clearColor({ 0, 1, 1, 0 }), vks::util::clearColor({ 1, 0, 1, 0 }), vks::util::clearColor({ 1, 1, 0, 0 }),
            vks::util::clearColor({ 1, 1, 1, 0 }),
        };

        vk::RenderingInfo renderingInfo;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.renderArea.extent = windowSize;

        vk::RenderingAttachmentInfo attachmentInfo;
        attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
        attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
        renderingInfo.pColorAttachments = &attachmentInfo;

        using vks::util::ImageTransitionState;
        vk::ImageSubresourceRange range{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        // The present layout will be used to present the color attachment to the swap chain.  There's no destination access mask or stage because
        // there's no corresponding pipeline operation.

        for (size_t i = 0; i < swapchain.imageCount; ++i) {
            const auto& swapchainImage = swapchain.images[i];
            const auto& commandBuffer = commandBuffers[i];
            commandBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            // Move the image into a state suitable for rendering.  The prior state is irrelevant.
            vks::util::setImageLayout(commandBuffer, swapchainImage.image, range, ImageTransitionState::UNDEFINED, ImageTransitionState::COLOR_ATTACHMENT);
            // begin rendering with the specified clear color and swapchain image
            attachmentInfo.clearValue.color = CLEAR_COLORS[i % CLEAR_COLORS.size()];
            attachmentInfo.imageView = swapchainImage.view;
            commandBuffer.beginRendering(renderingInfo);
            commandBuffer.endRendering();
            // Move the image from a rendering state into a presentation state.
            vks::util::setImageLayout(commandBuffer, swapchainImage.image, range, ImageTransitionState::COLOR_ATTACHMENT, ImageTransitionState::PRESENT);
            commandBuffer.end();
        }
    }

    void createSwapchain() {
        // Using the window surface, construct the swap chain.  The swap chain is dependent on both
        // the Vulkan instance as well as the window surface, so it needs to happen after
        swapchain.setWindowSurface(surface);
        swapchain.create(windowSize);
        // Create the CommandBuffer objects which will contain the commands we execute for our rendering.
        createCommandBuffers();
    }

    void onWindowResized(const vk::Extent2D& newSize) {
        queue.handle.waitIdle();
        context.device.waitIdle();
        windowSize = newSize;
        createSwapchain();
    }

    void prepare() {
        glfw::Window::init();
        // Construct the Vulkan instance just as we did in the init_context example
        context.setValidationEnabled(true);
        context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
        context.createInstance();

        // Construct the window.  The window doesn't need any special attributes, it just
        // need to be a native Win32 or XCB window surface. Window is independent of the contenxt and
        // RenderPass creation.  It can creation can occur before or after them.
        createWindow();

        context.requireDeviceExtensions({ VK_KHR_SWAPCHAIN_EXTENSION_NAME });
        context.pickDevice(surface);
        // Enable the use of required features
        context.enabledFeatures.core13.dynamicRendering = VK_TRUE;
        context.enabledFeatures.core13.synchronization2 = VK_TRUE;
        context.enabledFeatures.core13.maintenance4 = VK_TRUE;
        context.createDevice();

        // Finally, we need to create a number of Sempahores.  Semaphores are used for GPU<->GPU
        // synchronization.  Tyipically this means that you include them in certain function calls to
        // tell the GPU to wait until the semaphore is signalled before actually executing the commands
        // or that once it's completed the commands, it should signal the semaphore, or both.

        // Create a semaphore used to synchronize image presentation
        // This semaphore will be signaled when the system actually displays an image.  By waiting on this
        // semaphore, we can ensure that the GPU doesn't start working on the next frame until the image
        // for it has been acquired (typically meaning that it's previous contents have been presented to the screen)
        semaphores.acquireComplete = context.device.createSemaphore({});
        // Create a semaphore used to synchronize command submission
        // This semaphore is used to ensure that before we submit a given image for presentation, all the rendering
        // command for generating the image have been completed.
        semaphores.renderComplete = context.device.createSemaphore({});

        queue = vks::QueueManager{ context.device, context.queuesInfo.findQueue(context.physicalDevice, vk::QueueFlagBits::eGraphics, surface) };

        // Construct the swap chain and the associated framebuffers and command buffers
        createSwapchain();
    }

    void renderFrame() {
        // Acquire the next image from the swap chain.
        uint32_t currentBuffer = swapchain.acquireNextImage(semaphores.acquireComplete.semaphore);

        // We request a fence from the swap chain.  The swap chain code will
        // block on this fence until its operations are complete, guaranteeing
        // we don't run concurrent operations that are trying to write to a
        // given swap chain image
        vk::Fence submitFence = swapchain.getSubmitFence();

        // This is a helper function for submitting commands to the graphics graphicsQueue
        //
        // The first parameter is a command buffer or buffers to be executed.
        //
        // The second parameter is a set of wait semaphores and pipeline stages.
        //  Before the commands will execute, these semaphores must have reached the
        // specified stages.

        // The third paramater is a semaphore or semaphore array that will be signalled
        // as the command are processed through the pipeline.
        //
        // Finally, the submit fence is another synchornization primitive that will be signaled
        // when the commands have been fully completed, but the fence, unlike the semaphores,
        // can be queried by the client (us) to determine when it's signaled.
        queue.submit2(commandBuffers[currentBuffer], semaphores.acquireComplete, semaphores.renderComplete, submitFence);

        // Once the image has been written, the swap chain
        auto result = queue.present(swapchain.swapchain, swapchain.currentImage, semaphores.renderComplete.semaphore);
        if (result != vk::Result::eSuccess) {
            std::cout << vk::to_string(result) << std::endl;
        }
    }

    void cleanup() {
        queue.handle.waitIdle();
        context.device.waitIdle();
        context.device.destroy(semaphores.acquireComplete.semaphore);
        context.device.destroy(semaphores.renderComplete.semaphore);
        swapchain.cleanup();
        window.destroyWindow();
        context.destroy();
    }

    void run() {
        prepare();

        window.runWindowLoop([this] {
            renderFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        });

        cleanup();
    }
};
#else
class SwapChainExample {
public:
    void run(){};
};
#endif

RUN_EXAMPLE(SwapChainExample)
