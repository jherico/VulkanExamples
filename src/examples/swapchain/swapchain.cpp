#include <common.hpp>
#include <vks/context.hpp>

#if !defined(__ANDROID__)

struct SwapchainImage {
    vk::Image image;
    vk::ImageView view;
    vk::Fence fence;
};

class SwapChain {
private:
    const vks::Context& context;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapchain;
    std::vector<SwapchainImage> images;
    vk::PresentInfoKHR presentInfo;

public:
    vk::Extent2D swapchainExtent;
    vk::Format colorFormat;
    vk::ColorSpaceKHR colorSpace;
    uint32_t imageCount{ 0 };
    uint32_t currentImage{ 0 };
    // Index of the deteced graphics and presenting device queue
    uint32_t queueNodeIndex = UINT32_MAX;

    SwapChain(const vks::Context& context)
        : context(context) {
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &currentImage;
    }

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

        // Find a queue for both present and graphics
        queueNodeIndex = context.findQueue(vk::QueueFlagBits::eGraphics, surface);
    }

    // Creates an os specific surface
    // Tries to find a graphics and a present queue
    void create(const glm::uvec2& size) {
        vk::SwapchainKHR oldSwapchain = swapchain;
        // Get physical device surface properties and formats
        vk::SurfaceCapabilitiesKHR surfCaps = context.physicalDevice.getSurfaceCapabilitiesKHR(surface);
        // Get available present modes
        std::vector<vk::PresentModeKHR> presentModes = context.physicalDevice.getSurfacePresentModesKHR(surface);
        auto presentModeCount = presentModes.size();

        // width and height are either both -1, or both not -1.
        if (surfCaps.currentExtent.width == -1) {
            swapchainExtent.width = size.x;
            swapchainExtent.height = size.y;
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
        swapchainCI.imageArrayLayers = 1;
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
                context.device.destroyImageView(images[i].view);
            }
            context.device.destroySwapchainKHR(oldSwapchain);
        }

        vk::ImageViewCreateInfo colorAttachmentView;
        colorAttachmentView.format = colorFormat;
        colorAttachmentView.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        colorAttachmentView.subresourceRange.levelCount = 1;
        colorAttachmentView.subresourceRange.layerCount = 1;
        colorAttachmentView.viewType = vk::ImageViewType::e2D;

        // Get the swap chain images
        auto swapChainImages = context.device.getSwapchainImagesKHR(swapchain);
        imageCount = (uint32_t)swapChainImages.size();

        // Get the swap chain buffers containing the image and imageview
        images.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; i++) {
            images[i].image = swapChainImages[i];
            colorAttachmentView.image = swapChainImages[i];
            images[i].view = context.device.createImageView(colorAttachmentView);
        }
    }

    std::vector<vk::Framebuffer> createFramebuffers(vk::FramebufferCreateInfo framebufferCreateInfo) {
        // Verify that the first attachment is null
        assert(framebufferCreateInfo.pAttachments[0] == vk::ImageView());

        std::vector<vk::ImageView> attachments;
        attachments.resize(framebufferCreateInfo.attachmentCount);
        for (size_t i = 0; i < framebufferCreateInfo.attachmentCount; ++i) {
            attachments[i] = framebufferCreateInfo.pAttachments[i];
        }
        framebufferCreateInfo.pAttachments = attachments.data();

        std::vector<vk::Framebuffer> framebuffers;
        framebuffers.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; i++) {
            attachments[0] = images[i].view;
            framebuffers[i] = context.device.createFramebuffer(framebufferCreateInfo);
        }
        return framebuffers;
    }

    // Acquires the next image in the swap chain
    uint32_t acquireNextImage(const vk::Semaphore& presentCompleteSemaphore) {
        auto resultValue = context.device.acquireNextImageKHR(swapchain, UINT64_MAX, presentCompleteSemaphore, vk::Fence());
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
    // image.  If this fence is submitted to a queue along with the command buffer(s) that write to that image
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

    // Present the current image to the queue
    vk::Result queuePresent(const vk::Semaphore& waitSemaphore) {
        presentInfo.waitSemaphoreCount = waitSemaphore ? 1 : 0;
        presentInfo.pWaitSemaphores = &waitSemaphore;
        return context.queue.presentKHR(presentInfo);
    }

    // Free all Vulkan resources used by the swap chain
    void cleanup() {
        for (uint32_t i = 0; i < imageCount; i++) {
            auto& image = images[i];
            if (image.fence) {
                context.device.waitForFences(image.fence, VK_TRUE, UINT64_MAX);
                context.device.destroyFence(image.fence);
            }
            context.device.destroyImageView(image.view);
            // Note, we do not destroy the vk::Image itself  because we are not responsible for it. It is
            // owned by the underlying swap chain implementation and will be handled by destroySwapchainKHR
        }
        images.clear();
        context.device.destroySwapchainKHR(swapchain);
        context.instance.destroySurfaceKHR(surface);
    }
};

class SwapChainExample : glfw::Window {
    vks::Context context;
    SwapChain swapchain{ context };
    vk::RenderPass renderPass;
    glm::uvec2 windowSize;
    vk::SurfaceKHR surface;

    // List of available frame buffers (same as number of swap chain images)
    std::vector<vk::Framebuffer> framebuffers;

    // List of command buffers (same as number of swap chain images)
    std::vector<vk::CommandBuffer> commandBuffers;

    // Syncronization primitices
    struct {
        vk::Semaphore acquireComplete;
        vk::Semaphore renderComplete;
    } semaphores;

public:
    void createWindow() {
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto monitor = glfwGetPrimaryMonitor();
        auto mode = glfwGetVideoMode(monitor);
        windowSize = glm::uvec2{ mode->width / 2, mode->height / 2 };
        glfw::Window::createWindow(windowSize, { 100, 100 });
        showWindow();
        surface = createSurface(context.instance);
    }

    void createRenderPass() {
        std::vector<vk::AttachmentDescription> attachments;
        std::vector<vk::AttachmentReference> attachmentReferences;
        std::vector<vk::SubpassDescription> subpasses;
        std::vector<vk::SubpassDependency> subpassDependencies;

        vk::AttachmentDescription colorAttachment;
        colorAttachment.format = swapchain.colorFormat;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;
        attachments.push_back(colorAttachment);

        vk::AttachmentReference colorAttachmentReference;
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout = vk::ImageLayout::eColorAttachmentOptimal;
        attachmentReferences.push_back(colorAttachmentReference);

        vk::SubpassDescription subpass;
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = attachmentReferences.data();
        subpasses.push_back(subpass);

        vk::SubpassDependency dependency;
        dependency.srcSubpass = 0;
        dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

        dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
        subpassDependencies.push_back(dependency);

        vk::RenderPassCreateInfo renderPassInfo;
        renderPassInfo.attachmentCount = (uint32_t)attachments.size();
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = (uint32_t)subpasses.size();
        renderPassInfo.pSubpasses = subpasses.data();
        renderPassInfo.dependencyCount = (uint32_t)subpassDependencies.size();
        renderPassInfo.pDependencies = subpassDependencies.data();
        renderPass = context.device.createRenderPass(renderPassInfo);
    }

    void createFramebuffers() {
        std::array<vk::ImageView, 1> imageViews;
        vk::FramebufferCreateInfo framebufferCreateInfo;
        framebufferCreateInfo.renderPass = renderPass;
        // Create a placeholder image view for the swap chain color attachments
        framebufferCreateInfo.attachmentCount = (uint32_t)imageViews.size();
        framebufferCreateInfo.pAttachments = imageViews.data();
        framebufferCreateInfo.width = swapchain.swapchainExtent.width;
        framebufferCreateInfo.height = swapchain.swapchainExtent.height;
        framebufferCreateInfo.layers = 1;

        // Create frame buffers for every swap chain image
        framebuffers = swapchain.createFramebuffers(framebufferCreateInfo);
    }

    void createCommandBuffers() {
        // Allocate command buffers, 1 for each swap chain image
        if (commandBuffers.empty()) {
            commandBuffers = context.allocateCommandBuffers(swapchain.imageCount);
        }

        static const std::vector<vk::ClearColorValue> CLEAR_COLORS{
            vks::util::clearColor({ 1, 0, 0, 0 }), vks::util::clearColor({ 0, 1, 0, 0 }), vks::util::clearColor({ 0, 0, 1, 0 }),
            vks::util::clearColor({ 0, 1, 1, 0 }), vks::util::clearColor({ 1, 0, 1, 0 }), vks::util::clearColor({ 1, 1, 0, 0 }),
            vks::util::clearColor({ 1, 1, 1, 0 }),
        };

        vk::ClearValue clearValue;
        vk::RenderPassBeginInfo renderPassBeginInfo{
            renderPass,
            {},  // no framebuffer explicitly set
            { {}, swapchain.swapchainExtent },
            1,           // number of clear values
            &clearValue  // clear value
        };

        for (size_t i = 0; i < swapchain.imageCount; ++i) {
            const auto& commandBuffer = commandBuffers[i];
            clearValue.color = CLEAR_COLORS[i % CLEAR_COLORS.size()];
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = framebuffers[i];
            commandBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            commandBuffer.begin(vk::CommandBufferBeginInfo{});
            commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
            commandBuffer.endRenderPass();
            commandBuffer.end();
        }
    }

    void createSwapchain() {
        // Using the window surface, construct the swap chain.  The swap chain is dependent on both
        // the Vulkan instance as well as the window surface, so it needs to happen after
        swapchain.setWindowSurface(surface);

        swapchain.create(windowSize);

        if (!renderPass) {
            // Create a renderpass.
            //
            // A renderpass defines what combination of input and output attachments types will be used
            // during a given set of rendering operations, as well as what subpasses
            //
            // Note, it doesn't reference the actual images, just defines the kinds of images, they're
            // layouts, and how the layouts will change over the course of executing commands during the
            // renderpass.  Therefore it can be created almost immediately after the context and typically
            // doesn't need to change over time in response to things like window resizing , or rendering a
            // different set of objects, or using different pipelines for rendering.
            //
            // A RenderPass is required for creating framebuffers and pipelines, which can then only be used
            // with that specific RenderPass OR another RenderPass that is considered compatible.
            //
            // Creation of the RenderPass is dependent on the Vulkan context creation, and in this case on the
            // swap chain because we're using the swap chain images directly in the framebuffer
            createRenderPass();
        }

        // Create the Framebuffers to which we will render output that will be presented to the screen.
        // As noted above, any FrameBuffer is dependent on a RenderPass and can only be used with that
        // RenderPass or another RenderPass compatible with it.  It's also typically dpenedent on the
        // Window, since usually you'll be creating at least one set of Framebuffers specifically for
        // presentation to the window surface, and that set (which we are creating here) must must be using
        // the images acquired from the SwapChain, and must match the size of those images.
        //
        // Common practice is to create an individual Framebuffer for each of the SwapChain images,
        // although all of them can typically share the same depth image, since they will not be
        // in use concurrently
        createFramebuffers();

        // Create the CommandBuffer objects which will contain the commands we execute for our rendering.
        //
        // Similar to the Framebuffers, we will create one for each of the swap chain images.
        createCommandBuffers();
    }

    void onWindowResized(const glm::uvec2& newSize) override {
        context.queue.waitIdle();
        context.device.waitIdle();
        for (const auto& framebuffer : framebuffers) {
            context.device.destroyFramebuffer(framebuffer);
        }
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
        context.createDevice(surface);

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

        // Construct the swap chain and the associated framebuffers and command buffers
        createSwapchain();
    }

    void renderFrame() {
        // Acquire the next image from the swap chain.
        uint32_t currentBuffer = swapchain.acquireNextImage(semaphores.acquireComplete);

        // We request a fence from the swap chain.  The swap chain code will
        // block on this fence until its operations are complete, guaranteeing
        // we don't run concurrent operations that are trying to write to a
        // given swap chain image
        vk::Fence submitFence = swapchain.getSubmitFence();

        // This is a helper function for submitting commands to the graphics queue
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
        context.submit(commandBuffers[currentBuffer], { semaphores.acquireComplete, vk::PipelineStageFlagBits::eBottomOfPipe }, semaphores.renderComplete,
                       submitFence);

        // Once the image has been written, the swap chain
        swapchain.queuePresent(semaphores.renderComplete);
    }

    void cleanup() {
        context.queue.waitIdle();
        context.device.waitIdle();
        context.device.destroySemaphore(semaphores.acquireComplete);
        context.device.destroySemaphore(semaphores.renderComplete);
        for (const auto& framebuffer : framebuffers) {
            context.device.destroyFramebuffer(framebuffer);
        }
        context.device.destroyRenderPass(renderPass);
        swapchain.cleanup();
        destroyWindow();
        context.destroy();
    }

    void run() {
        prepare();

        runWindowLoop([this] {
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
