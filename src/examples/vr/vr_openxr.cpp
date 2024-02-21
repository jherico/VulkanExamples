#include <common/common.hpp>
#if 0
#include <examples/glfw.hpp>
#include <openxr/openxr.hpp>
#include <rendering/context.hpp>
#include <rendering/shapesRenderer.hpp>
#include <rendering/swapchain.hpp>


// FIXME
class VrExample {
public:
    glfw::Window window;
    vks::Context context;
    vks::Swapchain swapchain;
    vk::SurfaceKHR surface;
    std::shared_ptr<vkx::ShapesRenderer> shapesRenderer{ std::make_shared<vkx::ShapesRenderer>(context, true) };
    double fpsTimer{ 0 };
    float lastFPS{ 0 };
    uint32_t frameCounter{ 0 };
    vk::Extent2D size{ 1280, 720 };
    vk::Extent2D renderTargetSize;
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
        size.width /= 4;
        size.height /= 4;

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window.setKeyPressedHandler([&](int key, int mods) { onKeyPressed(key, mods); });
        //window.setKeyReleasedHandler([&](int key, int mods) { keyReleased(key); });
        //window.setMouseMovedHandler([&](float x, float y) { mouseMoved({ x, y }); });
        //window.setMouseScrolledHandler([&](float scrolly) { mouseScrolled(scrolly); });
        //window.setResizeHandler([&](const vk::Extent2D& size) { windowResize(size); });
        window.setCloseHandler([&]() { window.close(); });

        window.createWindow(size, "VR Example");
        context.requireExtensions(glfw::Window::getRequiredInstanceExtensions());
    }

    void prepareVulkan() {
        context.createInstance();
        surface = window.createSurface(context.instance);
        context.pickDevice(surface);
        context.createDevice();
    }

    void prepareSwapchain() {
        swapchain.setup(context.physicalDevice, context.device, nullptr, context.queuesInfo.graphics.index, surface);
        swapchain.create(size);
    }

    void prepareRenderer() {
        shapesRenderer->framebufferSize = renderTargetSize;
        shapesRenderer->colorFormats = { vk::Format::eR8G8B8A8Srgb };
        shapesRenderer->prepare();
    }

    virtual void recenter() = 0;


    void onKeyPressed(int key, int mods) {
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
        window.runWindowLoop([&] {
            auto tEnd = std::chrono::high_resolution_clock::now();
            auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            update((float)tDiff / 1000.0f);
            render();
            fpsTimer += (float)tDiff;
            if (fpsTimer > 1000.0f) {
                window.setTitle(getWindowTitle());
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


namespace openvr {
template <typename F>
void for_each_eye(F f) {
    f(vr::Hmd_Eye::Eye_Left);
    f(vr::Hmd_Eye::Eye_Right);
}

inline mat4 toGlm(const vr::HmdMatrix44_t& m) {
    return glm::transpose(glm::make_mat4(&m.m[0][0]));
}

inline vec3 toGlm(const vr::HmdVector3_t& v) {
    return vec3(v.v[0], v.v[1], v.v[2]);
}

inline mat4 toGlm(const vr::HmdMatrix34_t& m) {
    mat4 result = mat4(m.m[0][0], m.m[1][0], m.m[2][0], 0.0, m.m[0][1], m.m[1][1], m.m[2][1], 0.0, m.m[0][2], m.m[1][2], m.m[2][2], 0.0, m.m[0][3], m.m[1][3],
                       m.m[2][3], 1.0f);
    return result;
}

inline vr::HmdMatrix34_t toOpenVr(const mat4& m) {
    vr::HmdMatrix34_t result;
    for (uint8_t i = 0; i < 3; ++i) {
        for (uint8_t j = 0; j < 4; ++j) {
            result.m[i][j] = m[j][i];
        }
    }
    return result;
}

std::vector<std::string> toStringVec(const std::vector<char>& data) {
    std::vector<std::string> result;
    std::string buffer;
    for (char c : data) {
        if (c == 0 || c == ' ') {
            if (!buffer.empty()) {
                result.push_back(buffer);
                buffer.clear();
            }
            if (c == 0) {
                break;
            }
        } else {
            buffer += c;
        }
    }
    return result;
}

std::set<std::string> toStringSet(const std::vector<char>& data) {
    std::set<std::string> result;
    std::string buffer;
    for (char c : data) {
        if (c == 0 || c == ' ') {
            if (!buffer.empty()) {
                result.insert(buffer);
                buffer.clear();
            }
            if (c == 0) {
                break;
            }
        } else {
            buffer += c;
        }
    }

    return result;
}

std::vector<std::string> getInstanceExtensionsRequired(vr::IVRCompositor* compositor) {
    auto bytesRequired = compositor->GetVulkanInstanceExtensionsRequired(nullptr, 0);
    std::vector<char> extensions;
    extensions.resize(bytesRequired);
    compositor->GetVulkanInstanceExtensionsRequired(extensions.data(), (uint32_t)extensions.size());
    return toStringVec(extensions);
}

std::set<std::string> getDeviceExtensionsRequired(const vk::PhysicalDevice& physicalDevice, vr::IVRCompositor* compositor) {
    auto bytesRequired = compositor->GetVulkanDeviceExtensionsRequired(physicalDevice, nullptr, 0);
    std::vector<char> extensions;
    extensions.resize(bytesRequired);
    compositor->GetVulkanDeviceExtensionsRequired(physicalDevice, extensions.data(), (uint32_t)extensions.size());
    return toStringSet(extensions);
}
}  // namespace openvr

// Allow a maximum of two outstanding presentation operations.
#define FRAME_LAG 2

class OpenVrExample : public VrExample {
    using Parent = VrExample;

public:
    std::array<glm::mat4, 2> eyeOffsets;
    vr::IVRSystem* vrSystem{ nullptr };
    vr::IVRCompositor* vrCompositor{ nullptr };

    size_t frameIndex{ 0 };

    using EyeImages = std::array<vks::Image, 2>;
    using StagingImages = std::array<EyeImages, FRAME_LAG>;
    StagingImages stagingImages;
    std::array<std::vector<vk::CommandBuffer>, FRAME_LAG> stagingBlitCommands;
    std::array<vk::Semaphore, FRAME_LAG> stagingBlitCompletes;
    std::array<vk::Fence, FRAME_LAG> frameFences;

    vk::Semaphore mirrorBlitComplete;
    std::vector<vk::CommandBuffer> mirrorBlitCommands;

    ~OpenVrExample() {
        vrSystem = nullptr;
        vrCompositor = nullptr;
        vr::VR_Shutdown();
    }

    void recenter() override { vrSystem->ResetSeatedZeroPose(); }

    void prepareOpenVr() {
        vr::EVRInitError eError;
        vrSystem = vr::VR_Init(&eError, vr::VRApplication_Scene);
        vrSystem->GetRecommendedRenderTargetSize(&renderTargetSize.x, &renderTargetSize.y);
        vrCompositor = vr::VRCompositor();

        context.requireExtensions(openvr::getInstanceExtensionsRequired(vrCompositor));

        // Recommended render target size is per-eye, so double the X size for
        // left + right eyes
        renderTargetSize.x *= 2;

        openvr::for_each_eye([&](vr::Hmd_Eye eye) {
            eyeOffsets[eye] = openvr::toGlm(vrSystem->GetEyeToHeadTransform(eye));
            eyeProjections[eye] = openvr::toGlm(vrSystem->GetProjectionMatrix(eye, 0.1f, 256.0f));
            // FIXME Strange distortion and inverted Z view when doing this, but correct head tracking
            //eyeProjections[eye][1][1] *= -1.0f;
        });

        context.setDeviceExtensionsPicker([this](const vk::PhysicalDevice& physicalDevice) -> std::set<std::string> {
            return openvr::getDeviceExtensionsRequired(physicalDevice, vrCompositor);
        });
    }

    void prepareMirrorBlit() {
        mirrorBlitComplete = context.device.createSemaphore({});

        if (mirrorBlitCommands.empty()) {
            vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
            cmdBufAllocateInfo.commandPool = context.getCommandPool();
            cmdBufAllocateInfo.commandBufferCount = swapchain.imageCount;
            mirrorBlitCommands = context.device.allocateCommandBuffers(cmdBufAllocateInfo);
        }

        vk::ImageBlit mirrorBlit;
        mirrorBlit.dstSubresource.aspectMask = mirrorBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        mirrorBlit.dstSubresource.layerCount = mirrorBlit.srcSubresource.layerCount = 1;
        mirrorBlit.srcOffsets[1] = vk::Offset3D{ (int32_t)renderTargetSize.x, (int32_t)renderTargetSize.y, 1 };
        mirrorBlit.dstOffsets[1] = vk::Offset3D{ (int32_t)size.x, (int32_t)size.y, 1 };

        for (size_t i = 0; i < swapchain.imageCount; ++i) {
            vk::CommandBuffer& cmdBuffer = mirrorBlitCommands[i];
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(vk::CommandBufferBeginInfo{});
            context.setImageLayout(cmdBuffer, swapchain.images[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eTransferDstOptimal);
            cmdBuffer.blitImage(shapesRenderer->framebuffer.colors[0].image, vk::ImageLayout::eTransferSrcOptimal, swapchain.images[i].image,
                                vk::ImageLayout::eTransferDstOptimal, mirrorBlit, vk::Filter::eNearest);
            context.setImageLayout(cmdBuffer, swapchain.images[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::ePresentSrcKHR);
            cmdBuffer.end();
        }
    }

    void prepareStagingImages() {
        vk::ImageCreateInfo imageCreate;
        imageCreate.imageType = vk::ImageType::e2D;
        imageCreate.extent = vk::Extent3D{ renderTargetSize.x / 2, renderTargetSize.y, 1 };
        imageCreate.format = vk::Format::eR8G8B8A8Srgb;
        imageCreate.mipLevels = 1;
        imageCreate.arrayLayers = 1;
        imageCreate.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled;

        for (size_t frame = 0; frame < FRAME_LAG; ++frame) {
            for (size_t eye = 0; eye < 2; ++eye) {
                stagingImages[frame][eye] = context.createImage(imageCreate);
            }
        }
    }

    void prepareStagingBlit() {
        for (size_t frame = 0; frame < FRAME_LAG; ++frame) {
            stagingBlitCompletes[frame] = context.device.createSemaphore({});

            auto& blitCommands = stagingBlitCommands[frame];
            if (blitCommands.empty()) {
                vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
                cmdBufAllocateInfo.commandPool = context.getCommandPool();
                cmdBufAllocateInfo.commandBufferCount = 2;
                blitCommands = context.device.allocateCommandBuffers(cmdBufAllocateInfo);
            }

            for (size_t eye = 0; eye < 2; ++eye) {
                vk::ImageBlit blit;
                blit.dstSubresource.aspectMask = blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
                blit.dstSubresource.layerCount = blit.srcSubresource.layerCount = 1;
                blit.srcOffsets[1] = vk::Offset3D{ (int32_t)renderTargetSize.x / 2, (int32_t)renderTargetSize.y, 1 };
                blit.dstOffsets[1] = vk::Offset3D{ (int32_t)renderTargetSize.x / 2, (int32_t)renderTargetSize.y, 1 };
                // Offset the source image for the right eye
                if (eye == vr::Eye_Right) {
                    blit.srcOffsets[0].x = (int32_t)renderTargetSize.x / 2;
                    blit.srcOffsets[1].x += (int32_t)renderTargetSize.x / 2;
                }
                vk::CommandBuffer& cmdBuffer = blitCommands[eye];
                cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
                cmdBuffer.begin(vk::CommandBufferBeginInfo{});
                const auto& stagingImage = stagingImages[frame][eye];
                context.setImageLayout(cmdBuffer, stagingImage.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                                       vk::ImageLayout::eTransferDstOptimal);
                cmdBuffer.blitImage(shapesRenderer->framebuffer.colors[0].image, vk::ImageLayout::eTransferSrcOptimal, stagingImage.image,
                                    vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
                context.setImageLayout(cmdBuffer, stagingImage.image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
                                       vk::ImageLayout::eGeneral);
                cmdBuffer.end();
            }
        }
    }

    void prepareOpenVrVk() {
        prepareStagingImages();
        prepareStagingBlit();
        prepareMirrorBlit();
        for (size_t frame = 0; frame < FRAME_LAG; ++frame) {
            frameFences[frame] = context.device.createFence({ vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void prepare() {
        prepareOpenVr();
        Parent::prepare();
        prepareOpenVrVk();
    }

    void update(float delta) {
        vr::TrackedDevicePose_t currentTrackedDevicePose[vr::k_unMaxTrackedDeviceCount];
        vrCompositor->WaitGetPoses(currentTrackedDevicePose, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
        vr::TrackedDevicePose_t _trackedDevicePose[vr::k_unMaxTrackedDeviceCount];
        float displayFrequency = vrSystem->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float);
        float frameDuration = 1.f / displayFrequency;
        float vsyncToPhotons = vrSystem->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_SecondsFromVsyncToPhotons_Float);
        float predictedDisplayTime = frameDuration + vsyncToPhotons;
        vrSystem->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, (float)predictedDisplayTime, _trackedDevicePose, vr::k_unMaxTrackedDeviceCount);
        auto basePose = openvr::toGlm(_trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking);
        auto baseRotation = glm::quat_cast(glm::mat3(basePose));
        baseRotation = glm::quat(baseRotation.w, -baseRotation.x, baseRotation.y, -baseRotation.z);
        basePose = glm::mat4_cast(baseRotation);

        eyeViews = std::array<glm::mat4, 2>{ glm::inverse(basePose * eyeOffsets[0]), glm::inverse(basePose * eyeOffsets[1]) };
        Parent::update(delta);
    }

    void render() {
        context.device.waitForFences(frameFences[frameIndex], VK_TRUE, UINT64_MAX);
        context.device.resetFences(frameFences[frameIndex]);

        auto currentImageResult = swapchain.acquireNextImage(shapesRenderer->semaphores.renderStart, frameFences[frameIndex]);
        auto currentImage = currentImageResult.value;

        shapesRenderer->render();

        // Perform both eye blits and the mirror blit concurrently
        context.submit({ stagingBlitCommands[frameIndex][vr::Eye_Left], stagingBlitCommands[frameIndex][vr::Eye_Right], mirrorBlitCommands[currentImage] },
                       { { shapesRenderer->semaphores.swapchainFilled, vk::PipelineStageFlagBits::eColorAttachmentOutput } },
                       { stagingBlitCompletes[frameIndex] });

        //-----------------------------------------------------------------------------------------
        // OpenVR BEGIN: Submit eyes to compositor, left eye just rendered
        //-----------------------------------------------------------------------------------------
        vr::VRTextureBounds_t textureBounds;
        textureBounds.uMin = 0.0f;
        textureBounds.uMax = 1.0f;
        textureBounds.vMin = 0.0f;
        textureBounds.vMax = 1.0f;

        vr::VRVulkanTextureData_t vulkanData;
        vulkanData.m_pDevice = (VkDevice_T*)context.device;
        vulkanData.m_pPhysicalDevice = (VkPhysicalDevice_T*)context.physicalDevice;
        vulkanData.m_pInstance = (VkInstance_T*)context.instance;
        vulkanData.m_pQueue = (VkQueue_T*)context.graphicsQueue;
        vulkanData.m_nQueueFamilyIndex = context.queueIndices.graphics;
        vulkanData.m_nWidth = renderTargetSize.x / 2;
        vulkanData.m_nHeight = renderTargetSize.y;
        vulkanData.m_nFormat = (uint32_t)stagingImages[frameIndex][vr::Eye_Left].format;
        vulkanData.m_nSampleCount = 1;
        vr::Texture_t texture = { &vulkanData, vr::TextureType_Vulkan, vr::ColorSpace_Auto };

        // Submit left eye
        vulkanData.m_nImage = (uint64_t)(VkImage)stagingImages[frameIndex][vr::Eye_Left].image;
        vr::VRCompositor()->Submit(vr::Eye_Left, &texture, &textureBounds);

        // Submit right eye
        vulkanData.m_nImage = (uint64_t)(VkImage)stagingImages[frameIndex][vr::Eye_Right].image;
        vr::VRCompositor()->Submit(vr::Eye_Right, &texture, &textureBounds);

        swapchain.queuePresent(stagingBlitCompletes[frameIndex]);
        frameIndex = (frameIndex + 1) % FRAME_LAG;
    }

    std::string getWindowTitle() {
        std::string device(context.deviceProperties.deviceName);
        return "OpenVR SDK Example " + device + " - " + std::to_string((int)lastFPS) + " fps";
    }
};
#endif

class OpenXrExample {
public:
    void run() {}
};

RUN_EXAMPLE(OpenXrExample)
