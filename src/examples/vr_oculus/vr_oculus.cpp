#include <vr/vr_common.hpp>
#include <OVR_CAPI_Vk.h>

namespace ovr {
using TextureSwapChainDesc = ovrTextureSwapChainDesc;
using Session = ovrSession;
using HmdDesc = ovrHmdDesc;
using GraphicsLuid = ovrGraphicsLuid;
using TextureSwapChain = ovrTextureSwapChain;
using MirrorTexture = ovrMirrorTexture;
using EyeRenderDesc = ovrEyeRenderDesc;
using LayerEyeFov = ovrLayerEyeFov;
using ViewScaleDesc = ovrViewScaleDesc;
using Posef = ovrPosef;
using EyePoses = std::array<Posef, 2>;

//using EyeType = ovrEyeType;
enum class EyeType
{
    Left = ovrEye_Left,
    Right = ovrEye_Right
};

// Convenience method for looping over each eye with a lambda
template <typename Function>
inline void for_each_eye(Function function) {
    for (ovrEyeType eye = ovrEyeType::ovrEye_Left; eye < ovrEyeType::ovrEye_Count; eye = static_cast<ovrEyeType>(eye + 1)) {
        function(eye);
    }
}

inline mat4 toGlm(const ovrMatrix4f& om) {
    return glm::transpose(glm::make_mat4(&om.M[0][0]));
}

inline mat4 toGlm(const ovrFovPort& fovport, float nearPlane = 0.01f, float farPlane = 10000.0f) {
    return toGlm(ovrMatrix4f_Projection(fovport, nearPlane, farPlane, true));
}

inline vec3 toGlm(const ovrVector3f& ov) {
    return glm::make_vec3(&ov.x);
}

inline vec2 toGlm(const ovrVector2f& ov) {
    return glm::make_vec2(&ov.x);
}

inline uvec2 toGlm(const ovrSizei& ov) {
    return uvec2(ov.w, ov.h);
}

inline quat toGlm(const ovrQuatf& oq) {
    return glm::make_quat(&oq.x);
}

inline mat4 toGlm(const ovrPosef& op) {
    mat4 orientation = glm::mat4_cast(toGlm(op.Orientation));
    mat4 translation = glm::translate(mat4(), ovr::toGlm(op.Position));
    return translation * orientation;
}

inline std::array<glm::mat4, 2> toGlm(const EyePoses& eyePoses) {
    return std::array<glm::mat4, 2>{ toGlm(eyePoses[0]), toGlm(eyePoses[1]) };
}

inline ovrMatrix4f fromGlm(const mat4& m) {
    ovrMatrix4f result;
    mat4 transposed(glm::transpose(m));
    memcpy(result.M, &(transposed[0][0]), sizeof(float) * 16);
    return result;
}

inline ovrVector3f fromGlm(const vec3& v) {
    ovrVector3f result;
    result.x = v.x;
    result.y = v.y;
    result.z = v.z;
    return result;
}

inline ovrVector2f fromGlm(const vec2& v) {
    ovrVector2f result;
    result.x = v.x;
    result.y = v.y;
    return result;
}

inline ovrSizei fromGlm(const uvec2& v) {
    ovrSizei result;
    result.w = v.x;
    result.h = v.y;
    return result;
}

inline ovrQuatf fromGlm(const quat& q) {
    ovrQuatf result;
    result.x = q.x;
    result.y = q.y;
    result.z = q.z;
    result.w = q.w;
    return result;
}

void OVR_CDECL logger(uintptr_t userData, int level, const char* message) {
    OutputDebugStringA("OVR_SDK: ");
    OutputDebugStringA(message);
    OutputDebugStringA("\n");
}
}  // namespace ovr

class OculusExample : public VrExample {
    using Parent = VrExample;

public:
    ovr::Session _session{};
    ovr::HmdDesc _hmdDesc{};
    ovr::GraphicsLuid _luid{};
    ovr::LayerEyeFov _sceneLayer;
    ovr::TextureSwapChain& _eyeTexture = _sceneLayer.ColorTexture[0];
    ovr::MirrorTexture _mirrorTexture;
    ovr::ViewScaleDesc _viewScaleDesc;
    ovrLayerHeader* headerList = &_sceneLayer.Header;
    vk::Semaphore blitComplete;
    std::vector<vk::CommandBuffer> oculusBlitCommands;
    std::vector<vk::CommandBuffer> mirrorBlitCommands;

    ~OculusExample() {
        // Shut down Oculus
        ovr_Destroy(_session);
        _session = nullptr;
        ovr_Shutdown();
    }

    void recenter() override { ovr_RecenterTrackingOrigin(_session); }

    void prepareOculus() {
        ovrInitParams initParams{ 0, OVR_MINOR_VERSION, ovr::logger, (uintptr_t)this, 0 };
        if (!OVR_SUCCESS(ovr_Initialize(&initParams))) {
            throw std::runtime_error("Unable to initialize Oculus SDK");
        }

        if (!OVR_SUCCESS(ovr_Create(&_session, &_luid))) {
            throw std::runtime_error("Unable to create HMD session");
        }

        _hmdDesc = ovr_GetHmdDesc(_session);
        _viewScaleDesc.HmdSpaceToWorldScaleInMeters = 1.0f;
        memset(&_sceneLayer, 0, sizeof(ovrLayerEyeFov));
        _sceneLayer.Header.Type = ovrLayerType_EyeFov;
        _sceneLayer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;

        ovr::for_each_eye([&](ovrEyeType eye) {
            ovrEyeRenderDesc erd = ovr_GetRenderDesc(_session, eye, _hmdDesc.DefaultEyeFov[eye]);
            ovrMatrix4f ovrPerspectiveProjection = ovrMatrix4f_Projection(erd.Fov, 0.1f, 256.0f, ovrProjection_ClipRangeOpenGL);
            eyeProjections[eye] = ovr::toGlm(ovrPerspectiveProjection);
            _viewScaleDesc.HmdToEyePose[eye] = erd.HmdToEyePose;

            ovrFovPort& fov = _sceneLayer.Fov[eye] = erd.Fov;
            auto eyeSize = ovr_GetFovTextureSize(_session, eye, fov, 1.0f);
            _sceneLayer.Viewport[eye].Size = eyeSize;
            _sceneLayer.Viewport[eye].Pos = { (int)renderTargetSize.x, 0 };
            renderTargetSize.y = std::max(renderTargetSize.y, (uint32_t)eyeSize.h);
            renderTargetSize.x += eyeSize.w;
        });

        context.requireExtensions({ VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
        context.setDevicePicker([this](const std::vector<vk::PhysicalDevice>& devices) -> vk::PhysicalDevice {
            VkPhysicalDevice result;
            if (!OVR_SUCCESS(ovr_GetSessionPhysicalDeviceVk(_session, _luid, context.instance, &result))) {
                throw std::runtime_error("Unable to identify Vulkan device");
            }
            return result;
        });
    }

    void prepareOculusSwapchain() {
        ovrTextureSwapChainDesc desc = {};
        desc.Type = ovrTexture_2D;
        desc.ArraySize = 1;
        desc.Width = renderTargetSize.x;
        desc.Height = renderTargetSize.y;
        desc.MipLevels = 1;
        desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
        desc.SampleCount = 1;
        desc.StaticImage = ovrFalse;
        if (!OVR_SUCCESS(ovr_CreateTextureSwapChainVk(_session, context.device, &desc, &_eyeTexture))) {
            throw std::runtime_error("Unable to create swap chain");
        }

        int oculusSwapchainLength = 0;
        if (!OVR_SUCCESS(ovr_GetTextureSwapChainLength(_session, _eyeTexture, &oculusSwapchainLength)) || !oculusSwapchainLength) {
            throw std::runtime_error("Unable to count swap chain textures");
        }

        // Submission command buffers
        if (oculusBlitCommands.empty()) {
            vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
            cmdBufAllocateInfo.commandPool = context.getCommandPool();
            cmdBufAllocateInfo.commandBufferCount = oculusSwapchainLength;
            oculusBlitCommands = context.device.allocateCommandBuffers(cmdBufAllocateInfo);
        }

        vk::ImageBlit sceneBlit;
        sceneBlit.dstSubresource.aspectMask = sceneBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        sceneBlit.dstSubresource.layerCount = sceneBlit.srcSubresource.layerCount = 1;
        sceneBlit.dstOffsets[1] = sceneBlit.srcOffsets[1] = vk::Offset3D{ (int32_t)renderTargetSize.x, (int32_t)renderTargetSize.y, 1 };
        for (int i = 0; i < oculusSwapchainLength; ++i) {
            vk::CommandBuffer& cmdBuffer = oculusBlitCommands[i];
            VkImage oculusImage;
            if (!OVR_SUCCESS(ovr_GetTextureSwapChainBufferVk(_session, _eyeTexture, i, &oculusImage))) {
                throw std::runtime_error("Unable to acquire vulkan image for index " + std::to_string(i));
            }
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(vk::CommandBufferBeginInfo{});
            context.setImageLayout(cmdBuffer, oculusImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            cmdBuffer.blitImage(shapesRenderer->framebuffer.colors[0].image, vk::ImageLayout::eTransferSrcOptimal, oculusImage,
                                vk::ImageLayout::eTransferDstOptimal, sceneBlit, vk::Filter::eNearest);
            context.setImageLayout(cmdBuffer, oculusImage, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eTransferSrcOptimal);
            cmdBuffer.end();
        }
    }

    void prepareOculusMirror() {
        // Mirroring command buffers
        ovrMirrorTextureDesc mirrorDesc;
        memset(&mirrorDesc, 0, sizeof(mirrorDesc));
        mirrorDesc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
        mirrorDesc.Width = size.x;
        mirrorDesc.Height = size.y;
        if (!OVR_SUCCESS(ovr_CreateMirrorTextureWithOptionsVk(_session, context.device, &mirrorDesc, &_mirrorTexture))) {
            throw std::runtime_error("Could not create mirror texture");
        }

        VkImage mirrorImage;
        ovr_GetMirrorTextureBufferVk(_session, _mirrorTexture, &mirrorImage);
        if (mirrorBlitCommands.empty()) {
            vk::CommandBufferAllocateInfo cmdBufAllocateInfo;
            cmdBufAllocateInfo.commandPool = context.getCommandPool();
            cmdBufAllocateInfo.commandBufferCount = swapchain.imageCount;
            mirrorBlitCommands = context.device.allocateCommandBuffers(cmdBufAllocateInfo);
        }

        vk::ImageBlit mirrorBlit;
        mirrorBlit.dstSubresource.aspectMask = mirrorBlit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        mirrorBlit.dstSubresource.layerCount = mirrorBlit.srcSubresource.layerCount = 1;
        mirrorBlit.srcOffsets[1] = mirrorBlit.dstOffsets[1] = { (int32_t)size.x, (int32_t)size.y, 1 };

        for (size_t i = 0; i < swapchain.imageCount; ++i) {
            vk::CommandBuffer& cmdBuffer = mirrorBlitCommands[i];
            cmdBuffer.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
            cmdBuffer.begin(vk::CommandBufferBeginInfo{});
            context.setImageLayout(cmdBuffer, swapchain.images[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eTransferDstOptimal);
            cmdBuffer.blitImage(mirrorImage, vk::ImageLayout::eTransferSrcOptimal, swapchain.images[i].image, vk::ImageLayout::eTransferDstOptimal, mirrorBlit,
                                vk::Filter::eNearest);
            context.setImageLayout(cmdBuffer, swapchain.images[i].image, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::ePresentSrcKHR);
            cmdBuffer.end();
        }
    }

    // Setup any Oculus specific work that requires an existing Vulkan instance/device/queue
    void prepareOculusVk() {
        ovr_SetSynchonizationQueueVk(_session, context.queue);
        prepareOculusSwapchain();
        prepareOculusMirror();
        blitComplete = context.device.createSemaphore({});
    }

    void prepare() {
        prepareOculus();
        // FIXME the Oculus API hangs if validation is enabled
        // context.setValidationEnabled(true);
        Parent::prepare();
        prepareOculusVk();
    }

    void update(float delta) {
        ovrResult result;
        ovrSessionStatus status;
        result = ovr_GetSessionStatus(_session, &status);
        if (!OVR_SUCCESS(result)) {
            throw std::runtime_error("Can't get session status");
        }

        while (!status.IsVisible || !status.HmdMounted) {
            ovrResult result = ovr_GetSessionStatus(_session, &status);
            if (!OVR_SUCCESS(result)) {
                throw std::runtime_error("Can't get session status");
            }
            Sleep(100);
        }

        ovr_WaitToBeginFrame(_session, frameCounter);
        ovr::EyePoses eyePoses;
        ovr_GetEyePoses(_session, frameCounter, true, _viewScaleDesc.HmdToEyePose, eyePoses.data(), &_sceneLayer.SensorSampleTime);
        eyeViews = std::array<glm::mat4, 2>{ glm::inverse(ovr::toGlm(eyePoses[0])), glm::inverse(ovr::toGlm(eyePoses[1])) };
        ovr::for_each_eye([&](ovrEyeType eye) {
            const auto& vp = _sceneLayer.Viewport[eye];
            _sceneLayer.RenderPose[eye] = eyePoses[eye];
        });
        Parent::update(delta);
    }

    void render() {
        vk::Fence submitFence = swapchain.getSubmitFence(true);
        auto swapchainAcquireResult = swapchain.acquireNextImage(shapesRenderer->semaphores.renderStart);
        auto swapchainIndex = swapchainAcquireResult.value;

        ovrResult result;
        result = ovr_BeginFrame(_session, frameCounter);

        shapesRenderer->render();

        int oculusIndex;
        result = ovr_GetTextureSwapChainCurrentIndex(_session, _eyeTexture, &oculusIndex);
        if (!OVR_SUCCESS(result)) {
            throw std::runtime_error("Unable to acquire next texture index");
        }

        // Blit from our framebuffer to the Oculus output image (pre-recorded command buffer)
        context.submit(oculusBlitCommands[oculusIndex], { { shapesRenderer->semaphores.renderComplete, vk::PipelineStageFlagBits::eColorAttachmentOutput } });

        // The lack of explicit synchronization here is baffling.  One of these calls must be blocking,
        // meaning there would have to be some backend use of waitIdle or fences, meaning less optimal
        // performance than with semaphores
        result = ovr_CommitTextureSwapChain(_session, _eyeTexture);
        if (!OVR_SUCCESS(result)) {
            throw std::runtime_error("Unable to commit swap chain for index " + std::to_string(oculusIndex));
        }

        result = ovr_EndFrame(_session, frameCounter, &_viewScaleDesc, &headerList, 1);
        if (!OVR_SUCCESS(result)) {
            throw std::runtime_error("Unable to submit frame for index " + std::to_string(oculusIndex));
        }

        // Blit from the mirror buffer to the swap chain image
        // Technically I could move this to with the other submit, for blitting the framebuffer to the texture,
        // but there's no real way of knowing when this image is properly populated.  Presumably its reliable here
        // because of the blocking functionality of the ovr_SubmitFrame (or the ovr_CommitTextureSwapChain).
        context.submit(mirrorBlitCommands[swapchainIndex], {}, {}, blitComplete, submitFence);
        swapchain.queuePresent(blitComplete);
    }

    std::string getWindowTitle() {
        std::string device(context.deviceProperties.deviceName);
        return "Oculus SDK Example " + device + " - " + std::to_string((int)lastFPS) + " fps";
    }
};

RUN_EXAMPLE(OculusExample)
