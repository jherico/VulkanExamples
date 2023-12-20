
// FIXME work in progress
namespace oculus {

class exception : public std::runtime_error {
public:
    exception(const char* what)
        : std::runtime_error(what) {}
};

void checkResult(ovrResult result, const std::string& message) {
    if (!OVR_SUCCESS(result)) {
        throw exception(message.c_str());
    }
}

class Session {
public:
    static void initialize(const ::ovrInitParams& params = {}) { checkResult(::ovr_Initialize(&params), "Unable to intialize Oculus SDK"); }

    static void shutdown() { ovr_Shutdown(); }

    static ovrErrorInfo getLastErrorInfo() {
        ovrErrorInfo errorInfo;
        ovr_GetLastErrorInfo(&errorInfo);
        return errorInfo;
    }

    static std::string getVersionString() {
        std::string result;
        result = ovr_GetVersionString();
        return result;
    }

    static int traceMessage(ovrLogLevel logLevel, const std::string& message) { return ovr_TraceMessage(logLevel, message.c_str()); }

    static void identifyClient(const std::string& identity) { checkResult(::ovr_IdentifyClient(identity.c_str()), "Unable to identify client"); }

    void create() { checkResult(ovr_Create(&m_session & m_graphicsLuid), "Unable to create Oculus session"); }

    void destroy() {
        ::ovr_Destroy(m_session);
        m_session = nullptr;
    }

    ovrSessionStatus getStatus() const {
        ovrSessionStatus result;
        checkResult(::ovr_GetSessionStatus(m_session, &result), "Unable to get session status");
        return result;
    }

    ovrHmdDesc getHmdDesc() const { return ::ovr_GetHmdDesc(m_session); }

    unsigned int getTrackerCount() const { return ::ovr_GetTrackerCount(m_session); }

    ovrTrackerDesc getTrackerDesc(unsigned int trackerDescIndex) const { return ::ovr_GetTrackerDesc(m_session, trackerDescIndex); }

    std::vector<ovrTrackerDesc> getTrackerDescs() const {
        auto count = getTrackerCount();
        std::vector<ovrTrackerDesc> result;
        result.resize(count);
        for (unsigned int i = 0; i < count; ++i) {
            result[i] = getTrackerDesc(i);
        }
        return result;
    }

    ovrTrackerPose getTrackerPose(unsigned int trackerPoseIndex) const { return ::ovr_GetTrackerPose(m_session, trackerPoseIndex); }

    std::vector<ovrTrackerPose> getTrackerPoses() const {
        auto count = getTrackerCount();
        std::vector<ovrTrackerPose> result;
        result.resize(count);
        for (unsigned int i = 0; i < count; ++i) {
            result[i] = getTrackerPose(i);
        }
        return result;
    }

    void setTrackingOriginType(ovrTrackingOrigin origin) const { checkResult(::ovr_SetTrackingOriginType(m_session, origin), "Unable to set tracking status"); }

    ovrTrackingOrigin setTrackingOriginType() const { return ::ovr_GetTrackingOriginType(m_session); }

    void recenterTrackingOrigin() const { checkResult(::ovr_RecenterTrackingOrigin(m_session), "Unable to reset tracking origin"); }

    void specifyTrackingOrigin(const ovrPosef& originPose) const {
        checkResult(::ovr_SpecifyTrackingOrigin(m_session, originPose), "Unable to set tracking origin");
    }

    void clearShouldRecenterFlag() const { ::ovr_ClearShouldRecenterFlag(m_session); }

    ovrTrackingState getTrackingState(bool latencyMarker = false, double absTime = 0.0) const {
        return ::ovr_GetTrackingState(m_session, absTime, latencyMarker);
    }

    std::vector<ovrPoseStatef> getDevicePoses(std::vector<ovrTrackedDeviceType> devices, double absTime = 0.0) const {
        std::vector<ovrPoseStatef> result;
        int count = devices.size();
        result.resize(count);
        checkResult(::ovr_GetDevicePoses(m_session, devices.data(), count, absTime, result.data()), "Unable to fetch device poses");
        return result;
    }

private:
    ::ovrSession m_session{ nullptr };
    ::ovrGraphicsLuid m_graphicsLuid;
};

#if 0
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetInputState(ovrSession session, ovrControllerType controllerType, ovrInputState* inputState);
        OVR_PUBLIC_FUNCTION(unsigned int) ovr_GetConnectedControllerTypes(ovrSession session);
        OVR_PUBLIC_FUNCTION(ovrTouchHapticsDesc)
            ovr_GetTouchHapticsDesc(ovrSession session, ovrControllerType controllerType);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_SetControllerVibration(
                ovrSession session,
                ovrControllerType controllerType,
                float frequency,
                float amplitude);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_SubmitControllerVibration(
                ovrSession session,
                ovrControllerType controllerType,
                const ovrHapticsBuffer* buffer);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetControllerVibrationState(
                ovrSession session,
                ovrControllerType controllerType,
                ovrHapticsPlaybackState* outState);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_TestBoundary(
                ovrSession session,
                ovrTrackedDeviceType deviceBitmask,
                ovrBoundaryType boundaryType,
                ovrBoundaryTestResult* outTestResult);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_TestBoundaryPoint(
                ovrSession session,
                const ovrVector3f* point,
                ovrBoundaryType singleBoundaryType,
                ovrBoundaryTestResult* outTestResult);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_SetBoundaryLookAndFeel(ovrSession session, const ovrBoundaryLookAndFeel* lookAndFeel);
        OVR_PUBLIC_FUNCTION(ovrResult) ovr_ResetBoundaryLookAndFeel(ovrSession session);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetBoundaryGeometry(
                ovrSession session,
                ovrBoundaryType boundaryType,
                ovrVector3f* outFloorPoints,
                int* outFloorPointsCount);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetBoundaryDimensions(
                ovrSession session,
                ovrBoundaryType boundaryType,
                ovrVector3f* outDimensions);
        OVR_PUBLIC_FUNCTION(ovrResult) ovr_GetBoundaryVisible(ovrSession session, ovrBool* outIsVisible);
        OVR_PUBLIC_FUNCTION(ovrResult) ovr_RequestBoundaryVisible(ovrSession session, ovrBool visible);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetTextureSwapChainLength(ovrSession session, ovrTextureSwapChain chain, int* out_Length);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetTextureSwapChainCurrentIndex(ovrSession session, ovrTextureSwapChain chain, int* out_Index);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetTextureSwapChainDesc(
                ovrSession session,
                ovrTextureSwapChain chain,
                ovrTextureSwapChainDesc* out_Desc);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_CommitTextureSwapChain(ovrSession session, ovrTextureSwapChain chain);
        OVR_PUBLIC_FUNCTION(void)
            ovr_DestroyTextureSwapChain(ovrSession session, ovrTextureSwapChain chain);
        OVR_PUBLIC_FUNCTION(void)
            ovr_DestroyMirrorTexture(ovrSession session, ovrMirrorTexture mirrorTexture);
        OVR_PUBLIC_FUNCTION(ovrSizei)
            ovr_GetFovTextureSize(
                ovrSession session,
                ovrEyeType eye,
                ovrFovPort fov,
                float pixelsPerDisplayPixel);
        OVR_PUBLIC_FUNCTION(ovrEyeRenderDesc)
            ovr_GetRenderDesc(ovrSession session, ovrEyeType eyeType, ovrFovPort fov);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_SubmitFrame(
                ovrSession session,
                long long frameIndex,
                const ovrViewScaleDesc* viewScaleDesc,
                ovrLayerHeader const* const* layerPtrList,
                unsigned int layerCount);
        OVR_PUBLIC_FUNCTION(ovrResult) ovr_GetPerfStats(ovrSession session, ovrPerfStats* outStats);
        OVR_PUBLIC_FUNCTION(ovrResult) ovr_ResetPerfStats(ovrSession session);
        OVR_PUBLIC_FUNCTION(double) ovr_GetPredictedDisplayTime(ovrSession session, long long frameIndex);
        OVR_PUBLIC_FUNCTION(double) ovr_GetTimeInSeconds();
        OVR_PUBLIC_FUNCTION(ovrBool)
            ovr_GetBool(ovrSession session, const char* propertyName, ovrBool defaultVal);
        OVR_PUBLIC_FUNCTION(ovrBool)
            ovr_SetBool(ovrSession session, const char* propertyName, ovrBool value);
        OVR_PUBLIC_FUNCTION(int) ovr_GetInt(ovrSession session, const char* propertyName, int defaultVal);
        OVR_PUBLIC_FUNCTION(ovrBool) ovr_SetInt(ovrSession session, const char* propertyName, int value);
        OVR_PUBLIC_FUNCTION(float)
            ovr_GetFloat(ovrSession session, const char* propertyName, float defaultVal);
        OVR_PUBLIC_FUNCTION(ovrBool)
            ovr_SetFloat(ovrSession session, const char* propertyName, float value);
        OVR_PUBLIC_FUNCTION(unsigned int)
            ovr_GetFloatArray(
                ovrSession session,
                const char* propertyName,
                float values[],
                unsigned int valuesCapacity);
        OVR_PUBLIC_FUNCTION(ovrBool)
            ovr_SetFloatArray(
                ovrSession session,
                const char* propertyName,
                const float values[],
                unsigned int valuesSize);
        OVR_PUBLIC_FUNCTION(const char*)
            ovr_GetString(ovrSession session, const char* propertyName, const char* defaultVal);
        OVR_PUBLIC_FUNCTION(ovrBool)
            ovr_SetString(ovrSession session, const char* propertyName, const char* value);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_GetExternalCameras(
                ovrSession session,
                ovrExternalCamera* cameras,
                unsigned int* inoutCameraCount);
        OVR_PUBLIC_FUNCTION(ovrResult)
            ovr_SetExternalCameraProperties(
                ovrSession session,
                const char* name,
                const ovrCameraIntrinsics* const intrinsics,
                const ovrCameraExtrinsics* const extrinsics);
#endif

}  // namespace oculus
