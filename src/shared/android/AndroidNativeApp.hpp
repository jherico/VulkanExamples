//
// Created by Brad on 12/21/2017.
//

#pragma once
#ifndef VULKAN_ANDROIDNATIVEAPP_H
#define VULKAN_ANDROIDNATIVEAPP_H

#include "android_native_app_glue.h"
#include <type_traits>

namespace android {
class NativeApp {
public:
    virtual ~NativeApp() {}

protected:
    static int32_t inputCallback(struct android_app* app, AInputEvent* event) {
        NativeApp* appClass = (NativeApp*)app->userData;
        return appClass->onInput(event);
    }

    static void appCmdCallback(struct android_app* app, int32_t cmd) {
        NativeApp* appClass = (NativeApp*)app->userData;
        appClass->onCmd(cmd);
    }

    virtual int32_t onInput(AInputEvent* event) { return 0; }

    virtual void onCmd(int32_t cmd) {
        switch (cmd) {
            case APP_CMD_CONFIG_CHANGED:
                onConfigChanged();
                break;
            case APP_CMD_CONTENT_RECT_CHANGED:
                onContentRectChanged();
                break;
            case APP_CMD_DESTROY:
                onDestroy();
                break;
            case APP_CMD_GAINED_FOCUS:
                onGainedFocus();
                break;
            case APP_CMD_INIT_WINDOW:
                onInitWindow();
                break;
            case APP_CMD_INPUT_CHANGED:
                onInputCHanged();
                break;
            case APP_CMD_LOST_FOCUS:
                onLostFocus();
                break;
            case APP_CMD_LOW_MEMORY:
                onLowMemory();
                break;
            case APP_CMD_PAUSE:
                onPause();
                break;
            case APP_CMD_RESUME:
                onResume();
                break;
            case APP_CMD_SAVE_STATE:
                onSaveState();
                break;
            case APP_CMD_START:
                onStart();
                break;
            case APP_CMD_STOP:
                onStop();
                break;
            case APP_CMD_TERM_WINDOW:
                onTermWindow();
                break;
            case APP_CMD_WINDOW_REDRAW_NEEDED:
                onWindowRedrawNeeded();
                break;
            case APP_CMD_WINDOW_RESIZED:
                onWindowResized();
                break;
        }
    }

    virtual void onConfigChanged() {}

    virtual void onContentRectChanged() {}

    virtual void onDestroy() {}

    virtual void onGainedFocus() {}

    virtual void onInitWindow() {}

    virtual void onInputCHanged() {}

    virtual void onLostFocus() {}

    virtual void onLowMemory() {}

    virtual void onPause() {}

    virtual void onResume() {}

    virtual void onSaveState() {}

    virtual void onStart() {}

    virtual void onStop() {}

    virtual void onTermWindow() {}

    virtual void onWindowRedrawNeeded() {}

    virtual void onWindowResized() {}

protected:
    NativeApp(android_app* app)
        : app(app) {
        app->userData = this;
        app->onAppCmd = appCmdCallback;
        app->onInputEvent = inputCallback;
    }

    android_app* const app;
};

template <typename T>
class NativeStatefulApp : public NativeApp {
    using Parent = NativeApp;
    // FIXME
    // static_assert(std::is_trivally_copyable<T>::value);

public:
    NativeStatefulApp(android_app* app)
        : Parent(app) {
        if (app->savedState != nullptr) {
            // We are starting with a previous saved state; restore from it.
            state = *reinterpret_cast<const T*>(app->savedState);
        }
    }

protected:
    void onSaveState() override {
        size_t size = sizeof(T);
        app->savedStateSize = size;
        app->savedState = malloc(size);
        *((T*)app->savedState) = state;
    }

    T state;
};
}  // namespace android

#endif  //VULKAN_ANDROIDNATIVEAPP_H
