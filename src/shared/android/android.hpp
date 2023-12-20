/*
* Android Vulkan function pointer prototypes
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#if defined(__ANDROID__)

#include <android_native_app_glue.h>
#include <android/asset_manager.h>

namespace vkx { namespace android {
/* @brief Touch control thresholds from Android NDK samples */
const int32_t DOUBLE_TAP_TIMEOUT = 300 * 1000000;
const int32_t TAP_TIMEOUT = 180 * 1000000;
const int32_t DOUBLE_TAP_SLOP = 100;
const int32_t TAP_SLOP = 8;

/** @brief Density of the device screen (in DPI) */
extern int32_t screenDensity;
extern android_app* androidApp;

void getDeviceConfig(AAssetManager* assetManager);
}}  // namespace vkx::android

#endif
