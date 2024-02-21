cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("cmake/defaults.cmake")
set(NAME VulkanCppExamples)

include(${CMAKE_SOURCE_DIR}/cmake/ezvcpkg/ezvcpkg.cmake)

ezvcpkg_fetch(
    COMMIT 2024.02.14
    PACKAGES assimp basisu glfw3 gli glm imgui openxr-loader vulkan vulkan-memory-allocator
    UPDATE_TOOLCHAIN
)

project(${NAME})

if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    message( FATAL_ERROR "Only 64 bit builds supported." )
endif()

option(ENABLE_UI "help string describing option" 1)


include("cmake/compiler.cmake")

find_package(Threads REQUIRED)

# This define is specific to Vulkan has a depth range of [0, 1], unlike OpenGL which has a [-1, 1]
add_definitions(-DGLM_FORCE_DEPTH_ZERO_TO_ONE)

if (NOT ANDROID)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin_debug")
    set(CMAKE_DEBUG_POSTFIX "d")

    add_custom_target(SetupRelease ALL ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin)
    set_target_properties(SetupRelease PROPERTIES FOLDER "CMakeTargets")
    add_custom_target(SetupDebug ALL ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin_debug)
    set_target_properties(SetupDebug PROPERTIES FOLDER "CMakeTargets")
endif()

if (ANDROID)
    add_definitions(-DVULKAN_HPP_NO_SMART_HANDLE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")
    set(APP_GLUE_DIR ${ANDROID_NDK}/sources/android/native_app_glue)
    add_library(app-glue STATIC ${APP_GLUE_DIR}/android_native_app_glue.c)
endif()

if (ANDROID)
    add_definitions(-DVK_USE_PLATFORM_ANDROID_KHR)
elseif (WIN32)
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
else()
    add_definitions(-DVK_USE_PLATFORM_XCB_KHR)
    find_package(XCB REQUIRED)
    link_libraries(${XCB_LIBRARIES})
endif()

if (NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    link_libraries(${CMAKE_THREAD_LIBS_INIT})
endif()

include(cmake/shader_helpers.cmake)
add_subdirectory(src)
