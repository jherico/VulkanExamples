#
#  Created by Bradley Austin Davis on 2016/02/16
#
#  Distributed under the Apache License, Version 2.0.
#  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
#
macro(TARGET_VULKAN)
    if (ANDROID)
        target_link_libraries(${TARGET_NAME} PUBLIC vulkan)
        target_include_directories(${TARGET_NAME} PUBLIC $ENV{VULKAN_SDK}/include)
    else()
        find_package(Vulkan REQUIRED)
        target_link_libraries(${TARGET_NAME} PUBLIC Vulkan::Vulkan )
    endif()
    find_package(VulkanMemoryAllocator CONFIG REQUIRED)
    target_link_libraries(${TARGET_NAME} PUBLIC GPUOpen::VulkanMemoryAllocator)
    target_compile_definitions(${TARGET_NAME} PUBLIC USE_VMA=1)
    target_compile_definitions(${TARGET_NAME} PUBLIC VMA_VULKAN_VERSION=1003000)
endmacro()
