set(EXTERNAL_NAME vkcpp)
string(TOUPPER ${EXTERNAL_NAME} EXTERNAL_NAME_UPPER)

include(ExternalProject)
ExternalProject_Add(
    ${EXTERNAL_NAME}
    DOWNLOAD_NAME Vulkan-Hpp-cmake.zip
    URL https://codeload.github.com/jherico/Vulkan-Hpp/zip/cmake
    URL_MD5 ce91c22abe80475f3cb317982cedc1cc
    CMAKE_ARGS ${PLATFORM_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DVK_SPEC_URL:STRING=https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/v1.0-core%2Bwsi-20160812/src/spec/vk.xml
    LOG_DOWNLOAD 1
)

# Hide this external target (for ide users)
set_target_properties(${EXTERNAL_NAME} PROPERTIES FOLDER "externals")
ExternalProject_Get_Property(${EXTERNAL_NAME} INSTALL_DIR)
set(${EXTERNAL_NAME_UPPER}_INCLUDE_DIRS ${INSTALL_DIR}/include CACHE TYPE INTERNAL)

