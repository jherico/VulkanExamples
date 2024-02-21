# We can't use the vcpkg version of glad because it doesn't have the external memory extensions we need for Vulkan resource sharing
set(TARGET_NAME glad)

file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "ThirdParty")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


