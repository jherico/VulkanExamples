set(TARGET_NAME vks_vks)

file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "shared")
target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/shared)
target_compile_definitions(${TARGET_NAME} PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

target_vulkan()

