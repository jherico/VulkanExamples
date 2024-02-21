set(TARGET_NAME vks_examples)
GroupSources("examples")
file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
target_link_libraries(vks_examples PUBLIC vks_rendering)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "shared")

set(SHADER_FILES
    ${CMAKE_SOURCE_DIR}/data/shaders/base/uioverlay.vert
    ${CMAKE_SOURCE_DIR}/data/shaders/base/uioverlay.frag
)

target_shader_files()

target_glfw3()
if (ENABLE_UI)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_UI=1)
    target_imgui()
else()
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_UI=0)
endif()
