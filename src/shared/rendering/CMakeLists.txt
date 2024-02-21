set(TARGET_NAME vks_rendering)
file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
target_link_libraries(vks_rendering PUBLIC vks_common vks_vks)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "shared")
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_BINARY_DIR}/data)


set(SHADER_FILES
    ${CMAKE_SOURCE_DIR}/data/shaders/pbr/genbrdflut.vert
    ${CMAKE_SOURCE_DIR}/data/shaders/pbr/genbrdflut.frag
    ${CMAKE_SOURCE_DIR}/data/shaders/pbr/filtercube.vert
    ${CMAKE_SOURCE_DIR}/data/shaders/pbr/irradiancecube.frag
    ${CMAKE_SOURCE_DIR}/data/shaders/pbr/prefilterenvmap.frag
)

target_shader_files()

target_assimp()
