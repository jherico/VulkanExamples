macro(TARGET_GLM)
    find_package(glm CONFIG REQUIRED)
    target_link_libraries(${TARGET_NAME} PUBLIC glm::glm)
    target_compile_definitions(${TARGET_NAME} PUBLIC GLM_FORCE_RADIANS)
    target_compile_definitions(${TARGET_NAME} PUBLIC GLM_FORCE_CTOR_INIT)
    target_compile_definitions(${TARGET_NAME} PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
endmacro()

