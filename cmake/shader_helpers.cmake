macro(FIND_SHADERS)
    set(TARGET_SHADER_DIR "${PROJECT_SOURCE_DIR}/data/shaders")
    file(GLOB_RECURSE SHADER_FILES
        "${TARGET_SHADER_DIR}/*.vert"
        "${TARGET_SHADER_DIR}/*.frag"
        "${TARGET_SHADER_DIR}/*.tesc"
        "${TARGET_SHADER_DIR}/*.tese"
        "${TARGET_SHADER_DIR}/*.comp"
        "${TARGET_SHADER_DIR}/*.geom"
    )
endmacro()

function(COMPILE_SPIRV_SHADER SHADER_FILE)
    # Define the final name of the generated shader file
    find_program(GLSLANG_EXECUTABLE glslangValidator
        HINTS "$ENV{VULKAN_SDK}/bin")
    find_program(SPIRV_OPT_EXECUTABLE spirv-opt
        HINTS "$ENV{VULKAN_SDK}/bin")
    file(RELATIVE_PATH DEST_SHADER ${CMAKE_SOURCE_DIR} ${SHADER_FILE})
    set(COMPILE_OUTPUT "${CMAKE_BINARY_DIR}/${DEST_SHADER}.debug.spv")
    set(OPTIMIZE_OUTPUT "${CMAKE_BINARY_DIR}/${DEST_SHADER}.spv")
    message(DEBUG "Compiling shader ${SHADER_FILE} to ${COMPILE_OUTPUT}")
    add_custom_command(
        OUTPUT ${COMPILE_OUTPUT}
        COMMAND ${GLSLANG_EXECUTABLE} -V ${SHADER_FILE} -o ${COMPILE_OUTPUT}
        DEPENDS ${SHADER_FILE})
    add_custom_command(
        OUTPUT ${OPTIMIZE_OUTPUT}
        COMMAND ${SPIRV_OPT_EXECUTABLE} -O ${COMPILE_OUTPUT} -o ${OPTIMIZE_OUTPUT}
        DEPENDS ${COMPILE_OUTPUT})
    set(COMPILE_SPIRV_SHADER_RETURN ${OPTIMIZE_OUTPUT} PARENT_SCOPE)
endfunction()

function(COMPILE_SHADER SHADER_FILE)
    file(RELATIVE_PATH DEST_SHADER ${CMAKE_SOURCE_DIR}/data/shaders ${SHADER_FILE})
    set(SHADER_HEADER_NAME "${CMAKE_BINARY_DIR}/data/shaders/${DEST_SHADER}.inl")
    get_source_file_property(SHADER_EXISTS ${SHADER_HEADER_NAME} GENERATED)
    get_filename_component(SHADER_FOLDER ${DEST_SHADER} DIRECTORY)
    if (SHADER_EXISTS)
        message(DEBUG "Shader ${SHADER_FILE} already has build rules. Skipping.")
        return()
    endif()
    message(DEBUG "Compiling shader ${SHADER_FILE}")
    compile_spirv_shader(${SHADER_FILE})
    message(DEBUG "\tCompiled file ${COMPILE_SPIRV_SHADER_RETURN}")
    message(DEBUG "\tHeader file ${SHADER_HEADER_NAME}")
    add_custom_command(
        COMMAND ${CMAKE_COMMAND} -DCONFIG_FILE="${CMAKE_SOURCE_DIR}/cmake/ShaderData.inl.in" -DTARGET_NAME="${SHADER_FOLDER}" -DSHADER_FILE="${SHADER_FILE}" -DSHADER_SPIRV="${COMPILE_SPIRV_SHADER_RETURN}" -DSHADER_HEADER_NAME="${SHADER_HEADER_NAME}" -P ${CMAKE_SOURCE_DIR}/cmake/CreateShaderHeader.cmake
        OUTPUT ${SHADER_HEADER_NAME}
        DEPENDS ${COMPILE_SPIRV_SHADER_RETURN}
        COMMENT "Making Shader Header ${SHADER_HEADER_NAME}"
    )
    source_group("Shaders" FILES ${SHADER_FILE})
    source_group("Compiled Shaders" FILES ${SHADER_HEADER_NAME})
endfunction()

macro(FIND_TARGET_SHADERS)
    set(TARGET_SHADER_DIR "${PROJECT_SOURCE_DIR}/data/shaders/${TARGET_NAME}")
    file(GLOB SHADER_FILES
        "${TARGET_SHADER_DIR}/*.vert"
        "${TARGET_SHADER_DIR}/*.frag"
        "${TARGET_SHADER_DIR}/*.tesc"
        "${TARGET_SHADER_DIR}/*.tese"
        "${TARGET_SHADER_DIR}/*.comp"
        "${TARGET_SHADER_DIR}/*.geom"
    )
endmacro()

macro(TARGET_SHADER_FILES)
    foreach(SHADER_FILE ${SHADER_FILES})
        compile_shader(${SHADER_FILE})
        target_sources(${TARGET_NAME} PRIVATE ${SHADER_FILES})
        file(RELATIVE_PATH DEST_SHADER ${CMAKE_SOURCE_DIR} ${SHADER_FILE})
        set(SHADER_HEADER_NAME "${CMAKE_BINARY_DIR}/${DEST_SHADER}.inl")
        target_sources(${TARGET_NAME} PRIVATE ${SHADER_HEADER_NAME})
    endforeach()
endmacro()
