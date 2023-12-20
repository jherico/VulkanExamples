if (POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()

if (NOT ANDROID)
    add_subdirectory(thirdparty/glad)
endif()

add_subdirectory(shared/common)
target_include_directories(vks_common PUBLIC ${PROJECT_SOURCE_DIR}/src/shared)
add_subdirectory(shared/vks)

if (ANDROID)
    add_subdirectory(shared/android)
endif()

add_subdirectory(shared/opengl)
add_subdirectory(shared/rendering)
add_subdirectory(shared/examples)
add_subdirectory(examples)


