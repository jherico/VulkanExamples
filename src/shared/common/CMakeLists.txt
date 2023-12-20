set(TARGET_NAME vks_common)
GroupSources("common")
file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/shared)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "shared")

target_glm()
