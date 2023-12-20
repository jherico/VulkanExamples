set(TARGET_NAME vks_gl)

GroupSources("gl")
file(GLOB_RECURSE COMMON_SOURCE *.c *.cpp *.h *.hpp)
add_library(${TARGET_NAME} STATIC ${COMMON_SOURCE})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "shared")

target_glad()