# setup for find modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

# setup for custom macros
file(GLOB CUSTOM_MACROS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/macros/*.cmake")
foreach(CUSTOM_MACRO ${CUSTOM_MACROS})
  include(${CUSTOM_MACRO})
endforeach()

# setup for properties
set_property(GLOBAL PROPERTY CMAKE_CXX_STANDARD 20)
set_property(GLOBAL PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")

if (POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()

# Enable Hot Reload for MSVC compilers if supported. (Make changes to program during compilation)
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
