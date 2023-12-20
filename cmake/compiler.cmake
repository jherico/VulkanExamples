include(CheckCXXCompilerFlag)

set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

if (MSVC)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
    # Enable incremental compiling and "edit and continue"
    add_compile_options("/ZI")
    add_link_options("/INCREMENTAL")
    CHECK_CXX_COMPILER_FLAG("/openmp" _openmp_supported)
    if (_openmp_supported)
        add_compile_options("/openmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

