cmake_minimum_required(VERSION 3.20)

project(h5cpp
    VERSION 0.1.0
    DESCRIPTION "Modern C++ wrapper around h5c"
    LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(H5CPP_BUILD_TESTS "Build the test programs" ${BUILD_TESTING})
option(H5CPP_ENABLE_PARALLEL "Build and test the parallel (MPI) wrapper" OFF)
set(H5CPP_CXX_STANDARD 17 CACHE STRING "C++ standard to compile against (17 or 20)")
set_property(CACHE H5CPP_CXX_STANDARD PROPERTY STRINGS 17 20)

# h5c is an independent project. Prefer an installed one; fall back to a
# sibling or submodule checkout so the two can be developed together.
find_package(h5c CONFIG QUIET)
if(NOT h5c_FOUND)
    set(_h5c_candidates
        "${CMAKE_CURRENT_SOURCE_DIR}/external/h5c"
        "${CMAKE_CURRENT_SOURCE_DIR}/../h5c")
    foreach(_dir IN LISTS _h5c_candidates)
        if(EXISTS "${_dir}/CMakeLists.txt")
            message(STATUS "h5c not installed; building from ${_dir}")
            add_subdirectory("${_dir}" "${CMAKE_CURRENT_BINARY_DIR}/h5c-build")
            set(h5c_FOUND TRUE)
            break()
        endif()
    endforeach()
endif()
if(NOT h5c_FOUND)
    message(FATAL_ERROR
        "h5c not found. Install it and set CMAKE_PREFIX_PATH, or place a "
        "checkout at external/h5c.")
endif()

# The parallel wrapper is a header over h5c_mpi.h: it exists only if the h5c we
# found was built with H5C_ENABLE_PARALLEL=ON, which puts H5C_HAVE_PARALLEL on
# the h5c target. Checking the target -- not a path or a version -- keeps this
# honest for both an installed h5c and an add_subdirectory() one.
if(H5CPP_ENABLE_PARALLEL)
    get_target_property(_h5c_defs h5c::h5c INTERFACE_COMPILE_DEFINITIONS)
    if(NOT _h5c_defs OR NOT "H5C_HAVE_PARALLEL" IN_LIST _h5c_defs)
        message(FATAL_ERROR
            "H5CPP_ENABLE_PARALLEL requires an h5c built with "
            "H5C_ENABLE_PARALLEL=ON. Point CMAKE_PREFIX_PATH at a parallel h5c, "
            "or configure the sibling checkout with -DH5C_ENABLE_PARALLEL=ON.")
    endif()
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# ---------------------------------------------------------------------
# library
# ---------------------------------------------------------------------

# Header-only for now: the wrapper is templates plus inline glue. It is still
# a real target so dependents get the include path and h5c transitively, and
# so a compiled unit can be added later without changing user CMake code.
add_library(h5cpp INTERFACE)
add_library(h5cpp::h5cpp ALIAS h5cpp)

target_include_directories(h5cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# HDF5 include and library paths are carried by h5c; never repeat them here.
target_link_libraries(h5cpp INTERFACE h5c::h5c)

target_compile_features(h5cpp INTERFACE
    $<IF:$<STREQUAL:${H5CPP_CXX_STANDARD},20>,cxx_std_20,cxx_std_17>)

# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------

if(H5CPP_BUILD_TESTS)
    add_subdirectory(test)
endif()

# ---------------------------------------------------------------------
# install / export
# ---------------------------------------------------------------------

install(TARGETS h5cpp EXPORT h5cppTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT h5cppTargets
    FILE h5cppTargets.cmake
    NAMESPACE h5cpp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/h5cpp
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/h5cppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/h5cppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/h5cpp
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/h5cppConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/h5cppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/h5cppConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/h5cpp
)
