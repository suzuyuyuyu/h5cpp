cmake_minimum_required(VERSION 3.20)
project(h5cpp_examples LANGUAGES CXX)

# Standalone on purpose: the library build never references the examples.
# Consuming h5cpp through find_package() here is what a real user writes.
#
#   cmake -S example -B example/build/intel \
#         -DCMAKE_PREFIX_PATH="<h5cpp prefix>;<h5c prefix>;<hdf5 prefix>"
#   cmake --build example/build/intel --target examples
#
# When h5cpp is not installed, fall back to the sibling source tree.
find_package(h5cpp CONFIG QUIET)
if(NOT h5cpp_FOUND)
    message(STATUS "h5cpp not installed; building it from ${CMAKE_CURRENT_SOURCE_DIR}/..")
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(H5CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(".." "${CMAKE_CURRENT_BINARY_DIR}/h5cpp")
endif()

set(h5cpp_example_targets)

function(h5cpp_add_example target source)
    add_executable("${target}" "${source}")
    target_link_libraries("${target}" PRIVATE h5cpp::h5cpp)
    list(APPEND h5cpp_example_targets "${target}")
    set(h5cpp_example_targets "${h5cpp_example_targets}" PARENT_SCOPE)
endfunction()

h5cpp_add_example(example_serial             serial/main.cpp)
h5cpp_add_example(example_serial_interleaved serial-interleaved/main.cpp)

# The parallel example needs an h5c built with H5C_ENABLE_PARALLEL=ON, which
# puts H5C_HAVE_PARALLEL on the transitively linked h5c target.
get_target_property(_h5c_defs h5c::h5c INTERFACE_COMPILE_DEFINITIONS)
if(_h5c_defs AND "H5C_HAVE_PARALLEL" IN_LIST _h5c_defs)
    find_package(MPI REQUIRED COMPONENTS CXX)
    h5cpp_add_example(example_parallel parallel/main.cpp)
    target_link_libraries(example_parallel PRIVATE MPI::MPI_CXX)
    h5cpp_add_example(example_parallel_interleaved parallel-interleaved/main.cpp)
    target_link_libraries(example_parallel_interleaved PRIVATE MPI::MPI_CXX)
else()
    message(STATUS "h5c has no parallel support; skipping example_parallel")
endif()

add_custom_target(examples DEPENDS ${h5cpp_example_targets})
