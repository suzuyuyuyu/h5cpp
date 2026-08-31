# Test labels reflect WHERE a test may be run.
#
#   ctest -L quick   serial only. Safe to run directly on a login node.
#   ctest -L mpi     launches mpiexec. MUST be submitted through the batch
#                    system (scripts/run-mpi-tests.sh); never run on a login
#                    node, regardless of the rank count.
#
# `mpi` is deliberately NOT part of `quick`, so that the obvious command a
# developer types on a login node cannot start an MPI job by accident.
add_library(h5cpp_test_support INTERFACE)
target_include_directories(h5cpp_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(h5cpp_test_support INTERFACE h5cpp::h5cpp)

# Tests are discovered, so adding a test_*.cpp file needs no edit here.
# Parallel tests are named test_p*.cpp and are built only with MPI.
file(GLOB h5cpp_all_tests CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

set(h5cpp_serial_tests ${h5cpp_all_tests})
list(FILTER h5cpp_serial_tests EXCLUDE REGEX "/test_p[^/]*\\.cpp$")

set(h5cpp_parallel_tests ${h5cpp_all_tests})
list(FILTER h5cpp_parallel_tests INCLUDE REGEX "/test_p[^/]*\\.cpp$")

foreach(src IN LISTS h5cpp_serial_tests)
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE h5cpp_test_support)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "quick")
endforeach()

# Number of ranks used when the MPI tests are run from a job script.
set(H5CPP_TEST_NRANKS 2 CACHE STRING "MPI ranks for the parallel tests")

if(H5CPP_ENABLE_PARALLEL)
    foreach(src IN LISTS h5cpp_parallel_tests)
        get_filename_component(name "${src}" NAME_WE)
        add_executable(${name} "${src}")
        target_link_libraries(${name} PRIVATE h5cpp_test_support MPI::MPI_CXX)
        add_test(NAME ${name}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${H5CPP_TEST_NRANKS}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(${name} PROPERTIES
            LABELS "mpi"
            PROCESSORS ${H5CPP_TEST_NRANKS})
    endforeach()
endif()
