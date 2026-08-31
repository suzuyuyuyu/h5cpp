// Dependency-free assertions for the h5cpp tests, mirroring h5c/test/h5c_test.h.
#ifndef H5CPP_TEST_HPP
#define H5CPP_TEST_HPP

#include <cstdio>

#include "h5cpp/h5cpp.hpp"

extern int h5cpp_test_failures;

#define H5CPP_FAILF(...)                                                      \
    do {                                                                      \
        std::fprintf(stderr, "  FAIL %s:%d: ", __FILE__, __LINE__);           \
        std::fprintf(stderr, __VA_ARGS__);                                    \
        std::fputc('\n', stderr);                                             \
        h5cpp_test_failures++;                                                \
    } while (0)

#define H5CPP_ASSERT(cond, ...)                                               \
    do {                                                                      \
        if (!(cond)) {                                                        \
            H5CPP_FAILF(__VA_ARGS__);                                         \
        }                                                                     \
    } while (0)

/// Asserts that `expr` throws h5cpp::error.
#define H5CPP_THROWS(expr, what)                                              \
    do {                                                                      \
        bool threw_ = false;                                                  \
        try {                                                                 \
            (void)(expr);                                                     \
        } catch (const h5cpp::error&) {                                       \
            threw_ = true;                                                    \
        }                                                                     \
        if (!threw_) {                                                        \
            H5CPP_FAILF("%s: expected h5cpp::error from %s", (what), #expr);   \
        }                                                                     \
    } while (0)

#define H5CPP_TEST_MAIN_STATE int h5cpp_test_failures = 0

#define H5CPP_TEST_SUMMARY(name)                                              \
    (h5cpp_test_failures == 0                                                 \
         ? (std::printf("%s: all checks passed\n", (name)), 0)                \
         : (std::fprintf(stderr, "%s: %d check(s) failed\n", (name),          \
                         h5cpp_test_failures), 1))

#endif  // H5CPP_TEST_HPP
