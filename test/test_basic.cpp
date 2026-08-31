// Round-trip tests for the h5cpp core wrapper.
//
// Shapes and values are deliberately ASYMMETRIC so that a transposed or
// mis-strided implementation cannot reproduce them by accident.
#include "h5cpp_test.hpp"

#include <string>
#include <vector>

namespace {

const char* kPath = "test_basic.h5";

// Mirrors h5fortran's r64_2d(2,3): Fortran (2,3) == C shape {3, 2}.
const std::vector<double> k2d   = { 1, 2, 3, 4, 5, 6 };
const h5cpp::shape        k2dsh = { 3, 2 };

void test_scalar_and_vector()
{
    h5cpp::file f(kPath, h5cpp::mode::truncate);

    f.write_scalar<double>("/scalar/r64", 42.5);
    H5CPP_ASSERT(f.read_scalar<double>("/scalar/r64") == 42.5, "scalar round-trip");

    const std::vector<double> v = { 1.5, -2.5, 3.25 };
    f.write("/rank/one", v);
    H5CPP_ASSERT(f.read<double>("/rank/one") == v, "1-D round-trip");

    f.close();
}

void test_shape_and_order()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write("/rank/two", k2d, k2dsh);

    const h5cpp::dataset_info meta = f.info("/rank/two");
    H5CPP_ASSERT(meta.rank() == 2, "rank of /rank/two");
    H5CPP_ASSERT(meta.dims == k2dsh, "stored shape must be {3,2}, not {2,3}");
    H5CPP_ASSERT(meta.count == 6, "element count");
    H5CPP_ASSERT(meta.type == H5C_F64, "stored type");

    H5CPP_ASSERT(f.read<double>("/rank/two") == k2d, "2-D flat round-trip");

    // A swapped shape must be rejected, never silently reinterpreted.
    std::vector<double> six(6);
    H5CPP_THROWS(f.read_into<double>("/rank/two", six.data(), {2, 3}),
                 "swapped shape is rejected");

    f.close();
}

void test_types()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    const std::vector<std::int32_t> i32 = { 10, -20, 30, -40 };
    const std::vector<std::int64_t> i64 = { 1, -2000000000LL, 4000000000LL };
    const std::vector<float>        f32 = { 0.5f, -1.5f };
    const std::vector<h5c_bool_t>   flags = { H5C_TRUE, H5C_FALSE,
                                              H5C_FALSE, H5C_TRUE };

    f.write("/types/i32", i32, {2, 2});
    f.write("/types/i64", i64);
    f.write("/types/f32", f32);
    f.write("/types/bool", flags, {2, 2});

    H5CPP_ASSERT(f.read<std::int32_t>("/types/i32") == i32, "int32 round-trip");
    H5CPP_ASSERT(f.read<std::int64_t>("/types/i64") == i64, "int64 round-trip");
    H5CPP_ASSERT(f.read<float>("/types/f32") == f32, "float round-trip");
    H5CPP_ASSERT(f.read<h5c_bool_t>("/types/bool") == flags, "bool round-trip");

    f.close();
}

void test_errors_and_raii()
{
    h5cpp::file f(kPath, h5cpp::mode::read);

    // Missing datasets throw with a usable status.
    try {
        (void)f.read<double>("/does/not/exist");
        H5CPP_FAILF("reading a missing dataset should throw");
    } catch (const h5cpp::error& e) {
        H5CPP_ASSERT(e.status() == H5C_ERR_NOT_FOUND,
                     "status is NOT_FOUND, got %s",
                     h5c_status_string(e.status()));
        H5CPP_ASSERT(std::string(e.what()).find("does/not/exist") !=
                         std::string::npos,
                     "message names the path: %s", e.what());
    }

    // Writing to a read-only file throws rather than corrupting state.
    H5CPP_THROWS(f.write_scalar<double>("/nope", 1.0), "read-only write throws");

    // A container whose length disagrees with the shape is caught in h5cpp.
    H5CPP_ASSERT(f.exists("/rank/two"), "exists() on a real dataset");
    H5CPP_ASSERT(!f.exists("/nope"), "exists() on a missing dataset");

    // close() after earlier failures must succeed: every failure already threw.
    f.close();
    f.close();  // idempotent
}

void test_move_semantics()
{
    h5cpp::file a(kPath, h5cpp::mode::read);
    h5cpp::file b(std::move(a));
    H5CPP_ASSERT(b.exists("/rank/two"), "moved-to file works");
    H5CPP_ASSERT(a.c_handle() == nullptr, "moved-from file is empty");

    // Destructor of a moved-from file must not double close.
}

void test_shape_mismatch_is_caught_early()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);
    std::vector<double> three = { 1, 2, 3 };

    // 3 elements cannot fill a {2,2} shape; h5cpp rejects it before h5c.
    H5CPP_THROWS(f.write("/bad", three, {2, 2}), "container length is checked");

    std::vector<double> small(2);
    H5CPP_THROWS(f.read_into("/rank/two", small, k2dsh),
                 "read buffer length is checked");
    f.close();
}

}  // namespace

H5CPP_TEST_MAIN_STATE;

int main()
{
    test_scalar_and_vector();
    test_shape_and_order();
    test_types();
    test_errors_and_raii();
    test_move_semantics();
    test_shape_mismatch_is_caught_early();

    return H5CPP_TEST_SUMMARY("test_basic");
}
