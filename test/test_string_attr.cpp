// Strings and attributes.
//
// Both are metadata-shaped APIs whose failure modes are quiet: a string read
// back truncated, or an attribute silently attached to the wrong object, looks
// like valid data. So every check here pins the exact value, and the values are
// deliberately distinct per object so a mixed-up target cannot pass.
#include "h5cpp_test.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

const char* kPath = "test_string_attr.h5";

/// Runs `fn` and reports the status it threw, or H5C_OK if it did not throw.
template <class F>
h5c_status_t status_of(F&& fn)
{
    try {
        fn();
    } catch (const h5cpp::error& e) {
        return e.status();
    }
    return H5C_OK;
}

// ---------------------------------------------------------------------------
// strings
// ---------------------------------------------------------------------------

void test_string_fixed()
{
    h5cpp::file f(kPath, h5cpp::mode::truncate);

    f.write_string("/title", "example field");
    H5CPP_ASSERT(f.read_string("/title") == "example field",
                 "fixed-length string round-trip: got '%s'",
                 f.read_string("/title").c_str());

    // The dataset really is a string, and a scalar one.
    const h5cpp::dataset_info meta = f.info("/title");
    H5CPP_ASSERT(meta.type == H5C_STRING, "stored type is H5C_STRING, got %d",
                 static_cast<int>(meta.type));
    H5CPP_ASSERT(meta.rank() == 0, "a string dataset is scalar, rank=%d",
                 meta.rank());

    // Without replace an existing dataset is written IN PLACE, which requires
    // the extents and type to match -- and for a fixed-length string the
    // length is part of the type. So an equal-length value succeeds ...
    f.write_string("/title", "EXAMPLE FIELD");
    H5CPP_ASSERT(f.read_string("/title") == "EXAMPLE FIELD",
                 "equal-length in-place write: got '%s'",
                 f.read_string("/title").c_str());

    // ... and a different length is refused rather than truncated or padded.
    for (const char* other : { "short", "much longer than the original" }) {
        const h5c_status_t st =
            status_of([&] { f.write_string("/title", other); });
        H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                     "write_string '%s' without replace: got %s", other,
                     h5c_status_string(st));
        H5CPP_ASSERT(f.read_string("/title") == "EXAMPLE FIELD",
                     "a refused write must not change the value: got '%s'",
                     f.read_string("/title").c_str());
    }

    f.close();
}

/// Replacing with a LONGER value is the interesting case: a fixed-length
/// dataset is sized from the first value, so growing it means the type has to
/// be recreated rather than reused.
void test_string_replace_longer()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_string("/label", "ab");
    H5CPP_ASSERT(f.read_string("/label") == "ab", "short value");

    f.write_string("/label", "abcdefghijklmnop", /*replace=*/true);
    H5CPP_ASSERT(f.read_string("/label") == "abcdefghijklmnop",
                 "replacing with a longer value: got '%s'",
                 f.read_string("/label").c_str());

    // ... and shrinking again must not leave the old tail behind.
    f.write_string("/label", "xy", /*replace=*/true);
    H5CPP_ASSERT(f.read_string("/label") == "xy",
                 "replacing with a shorter value: got '%s'",
                 f.read_string("/label").c_str());

    f.close();
}

void test_string_empty()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_string("/empty", "");
    H5CPP_ASSERT(f.read_string("/empty").empty(),
                 "empty string round-trip: got '%s'",
                 f.read_string("/empty").c_str());

    // An empty value must still be replaceable by a real one.
    f.write_string("/empty", "now set", /*replace=*/true);
    H5CPP_ASSERT(f.read_string("/empty") == "now set",
                 "replacing an empty string");

    f.close();
}

/// vlen strings are read by the same read_string(); only the stored
/// representation differs (h5fortran cannot read this one).
void test_string_vlen()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_string("/vlen/one", "variable length value",
                   /*replace=*/false, /*vlen=*/true);
    H5CPP_ASSERT(f.read_string("/vlen/one") == "variable length value",
                 "vlen string round-trip: got '%s'",
                 f.read_string("/vlen/one").c_str());

    f.write_string("/vlen/one", "", /*replace=*/true, /*vlen=*/true);
    H5CPP_ASSERT(f.read_string("/vlen/one").empty(),
                 "empty vlen string: got '%s'",
                 f.read_string("/vlen/one").c_str());

    f.write_string("/vlen/one", "a much longer replacement value",
                   /*replace=*/true, /*vlen=*/true);
    H5CPP_ASSERT(f.read_string("/vlen/one") == "a much longer replacement value",
                 "replacing a vlen string with a longer value: got '%s'",
                 f.read_string("/vlen/one").c_str());

    // Unlike the fixed-length form, a vlen dataset is never written in place:
    // without replace an existing path is refused outright, whatever the
    // lengths. Pinned here because the two forms differ.
    h5c_status_t st = status_of([&] {
        f.write_string("/vlen/one", "a much longer replacement value",
                       /*replace=*/false, /*vlen=*/true);
    });
    H5CPP_ASSERT(st == H5C_ERR_EXISTS,
                 "vlen write without replace, equal length: got %s",
                 h5c_status_string(st));
    H5CPP_ASSERT(f.read_string("/vlen/one") == "a much longer replacement value",
                 "a refused vlen write must not change the value");

    st = status_of([&] { (void)f.read_string("/vlen/nope"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "missing string: got %s",
                 h5c_status_string(st));

    f.close();
}

// ---------------------------------------------------------------------------
// attributes
// ---------------------------------------------------------------------------

/// Attributes on a dataset, on a group and on the root, all named the same but
/// carrying different values: attaching one to the wrong object cannot pass.
void test_attr_targets()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    // Asymmetric shape {3,2}, never {2,2}: a transposed write would show up.
    const std::vector<double> values = { 1, 2, 3, 4, 5, 6 };
    f.write("/mesh/coords", values, {3, 2});

    f.write_attr_str("/mesh/coords", "where", "dataset");
    f.write_attr_str("/mesh", "where", "group");
    f.write_attr_str("/", "where", "root");

    H5CPP_ASSERT(f.read_attr_str("/mesh/coords", "where") == "dataset",
                 "dataset attribute: got '%s'",
                 f.read_attr_str("/mesh/coords", "where").c_str());
    H5CPP_ASSERT(f.read_attr_str("/mesh", "where") == "group",
                 "group attribute: got '%s'",
                 f.read_attr_str("/mesh", "where").c_str());
    H5CPP_ASSERT(f.read_attr_str("/", "where") == "root",
                 "root attribute: got '%s'",
                 f.read_attr_str("/", "where").c_str());

    H5CPP_ASSERT(f.attr_exists("/mesh/coords", "where"), "attr_exists dataset");
    H5CPP_ASSERT(f.attr_exists("/mesh", "where"), "attr_exists group");
    H5CPP_ASSERT(f.attr_exists("/", "where"), "attr_exists root");
    H5CPP_ASSERT(!f.attr_exists("/mesh/coords", "absent"),
                 "attr_exists on a missing attribute");
    H5CPP_ASSERT(!f.attr_exists("/no/such/object", "where"),
                 "attr_exists on a missing object");

    // Writing the attribute must not disturb the data it annotates.
    H5CPP_ASSERT(f.read<double>("/mesh/coords") == values,
                 "data survives attribute writes");
    H5CPP_ASSERT(f.info("/mesh/coords").dims == (h5cpp::shape{3, 2}),
                 "shape survives attribute writes");

    // Empty attribute values are legal.
    f.write_attr_str("/mesh/coords", "blank", "");
    H5CPP_ASSERT(f.read_attr_str("/mesh/coords", "blank").empty(),
                 "empty string attribute");

    f.close();
}

/// Every type type_of<> supports, on the same object, with distinct values so
/// a type mix-up cannot read back the right number.
void test_attr_scalar_types()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_attr_scalar<double>("/mesh/coords", "time", 0.125);
    f.write_attr_scalar<float>("/mesh/coords", "dt", 0.5f);
    f.write_attr_scalar<std::int32_t>("/mesh/coords", "step", -1234567);
    f.write_attr_scalar<std::int64_t>("/mesh/coords", "cycle",
                                      9007199254740993LL);
    f.write_attr_scalar<h5c_bool_t>("/mesh/coords", "active", H5C_TRUE);

    H5CPP_ASSERT(f.read_attr_scalar<double>("/mesh/coords", "time") == 0.125,
                 "double attribute");
    H5CPP_ASSERT(f.read_attr_scalar<float>("/mesh/coords", "dt") == 0.5f,
                 "float attribute");
    H5CPP_ASSERT(f.read_attr_scalar<std::int32_t>("/mesh/coords", "step") ==
                     -1234567,
                 "int32 attribute");
    H5CPP_ASSERT(f.read_attr_scalar<std::int64_t>("/mesh/coords", "cycle") ==
                     9007199254740993LL,
                 "int64 attribute (a value double cannot represent)");
    H5CPP_ASSERT(f.read_attr_scalar<h5c_bool_t>("/mesh/coords", "active") ==
                     H5C_TRUE,
                 "bool attribute");

    // Numeric attributes on a group and on the root too.
    f.write_attr_scalar<std::int32_t>("/mesh", "level", 7);
    f.write_attr_scalar<double>("/", "version", 2.5);
    H5CPP_ASSERT(f.read_attr_scalar<std::int32_t>("/mesh", "level") == 7,
                 "int32 attribute on a group");
    H5CPP_ASSERT(f.read_attr_scalar<double>("/", "version") == 2.5,
                 "double attribute on the root");
    H5CPP_ASSERT(f.attr_length("/mesh/coords", "time") == 1,
                 "scalar attribute length");

    f.close();
}

template <class T>
void check_attr_array_type(h5cpp::file& f, const char* name,
                           const std::vector<T>& values)
{
    for (const char* obj : { "/mesh/coords", "/mesh", "/" }) {
        // There is deliberately no count argument: it comes from values.size().
        f.write_attr_array(obj, name, values);
        const std::vector<T> got = f.read_attr_array<T>(obj, name);
        H5CPP_ASSERT(got.size() == values.size(),
                     "%s on %s: self-sized read has %lu elements, want %lu",
                     name, obj, static_cast<unsigned long>(got.size()),
                     static_cast<unsigned long>(values.size()));
        H5CPP_ASSERT(got == values, "%s on %s: array values differ", name,
                     obj);
        H5CPP_ASSERT(f.attr_length(obj, name) == values.size(),
                     "%s on %s: attribute length differs", name, obj);
    }
}

/// Every numeric type h5cpp supports, on a dataset, group and root.
void test_attr_array_types()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    check_attr_array_type(f, "array_f32",
                          std::vector<float>{ 1.25f, -3.5f, 8.75f });
    check_attr_array_type(f, "array_f64",
                          std::vector<double>{ -2.125, 7.5, 19.25 });
    check_attr_array_type(f, "array_i32",
                          std::vector<std::int32_t>{ -700001, 23, 9000007 });
    check_attr_array_type(
        f, "array_i64",
        std::vector<std::int64_t>{ -9000000001LL, 31, 700000000003LL });
    check_attr_array_type(
        f, "array_bool",
        std::vector<h5c_bool_t>{ H5C_TRUE, H5C_FALSE, H5C_TRUE });

    const std::vector<std::int32_t> first = { 41, -3, 700, 19 };
    const std::vector<std::int32_t> second = { -11, 83 };
    f.write_attr_array("/mesh", "replace_array", first);
    f.write_attr_array("/mesh", "replace_array", second);
    H5CPP_ASSERT(f.attr_length("/mesh", "replace_array") == second.size(),
                 "replaced array attribute length");
    H5CPP_ASSERT(f.read_attr_array<std::int32_t>("/mesh", "replace_array") ==
                     second,
                 "replaced array attribute values");

    const std::vector<double> empty;
    f.write_attr_array("/", "empty_array", empty);
    H5CPP_ASSERT(f.attr_length("/", "empty_array") == 0,
                 "empty array attribute length");
    H5CPP_ASSERT(f.read_attr_array<double>("/", "empty_array").empty(),
                 "empty array attribute round-trip");

    f.close();
}

/// Attributes replace in place: unlike datasets there is no `replace` flag,
/// so a second write of the same name must overwrite rather than fail.
void test_attr_replace()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_attr_str("/mesh/coords", "units", "m");
    f.write_attr_str("/mesh/coords", "units", "millimetre");  // longer
    H5CPP_ASSERT(f.read_attr_str("/mesh/coords", "units") == "millimetre",
                 "replacing a string attribute with a longer value: got '%s'",
                 f.read_attr_str("/mesh/coords", "units").c_str());
    f.write_attr_str("/mesh/coords", "units", "s");           // shorter again
    H5CPP_ASSERT(f.read_attr_str("/mesh/coords", "units") == "s",
                 "replacing a string attribute with a shorter value: got '%s'",
                 f.read_attr_str("/mesh/coords", "units").c_str());

    f.write_attr_scalar<double>("/mesh/coords", "time", 9.75);
    H5CPP_ASSERT(f.read_attr_scalar<double>("/mesh/coords", "time") == 9.75,
                 "replacing a numeric attribute");

    f.close();
}

void test_attr_missing()
{
    h5cpp::file f(kPath, h5cpp::mode::read);

    h5c_status_t st =
        status_of([&] { (void)f.read_attr_str("/mesh/coords", "absent"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "missing string attribute: got %s",
                 h5c_status_string(st));

    st = status_of(
        [&] { (void)f.read_attr_scalar<double>("/mesh/coords", "absent"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "missing numeric attribute: got %s",
                 h5c_status_string(st));

    st = status_of(
        [&] { (void)f.read_attr_array<double>("/mesh/coords", "absent"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "missing array attribute: got %s",
                 h5c_status_string(st));

    // A missing OBJECT is also NOT_FOUND, and the message must name it.
    try {
        (void)f.read_attr_str("/no/such/object", "where");
        H5CPP_FAILF("reading an attribute of a missing object should throw");
    } catch (const h5cpp::error& e) {
        H5CPP_ASSERT(e.status() == H5C_ERR_NOT_FOUND,
                     "missing object: got %s",
                     h5c_status_string(e.status()));
        H5CPP_ASSERT(std::string(e.what()).find("no/such/object") !=
                         std::string::npos,
                     "message names the object: %s", e.what());
    }

    // Attributes cannot be written to a read-only file.
    st = status_of([&] { f.write_attr_str("/mesh/coords", "new", "x"); });
    H5CPP_ASSERT(st != H5C_OK, "attribute write on a read-only file must fail");

    f.close();
}

}  // namespace

H5CPP_TEST_MAIN_STATE;

int main()
{
    test_string_fixed();
    test_string_replace_longer();
    test_string_empty();
    test_string_vlen();
    test_attr_targets();
    test_attr_scalar_types();
    test_attr_array_types();
    test_attr_replace();
    test_attr_missing();

    return H5CPP_TEST_SUMMARY("test_string_attr");
}
