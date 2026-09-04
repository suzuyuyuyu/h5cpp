// Serial interleaved (multi-component) I/O.
//
// The stored layout is [n, ncomp] with the component index varying fastest,
// because that is what XDMF needs to see a vector. The components here have
// DISTINCT magnitudes (1s, 10s, 100s) and n != ncomp, so a swapped layout, a
// transposed write or a shuffled scatter cannot reproduce them by accident.
#include "h5cpp_test.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

const char* kPath = "test_interleaved.h5";

// n = 4 points, ncomp = 3: never square, so [4,3] and [3,4] are telling apart.
const std::vector<double> kU = { 1, 2, 3, 4 };
const std::vector<double> kV = { 10, 20, 30, 40 };
const std::vector<double> kW = { -100, -200, -300, -400 };

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

/// Reads the dataset as a plain [n, 3] array and checks it element by element
/// against the interleaved order. This is the check that actually pins the
/// LAYOUT; read_interleaved() alone would agree with a component-major file.
void check_stored_layout(h5cpp::file& f, const std::string& path)
{
    const h5cpp::dataset_info meta = f.info(path);
    H5CPP_ASSERT(meta.dims == (h5cpp::shape{kU.size(), 3}),
                 "%s stored shape must be {%lu,3}", path.c_str(),
                 static_cast<unsigned long>(kU.size()));

    const std::vector<double> flat = f.read<double>(path);
    H5CPP_ASSERT(flat.size() == kU.size() * 3, "%s flat size", path.c_str());
    for (std::size_t i = 0; i < kU.size() && (i + 1) * 3 <= flat.size(); ++i) {
        H5CPP_ASSERT(flat[i * 3 + 0] == kU[i], "%s[%lu,0]: got %g want %g",
                     path.c_str(), static_cast<unsigned long>(i),
                     flat[i * 3 + 0], kU[i]);
        H5CPP_ASSERT(flat[i * 3 + 1] == kV[i], "%s[%lu,1]: got %g want %g",
                     path.c_str(), static_cast<unsigned long>(i),
                     flat[i * 3 + 1], kV[i]);
        H5CPP_ASSERT(flat[i * 3 + 2] == kW[i], "%s[%lu,2]: got %g want %g",
                     path.c_str(), static_cast<unsigned long>(i),
                     flat[i * 3 + 2], kW[i]);
    }
}

/// The view overload: n is derived from the components.
void test_view_overload()
{
    h5cpp::file f(kPath, h5cpp::mode::truncate);

    f.write_interleaved<double>("/fields/velocity", { kU, kV, kW });
    check_stored_layout(f, "/fields/velocity");

    std::vector<double> gu(kU.size(), 1e9);
    std::vector<double> gv(kV.size(), 1e9);
    std::vector<double> gw(kW.size(), 1e9);
    f.read_interleaved<double>("/fields/velocity", { gu, gv, gw });
    H5CPP_ASSERT(gu == kU, "u scatter");
    H5CPP_ASSERT(gv == kV, "v scatter");
    H5CPP_ASSERT(gw == kW, "w scatter");

    f.close();
}

/// The pointer + n overload must produce exactly the same file.
void test_pointer_overload()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    const std::vector<const double*> in = { kU.data(), kV.data(), kW.data() };
    f.write_interleaved<double>("/fields/by_ptr", in, kU.size());
    check_stored_layout(f, "/fields/by_ptr");

    std::vector<double> gu(kU.size(), 1e9);
    std::vector<double> gv(kV.size(), 1e9);
    std::vector<double> gw(kW.size(), 1e9);
    const std::vector<double*> out = { gu.data(), gv.data(), gw.data() };
    f.read_interleaved<double>("/fields/by_ptr", out, kU.size());
    H5CPP_ASSERT(gu == kU, "u scatter (pointer overload)");
    H5CPP_ASSERT(gv == kV, "v scatter (pointer overload)");
    H5CPP_ASSERT(gw == kW, "w scatter (pointer overload)");

    // Both overloads wrote the same bytes.
    H5CPP_ASSERT(f.read<double>("/fields/by_ptr") ==
                     f.read<double>("/fields/velocity"),
                 "the two write overloads must agree");

    f.close();
}

void test_read_component()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    H5CPP_ASSERT(f.read_component<double>("/fields/velocity", 0) == kU,
                 "read_component 0");
    H5CPP_ASSERT(f.read_component<double>("/fields/velocity", 1) == kV,
                 "read_component 1");
    H5CPP_ASSERT(f.read_component<double>("/fields/velocity", 2) == kW,
                 "read_component 2");

    // Out of range must fail, not read past the last component.
    const h5c_status_t st =
        status_of([&] { (void)f.read_component<double>("/fields/velocity", 3); });
    H5CPP_ASSERT(st != H5C_OK, "component index 3 of a 3-component dataset");

    // read_component() only makes sense on an [n, ncomp] dataset; h5cpp
    // rejects other ranks before touching h5c.
    f.write("/plain/one", kU);
    const h5c_status_t st1 =
        status_of([&] { (void)f.read_component<double>("/plain/one", 0); });
    H5CPP_ASSERT(st1 == H5C_ERR_SHAPE_MISMATCH,
                 "read_component on a 1-D dataset: got %s",
                 h5c_status_string(st1));

    f.close();
}

/// The whole reason the view overloads exist: ragged components must be
/// rejected BEFORE any I/O, since the C API cannot verify `n`.
void test_ragged_is_rejected_before_io()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    const std::vector<double> shorter = { 1, 2 };  // 2, not 4

    // Ragged in the LAST position ...
    h5c_status_t st = status_of(
        [&] { f.write_interleaved<double>("/bad/tail", { kU, kV, shorter }); });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH, "ragged tail: got %s",
                 h5c_status_string(st));
    H5CPP_ASSERT(!f.exists("/bad/tail"),
                 "a ragged write must not create '/bad/tail'");

    // ... and in the FIRST, where the derived n itself comes from the odd one.
    st = status_of(
        [&] { f.write_interleaved<double>("/bad/head", { shorter, kV, kW }); });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH, "ragged head: got %s",
                 h5c_status_string(st));
    H5CPP_ASSERT(!f.exists("/bad/head"),
                 "a ragged write must not create '/bad/head'");

    // The same guard on the read side, where a short buffer would otherwise
    // be overrun.
    std::vector<double> gu(kU.size()), gv(kV.size()), gshort(2);
    st = status_of([&] {
        f.read_interleaved<double>("/fields/velocity", { gu, gv, gshort });
    });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH, "ragged read: got %s",
                 h5c_status_string(st));

    // No components at all is an argument error, not an empty write.
    const std::vector<h5cpp::span<const double>> none;
    st = status_of([&] { f.write_interleaved<double>("/bad/none", none); });
    H5CPP_ASSERT(st == H5C_ERR_INVALID_ARG, "zero components: got %s",
                 h5c_status_string(st));
    H5CPP_ASSERT(!f.exists("/bad/none"),
                 "a componentless write must not create '/bad/none'");

    f.close();
}

/// n == 0 is legal: it describes a field with no points.
///
/// With no rows there is nothing to dereference, so NULL components are legal.
/// That matters in C++ because `std::vector<T>{}.data()` is normally NULL, so
/// the natural spelling of an empty field
///
///     f.write_interleaved<double>("/x", { none, none, none });   // empty
///
/// must work -- and it must work identically in serial and in parallel, where
/// a rank owning an empty block passes exactly this.
void test_zero_length()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    const std::vector<double> none;
    f.write_interleaved<double>("/fields/null", { none, none, none });

    const h5cpp::dataset_info null_meta = f.info("/fields/null");
    H5CPP_ASSERT(null_meta.dims == (h5cpp::shape{0, 3}),
                 "empty components must still give a {0,3} dataset");

    // Zero length with non-NULL pointers must behave the same way.
    const std::vector<double> storage = { 0, 0, 0, 0 };
    const h5cpp::span<const double> empty(storage.data(), 0);
    f.write_interleaved<double>("/fields/empty", { empty, empty, empty });

    const h5cpp::dataset_info meta = f.info("/fields/empty");
    H5CPP_ASSERT(meta.dims == (h5cpp::shape{0, 3}),
                 "empty interleaved shape must be {0,3}");
    H5CPP_ASSERT(meta.count == 0, "empty interleaved count");
    H5CPP_ASSERT(f.read<double>("/fields/empty").empty(),
                 "reading an empty interleaved dataset");

    std::vector<double> sink = { 1, 2, 3, 4 };
    const h5cpp::span<double> out(sink.data(), 0);
    f.read_interleaved<double>("/fields/empty", { out, out, out });
    H5CPP_ASSERT(sink == (std::vector<double>{ 1, 2, 3, 4 }),
                 "an empty scatter must not touch the buffers");
    // read_component() sizes its vector from the stored shape, so on an empty
    // dataset it passes a NULL buffer with n == 0. That must succeed too.
    H5CPP_ASSERT(f.read_component<double>("/fields/empty", 0).empty(),
                 "read_component of an empty dataset yields an empty vector");

    f.close();
}

/// Interleaved data of the other supported types, and 2 components rather
/// than 3, so ncomp is not baked in anywhere.
void test_other_types_and_ncomp()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    const std::vector<std::int32_t> a = { 1, 2, 3, 4, 5 };       // n = 5
    const std::vector<std::int32_t> b = { -10, -20, -30, -40, -50 };
    f.write_interleaved<std::int32_t>("/fields/pair", { a, b });

    H5CPP_ASSERT(f.info("/fields/pair").dims == (h5cpp::shape{5, 2}),
                 "2-component shape must be {5,2}");
    const std::vector<std::int32_t> flat = f.read<std::int32_t>("/fields/pair");
    for (std::size_t i = 0; i < a.size(); ++i) {
        H5CPP_ASSERT(flat[i * 2 + 0] == a[i] && flat[i * 2 + 1] == b[i],
                     "int32 interleaved row %lu",
                     static_cast<unsigned long>(i));
    }
    H5CPP_ASSERT(f.read_component<std::int32_t>("/fields/pair", 1) == b,
                 "int32 read_component 1");

    const std::vector<float> p = { 0.5f, 1.5f, 2.5f };
    const std::vector<float> q = { -0.25f, -1.25f, -2.25f };
    f.write_interleaved<float>("/fields/f32", { p, q });
    H5CPP_ASSERT(f.read_component<float>("/fields/f32", 0) == p,
                 "float read_component 0");
    H5CPP_ASSERT(f.read_component<float>("/fields/f32", 1) == q,
                 "float read_component 1");

    // Replacing an interleaved dataset with a different n needs replace=true,
    // exactly as for a plain dataset.
    const std::vector<std::int32_t> a2 = { 9, 8, 7 };
    const std::vector<std::int32_t> b2 = { -9, -8, -7 };
    const h5c_status_t st = status_of(
        [&] { f.write_interleaved<std::int32_t>("/fields/pair", { a2, b2 }); });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "shrinking without replace: got %s", h5c_status_string(st));

    f.write_interleaved<std::int32_t>("/fields/pair", { a2, b2 },
                                      /*replace=*/true);
    H5CPP_ASSERT(f.info("/fields/pair").dims == (h5cpp::shape{3, 2}),
                 "replaced shape must be {3,2}");
    H5CPP_ASSERT(f.read_component<std::int32_t>("/fields/pair", 0) == a2,
                 "replaced component 0");

    f.close();
}

/// Interleaved data is what carries the XDMF "attribute_type" annotation, so
/// the two APIs are used together.
void test_annotated()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_attr_str("/fields/velocity", "attribute_type", "Vector");
    H5CPP_ASSERT(f.read_attr_str("/fields/velocity", "attribute_type") ==
                     "Vector",
                 "attribute on an interleaved dataset");
    check_stored_layout(f, "/fields/velocity");

    f.close();
}

}  // namespace

H5CPP_TEST_MAIN_STATE;

int main()
{
    test_view_overload();
    test_pointer_overload();
    test_read_component();
    test_ragged_is_rejected_before_io();
    test_zero_length();
    test_other_types_and_ncomp();
    test_annotated();

    return H5CPP_TEST_SUMMARY("test_interleaved");
}
