// Lifetime, move semantics and edge-shaped data.
//
// Everything here is about the wrapper itself rather than about HDF5: who owns
// the handle, what a moved-from file is allowed to do, and whether the API
// survives the degenerate shapes that a solver hits at the edge of a domain.
//
// NOTE on type mapping: the guards that make `bool` and other unsupported
// element types fail are static_asserts (type_of<T> and the std::vector<bool>
// checks in write()/read_into()). A compile error cannot be exercised from a
// test that must itself compile, so there is deliberately no case for it here;
// verifying it means uncommenting a line and watching the build fail.
#include "h5cpp_test.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

const char* kPath  = "test_raii.h5";
const char* kPath2 = "test_raii_other.h5";

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

/// Populates both files so the later tests have something to look at.
void setup()
{
    {
        h5cpp::file f(kPath, h5cpp::mode::truncate);
        // Asymmetric {3,2}: a transposed store would be visible.
        f.write("/rank/two", std::vector<double>{ 1, 2, 3, 4, 5, 6 }, {3, 2});
        f.write_scalar<std::int32_t>("/step", 17);
        f.close();
    }
    {
        h5cpp::file f(kPath2, h5cpp::mode::truncate);
        f.write_scalar<std::int32_t>("/step", -99);
        f.close();
    }
}

void test_move_construction()
{
    h5cpp::file a(kPath, h5cpp::mode::read);
    h5c_file_t* raw = a.c_handle();
    H5CPP_ASSERT(raw != nullptr, "an open file has a handle");
    H5CPP_ASSERT(a.hid() >= 0, "an open file has an HDF5 id");

    h5cpp::file b(std::move(a));
    H5CPP_ASSERT(b.c_handle() == raw, "move construction transfers the handle");
    H5CPP_ASSERT(a.c_handle() == nullptr, "moved-from file is empty");
    H5CPP_ASSERT(b.read_scalar<std::int32_t>("/step") == 17,
                 "moved-to file still reads");

    // A moved-from file must be safely closable, and closing it must not touch
    // the handle the moved-to file now owns.
    a.close();
    H5CPP_ASSERT(b.exists("/rank/two"),
                 "closing the moved-from file must not close the moved-to one");
    b.close();
    // Both destructors now run on empty handles: no double close.
}

void test_move_assignment()
{
    h5cpp::file a(kPath, h5cpp::mode::read);
    h5cpp::file b(kPath2, h5cpp::mode::read);
    H5CPP_ASSERT(b.read_scalar<std::int32_t>("/step") == -99, "second file");

    // Assigning over `b` must close what it held and adopt a's handle. The two
    // files carry different values, so adopting the wrong one is detectable.
    h5c_file_t* raw = a.c_handle();
    b = std::move(a);
    H5CPP_ASSERT(b.c_handle() == raw, "move assignment transfers the handle");
    H5CPP_ASSERT(a.c_handle() == nullptr, "moved-from file is empty");
    H5CPP_ASSERT(b.read_scalar<std::int32_t>("/step") == 17,
                 "move-assigned file reads the SOURCE file, got %d",
                 b.read_scalar<std::int32_t>("/step"));

    // Self-move-assignment must not close the handle out from under us.
    h5cpp::file& alias = b;
    b = std::move(alias);
    H5CPP_ASSERT(b.c_handle() == raw, "self-move keeps the handle");
    H5CPP_ASSERT(b.read_scalar<std::int32_t>("/step") == 17,
                 "self-move keeps the file usable");

    b.close();
}

/// Move-assigning INTO a moved-from file, and out of a closed one: both are
/// states an ordinary vector<file> shuffle can produce.
void test_move_edge_states()
{
    h5cpp::file a(kPath, h5cpp::mode::read);
    h5cpp::file moved_from(std::move(a));  // `a` is now empty
    (void)moved_from;

    h5cpp::file c(kPath2, h5cpp::mode::read);
    a = std::move(c);  // assign into the empty one
    H5CPP_ASSERT(a.read_scalar<std::int32_t>("/step") == -99,
                 "assigning into a moved-from file");

    h5cpp::file closed(kPath, h5cpp::mode::read);
    closed.close();
    H5CPP_ASSERT(closed.c_handle() == nullptr, "close() empties the handle");
    h5cpp::file adopted(std::move(closed));
    H5CPP_ASSERT(adopted.c_handle() == nullptr, "moving a closed file");
    adopted.close();  // no-op on an empty handle

    // Files living in a container get moved by the container itself.
    std::vector<h5cpp::file> files;
    files.reserve(1);
    files.push_back(h5cpp::file(kPath, h5cpp::mode::read));
    files.push_back(h5cpp::file(kPath2, h5cpp::mode::read));  // forces a realloc
    H5CPP_ASSERT(files[0].read_scalar<std::int32_t>("/step") == 17,
                 "vector<file> element 0 after reallocation");
    H5CPP_ASSERT(files[1].read_scalar<std::int32_t>("/step") == -99,
                 "vector<file> element 1 after reallocation");
}

void test_close_and_destructor()
{
    h5cpp::file f(kPath, h5cpp::mode::read);
    f.close();
    f.close();
    f.close();  // idempotent, however many times

    // Queries on a closed file must answer, not crash: exists() is documented
    // as never touching the sticky status and returns 0 for an unusable handle.
    H5CPP_ASSERT(!f.exists("/rank/two"), "exists() on a closed file is false");
    H5CPP_ASSERT(!f.attr_exists("/", "anything"),
                 "attr_exists() on a closed file is false");

    // Real work on a closed file must fail cleanly rather than dereference the
    // null handle.
    const h5c_status_t st =
        status_of([&] { (void)f.read_scalar<std::int32_t>("/step"); });
    H5CPP_ASSERT(st == H5C_ERR_INVALID_ARG, "read on a closed file: got %s",
                 h5c_status_string(st));

    // Opening a file that does not exist throws instead of yielding a
    // half-constructed object.
    //
    // An absent file is H5C_ERR_NOT_FOUND, not a generic HDF5 failure, so a
    // caller can tell "create it then" from "it is there but unusable".
    const h5c_status_t missing = status_of(
        [&] { h5cpp::file g("test_raii_no_such_file.h5", h5cpp::mode::read); });
    H5CPP_ASSERT(missing == H5C_ERR_NOT_FOUND, "opening a missing file: got %s",
                 h5c_status_string(missing));
}

/// The destructor must swallow failures: throwing out of one during stack
/// unwinding calls std::terminate. Provoke it by closing the handle behind
/// the object's back, then letting the destructor run.
void test_destructor_does_not_throw()
{
    bool escaped = false;
    try {
        h5cpp::file f(kPath, h5cpp::mode::read);
        f.close();
        // Now let a real exception unwind past a live (empty) file object.
        h5cpp::file g(kPath, h5cpp::mode::read);
        throw std::runtime_error("unwinding past a live h5cpp::file");
    } catch (const std::runtime_error&) {
        escaped = true;
    }
    H5CPP_ASSERT(escaped, "unwinding past a file destructor must be clean");
}

// ---------------------------------------------------------------------------
// degenerate shapes
// ---------------------------------------------------------------------------

/// A zero extent is legal and transfers nothing. Shapes are asymmetric so a
/// {0,3} dataset cannot be confused with a {3,0} one.
void test_zero_extents()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    const std::vector<double> none;
    f.write("/empty/rows", none, {0, 3});
    f.write("/empty/cols", none, {3, 0});
    f.write("/empty/one", none);

    const h5cpp::dataset_info rows = f.info("/empty/rows");
    H5CPP_ASSERT(rows.dims == (h5cpp::shape{0, 3}), "{0,3} shape");
    H5CPP_ASSERT(rows.count == 0, "{0,3} count");
    H5CPP_ASSERT(f.info("/empty/cols").dims == (h5cpp::shape{3, 0}),
                 "{3,0} shape must not be normalised to {0,3}");
    H5CPP_ASSERT(f.info("/empty/one").dims == (h5cpp::shape{0}), "{0} shape");

    H5CPP_ASSERT(f.read<double>("/empty/rows").empty(), "reading {0,3}");
    H5CPP_ASSERT(f.read<double>("/empty/cols").empty(), "reading {3,0}");
    H5CPP_ASSERT(f.read<double>("/empty/one").empty(), "reading {0}");

    // The container overload's length check must accept an empty buffer for a
    // zero-element shape, and still reject a non-empty one.
    std::vector<double> sink;
    f.read_into("/empty/rows", sink, {0, 3});
    const h5c_status_t st =
        status_of([&] { f.write("/empty/bad", std::vector<double>{1.0}, {0, 3}); });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "a non-empty buffer for a zero shape: got %s",
                 h5c_status_string(st));

    // And the extents still have to match on read: {0,3} is not {3,0}.
    const h5c_status_t swapped =
        status_of([&] { f.read_into("/empty/rows", sink, {3, 0}); });
    H5CPP_ASSERT(swapped == H5C_ERR_SHAPE_MISMATCH,
                 "{3,0} must not read {0,3}: got %s",
                 h5c_status_string(swapped));

    f.close();
}

/// A scalar is rank 0, not rank 1 of length 1.
void test_scalar_shape()
{
    h5cpp::file f(kPath, h5cpp::mode::readwrite);

    f.write_scalar<double>("/scalar/one", -0.75);
    const h5cpp::dataset_info meta = f.info("/scalar/one");
    H5CPP_ASSERT(meta.rank() == 0, "a scalar has rank 0, got %d", meta.rank());
    H5CPP_ASSERT(meta.dims.empty(), "a scalar has no extents");
    H5CPP_ASSERT(meta.count == 1, "a scalar holds one element");
    H5CPP_ASSERT(f.read_scalar<double>("/scalar/one") == -0.75,
                 "scalar round-trip");

    // Reading it as a 1-element vector must be refused: the ranks differ.
    std::vector<double> one(1);
    const h5c_status_t st =
        status_of([&] { f.read_into("/scalar/one", one, {1}); });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "rank 1 must not read a scalar: got %s",
                 h5c_status_string(st));

    // A rank-3 shape, asymmetric in every extent, to check that nothing is
    // hard-coded for ranks 0..2.
    const std::vector<std::int32_t> cube = { 1,  2,  3,  4,  5,  6,
                                             7,  8,  9,  10, 11, 12,
                                             13, 14, 15, 16, 17, 18,
                                             19, 20, 21, 22, 23, 24 };
    f.write("/rank/three", cube, {2, 3, 4});
    H5CPP_ASSERT(f.info("/rank/three").dims == (h5cpp::shape{2, 3, 4}),
                 "rank-3 shape must be {2,3,4}");
    H5CPP_ASSERT(f.read<std::int32_t>("/rank/three") == cube,
                 "rank-3 round-trip (row-major, last extent fastest)");

    std::vector<std::int32_t> got(24);
    const h5c_status_t rot =
        status_of([&] { f.read_into("/rank/three", got, {4, 3, 2}); });
    H5CPP_ASSERT(rot == H5C_ERR_SHAPE_MISMATCH,
                 "a rotated shape of the same size must be refused: got %s",
                 h5c_status_string(rot));

    f.close();
}

}  // namespace

H5CPP_TEST_MAIN_STATE;

int main()
{
    setup();
    test_move_construction();
    test_move_assignment();
    test_move_edge_states();
    test_close_and_destructor();
    test_destructor_does_not_throw();
    test_zero_extents();
    test_scalar_shape();

    return H5CPP_TEST_SUMMARY("test_raii");
}
