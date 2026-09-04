// Attributes on a parallel_file, plus the partition accessors checked against
// a decomposition computed independently with MPI_Allgather.
//
// Attribute writes on a parallel file are collective HDF5 metadata operations:
// every rank calls them with the same value. What must be verified is that the
// value actually lands on the intended object -- annotating "<path>" instead of
// "<path>/data" is the mistake this file is here to catch -- and that every
// rank reads the same thing back.
//
// Local extents are ASYMMETRIC (no two ranks own the same number of rows, one
// rank may own none) and every value encodes its rank, so a mis-offset or
// transposed implementation cannot pass.
//
// A watchdog bounds the whole program: a collective call that fails to agree
// must surface as a FAILURE, never as a hung job. Any rank's failure fails the
// whole program.
#define _POSIX_C_SOURCE 200809L

#include "h5cpp_test.hpp"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "h5cpp/h5cpp_mpi.hpp"

namespace {

const char* kPath = "test_pattr.h5";

const unsigned kWatchdogSeconds = 120;

int g_me     = 0;
int g_nprocs = 1;

extern "C" void on_watchdog(int)
{
    std::fprintf(stderr,
                 "test_pattr: WATCHDOG fired on rank %d "
                 "(likely a collective call that did not agree)\n", g_me);
    std::fflush(stderr);
    _exit(2);
}

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

/// Asserts that every rank observed the same status. A one-sided early return
/// is exactly what would deadlock, so it must be checked, not assumed.
void assert_same_status(h5c_status_t st, const char* what)
{
    int mine = static_cast<int>(st);
    int lo = 0, hi = 0;
    MPI_Allreduce(&mine, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mine, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    H5CPP_ASSERT(lo == hi, "%s: ranks returned different statuses (%d..%d)",
                 what, lo, hi);
}

/// Asserts that every rank read the same string. Compared by length and by a
/// simple checksum, which is enough to catch a per-rank difference.
void assert_same_string(const std::string& s, const char* what)
{
    long long mine[2] = { static_cast<long long>(s.size()), 0 };
    for (unsigned char c : s) {
        mine[1] = mine[1] * 131 + c;
    }
    long long lo[2] = { 0, 0 };
    long long hi[2] = { 0, 0 };
    MPI_Allreduce(mine, lo, 2, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(mine, hi, 2, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    H5CPP_ASSERT(lo[0] == hi[0] && lo[1] == hi[1],
                 "%s: ranks read different values (rank %d has '%s')", what,
                 g_me, s.c_str());
}

/// Partition boundaries, computed independently of h5c/h5cpp.
std::vector<long long> expected_partition(std::size_t nlocal)
{
    std::vector<long long> part(static_cast<std::size_t>(g_nprocs) + 1, 0);
    long long mine = static_cast<long long>(nlocal);

    MPI_Allgather(&mine, 1, MPI_LONG_LONG, part.data() + 1, 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    for (int r = 0; r < g_nprocs; ++r) {
        part[static_cast<std::size_t>(r) + 1] +=
            part[static_cast<std::size_t>(r)];
    }
    return part;
}

/// partition() and local_block() must agree with that independent answer.
void check_partition(h5cpp::parallel_file& f, const std::string& path,
                     std::size_t nlocal)
{
    const std::vector<long long> want = expected_partition(nlocal);

    const std::vector<std::int64_t> got = f.partition(path);
    H5CPP_ASSERT(got.size() == want.size(),
                 "%s partition length: got %lu want %lu", path.c_str(),
                 static_cast<unsigned long>(got.size()),
                 static_cast<unsigned long>(want.size()));
    H5CPP_ASSERT(!got.empty() && got.front() == 0,
                 "%s partition must start at 0", path.c_str());

    for (std::size_t r = 0; r < got.size() && r < want.size(); ++r) {
        H5CPP_ASSERT(static_cast<long long>(got[r]) == want[r],
                     "%s partition[%lu]: got %lld want %lld", path.c_str(),
                     static_cast<unsigned long>(r),
                     static_cast<long long>(got[r]), want[r]);
        if (r > 0) {
            H5CPP_ASSERT(got[r] >= got[r - 1],
                         "%s partition must be non-decreasing at %lu",
                         path.c_str(), static_cast<unsigned long>(r));
        }
    }

    const h5cpp::block_extent mine = f.local_block(path);
    H5CPP_ASSERT(mine.offset == static_cast<std::size_t>(want[g_me]),
                 "%s local offset: got %lu want %lld", path.c_str(),
                 static_cast<unsigned long>(mine.offset), want[g_me]);
    H5CPP_ASSERT(mine.count == nlocal,
                 "%s local rows: got %lu want %lu", path.c_str(),
                 static_cast<unsigned long>(mine.count),
                 static_cast<unsigned long>(nlocal));
    H5CPP_ASSERT(mine.offset + mine.count ==
                     static_cast<std::size_t>(want[g_me + 1]),
                 "%s local end: got %lu want %lld", path.c_str(),
                 static_cast<unsigned long>(mine.offset + mine.count),
                 want[g_me + 1]);

    // The global extent must be the last boundary, not this rank's count.
    const h5cpp::distributed_info meta = f.info(path);
    H5CPP_ASSERT(meta.global.dims[0] ==
                     static_cast<std::size_t>(want[g_nprocs]),
                 "%s global rows: got %lu want %lld", path.c_str(),
                 static_cast<unsigned long>(meta.global.dims[0]),
                 want[g_nprocs]);
}

/// This rank's local row count. Rank 0 owns nothing when there is more than
/// one rank, so the empty-block case is always exercised.
std::size_t local_rows()
{
    return (g_nprocs > 1 && g_me == 0) ? 0u
                                       : static_cast<std::size_t>(2 + g_me);
}

// ---------------------------------------------------------------------------

/// Writes the distributed field the attribute tests annotate, and checks the
/// layout while we are here.
void write_field(h5cpp::parallel_file& f)
{
    const std::size_t n = local_rows();

    std::vector<double> u(n), v(n), w(n);
    for (std::size_t i = 0; i < n; ++i) {
        u[i] = 1000.0 * g_me + static_cast<double>(i);
        v[i] = 2000.0 * g_me + static_cast<double>(i);
        w[i] = -(3000.0 * g_me + static_cast<double>(i));
    }

    f.write_interleaved<double>("/fields/velocity", { u, v, w });
    check_partition(f, "/fields/velocity", n);

    const h5cpp::distributed_info meta = f.info("/fields/velocity");
    H5CPP_ASSERT(meta.local.dims == (h5cpp::shape{n, 3}),
                 "local shape must be {%lu,3}",
                 static_cast<unsigned long>(n));
    H5CPP_ASSERT(meta.global.dims.size() == 2 && meta.global.dims[1] == 3,
                 "global shape must be {total,3}");

    // A plain distributed array too, with a different decomposition, so the
    // partition accessors are exercised on more than one layout.
    const std::size_t m = static_cast<std::size_t>(g_me) + 1;
    std::vector<std::int32_t> rows(m * 4);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            rows[i * 4 + j] = static_cast<std::int32_t>(100 * g_me + 10 * i + j);
        }
    }
    f.write("/fields/rows", rows, {m, 4});
    check_partition(f, "/fields/rows", m);
    H5CPP_ASSERT(f.read<std::int32_t>("/fields/rows") == rows,
                 "plain distributed round-trip");
}

/// Attributes on the dataset, on the enclosing group and on the root, all
/// named the same and all carrying different values: an attribute attached to
/// the wrong object cannot pass.
void test_attr_targets(h5cpp::parallel_file& f)
{
    f.write_attr_str("/fields/velocity/data", "where", "dataset");
    f.write_attr_str("/fields/velocity", "where", "group");
    f.write_attr_str("/", "where", "root");

    const std::string on_data = f.read_attr_str("/fields/velocity/data", "where");
    const std::string on_grp  = f.read_attr_str("/fields/velocity", "where");
    const std::string on_root = f.read_attr_str("/", "where");

    H5CPP_ASSERT(on_data == "dataset", "attribute on <path>/data: got '%s'",
                 on_data.c_str());
    H5CPP_ASSERT(on_grp == "group", "attribute on the group: got '%s'",
                 on_grp.c_str());
    H5CPP_ASSERT(on_root == "root", "attribute on the root: got '%s'",
                 on_root.c_str());

    // Every rank must see the same metadata.
    assert_same_string(on_data, "dataset attribute");
    assert_same_string(on_grp, "group attribute");
    assert_same_string(on_root, "root attribute");

    H5CPP_ASSERT(f.attr_exists("/fields/velocity/data", "where"),
                 "attr_exists on the dataset");
    H5CPP_ASSERT(f.attr_exists("/fields/velocity", "where"),
                 "attr_exists on the group");
    H5CPP_ASSERT(f.attr_exists("/", "where"), "attr_exists on the root");
    H5CPP_ASSERT(!f.attr_exists("/fields/velocity/data", "absent"),
                 "attr_exists on a missing attribute");
    H5CPP_ASSERT(!f.attr_exists("/no/such/object", "where"),
                 "attr_exists on a missing object");

    // The XDMF annotation this API exists for.
    f.write_attr_str("/fields/velocity/data", "attribute_type", "Vector");
    H5CPP_ASSERT(f.read_attr_str("/fields/velocity/data", "attribute_type") ==
                     "Vector",
                 "attribute_type on the interleaved dataset");

    // Annotating must not disturb the data or the partition.
    check_partition(f, "/fields/velocity", local_rows());
}

void test_attr_scalar_types(h5cpp::parallel_file& f)
{
    const char* obj = "/fields/velocity/data";

    f.write_attr_scalar<double>(obj, "time", 0.125);
    f.write_attr_scalar<float>(obj, "dt", 0.5f);
    f.write_attr_scalar<std::int32_t>(obj, "step", -1234567);
    f.write_attr_scalar<std::int64_t>(obj, "cycle", 9007199254740993LL);
    f.write_attr_scalar<h5c_bool_t>(obj, "active", H5C_TRUE);

    H5CPP_ASSERT(f.read_attr_scalar<double>(obj, "time") == 0.125,
                 "double attribute");
    H5CPP_ASSERT(f.read_attr_scalar<float>(obj, "dt") == 0.5f,
                 "float attribute");
    H5CPP_ASSERT(f.read_attr_scalar<std::int32_t>(obj, "step") == -1234567,
                 "int32 attribute");
    H5CPP_ASSERT(f.read_attr_scalar<std::int64_t>(obj, "cycle") ==
                     9007199254740993LL,
                 "int64 attribute (a value double cannot represent)");
    H5CPP_ASSERT(f.read_attr_scalar<h5c_bool_t>(obj, "active") == H5C_TRUE,
                 "bool attribute");

    // Numeric attributes on a group and on the root too.
    f.write_attr_scalar<std::int32_t>("/fields", "level", 7);
    f.write_attr_scalar<double>("/", "version", 2.5);
    H5CPP_ASSERT(f.read_attr_scalar<std::int32_t>("/fields", "level") == 7,
                 "int32 attribute on a group");
    H5CPP_ASSERT(f.read_attr_scalar<double>("/", "version") == 2.5,
                 "double attribute on the root");

    // Same value on every rank.
    long long step = f.read_attr_scalar<std::int32_t>(obj, "step");
    long long lo = 0, hi = 0;
    MPI_Allreduce(&step, &lo, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&step, &hi, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    H5CPP_ASSERT(lo == hi, "ranks read different int32 attributes (%lld..%lld)",
                 lo, hi);
}

/// Attributes have no `replace` flag: writing the same name again overwrites.
void test_attr_replace(h5cpp::parallel_file& f)
{
    const char* obj = "/fields/velocity/data";

    f.write_attr_str(obj, "units", "m");
    f.write_attr_str(obj, "units", "millimetre");  // longer
    H5CPP_ASSERT(f.read_attr_str(obj, "units") == "millimetre",
                 "replacing a string attribute with a longer value: got '%s'",
                 f.read_attr_str(obj, "units").c_str());
    f.write_attr_str(obj, "units", "s");           // shorter again
    H5CPP_ASSERT(f.read_attr_str(obj, "units") == "s",
                 "replacing a string attribute with a shorter value: got '%s'",
                 f.read_attr_str(obj, "units").c_str());
    assert_same_string(f.read_attr_str(obj, "units"), "replaced attribute");

    f.write_attr_scalar<double>(obj, "time", 9.75);
    H5CPP_ASSERT(f.read_attr_scalar<double>(obj, "time") == 9.75,
                 "replacing a numeric attribute");
}

/// A missing attribute must fail on EVERY rank with the same status: a rank
/// that returned early would walk into the next collective call alone.
void test_attr_missing(h5cpp::parallel_file& f)
{
    const char* obj = "/fields/velocity/data";

    h5c_status_t st = status_of([&] { (void)f.read_attr_str(obj, "absent"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "missing string attribute: got %s",
                 h5c_status_string(st));
    assert_same_status(st, "missing string attribute");

    st = status_of([&] { (void)f.read_attr_scalar<double>(obj, "absent"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "missing numeric attribute: got %s",
                 h5c_status_string(st));
    assert_same_status(st, "missing numeric attribute");

    st = status_of([&] { (void)f.read_attr_str("/no/such/object", "where"); });
    H5CPP_ASSERT(st == H5C_ERR_NOT_FOUND, "attribute of a missing object: %s",
                 h5c_status_string(st));
    assert_same_status(st, "attribute of a missing object");
}

/// The partition accessors must also work on a file opened read-only, where
/// nothing has been written in this session.
void test_reopen_read()
{
    h5cpp::parallel_file f(kPath, h5cpp::mode::read, MPI_COMM_WORLD,
                           MPI_INFO_NULL);

    check_partition(f, "/fields/velocity", local_rows());

    const std::string units = f.read_attr_str("/fields/velocity/data", "units");
    H5CPP_ASSERT(units == "s", "attribute survives close/reopen: got '%s'",
                 units.c_str());
    assert_same_string(units, "reopened attribute");

    H5CPP_ASSERT(f.read_attr_scalar<double>("/", "version") == 2.5,
                 "root attribute survives close/reopen");

    // The data really is at this rank's offset in the global array.
    const std::size_t n = local_rows();
    const h5cpp::block_extent mine = f.local_block("/fields/velocity");
    const std::size_t total = f.info("/fields/velocity").global.dims[0];

    std::vector<double> whole(total * 3);
    f.read_replicated("/fields/velocity/data", whole, {total, 3});
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = mine.offset + i;
        H5CPP_ASSERT(whole[row * 3 + 0] == 1000.0 * g_me + static_cast<double>(i),
                     "u at global row %lu", static_cast<unsigned long>(row));
        H5CPP_ASSERT(whole[row * 3 + 1] == 2000.0 * g_me + static_cast<double>(i),
                     "v at global row %lu", static_cast<unsigned long>(row));
        H5CPP_ASSERT(whole[row * 3 + 2] ==
                         -(3000.0 * g_me + static_cast<double>(i)),
                     "w at global row %lu", static_cast<unsigned long>(row));
    }

    // Writing an attribute to a read-only file must fail everywhere alike.
    const h5c_status_t st = status_of(
        [&] { f.write_attr_str("/fields/velocity/data", "nope", "x"); });
    H5CPP_ASSERT(st != H5C_OK, "attribute write on a read-only parallel file");
    assert_same_status(st, "read-only attribute write");

    f.close();
    f.close();  // idempotent
}

/// Move assignment on a parallel_file, and the moved-from object staying safe.
/// Not collective in itself, but every rank does the same thing.
void test_move_assignment()
{
    h5cpp::parallel_file a(kPath, h5cpp::mode::read);
    h5cpp::parallel_file b(kPath, h5cpp::mode::read);

    h5c_file_t* raw = a.c_handle();
    b = std::move(a);  // closes b's handle, adopts a's
    H5CPP_ASSERT(b.c_handle() == raw, "move assignment transfers the handle");
    H5CPP_ASSERT(a.c_handle() == nullptr, "moved-from file is empty");
    H5CPP_ASSERT(b.attr_exists("/", "version"), "move-assigned file works");

    h5cpp::parallel_file& alias = b;
    b = std::move(alias);  // self-move must not close the handle
    H5CPP_ASSERT(b.c_handle() == raw, "self-move keeps the handle");
    H5CPP_ASSERT(b.attr_exists("/", "version"), "self-move keeps it usable");

    a.close();  // no-op on an empty handle
    b.close();
    // The moved-from destructor must not double close.
}

}  // namespace

H5CPP_TEST_MAIN_STATE;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_me);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    std::signal(SIGALRM, on_watchdog);
    alarm(kWatchdogSeconds);

    {
        h5cpp::parallel_file f(kPath, h5cpp::mode::truncate);
        write_field(f);
        test_attr_targets(f);
        test_attr_scalar_types(f);
        test_attr_replace(f);
        test_attr_missing(f);
        f.close();
    }
    test_reopen_read();
    test_move_assignment();

    alarm(0);

    // Any rank's failure must fail the whole test.
    int total = 0;
    MPI_Allreduce(&h5cpp_test_failures, &total, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    h5cpp_test_failures = total;
    if (g_me != 0) {
        MPI_Finalize();
        return (total == 0) ? 0 : 1;
    }
    MPI_Finalize();
    return H5CPP_TEST_SUMMARY("test_pattr");
}
