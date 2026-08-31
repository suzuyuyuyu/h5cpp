// Parallel round-trip tests for the h5cpp MPI wrapper. Run under mpiexec with
// 2 ranks by `ctest -L quick`.
//
// Local extents, shapes and values are deliberately ASYMMETRIC: no two ranks
// own the same number of rows, the non-split extent is 3, and every element
// encodes both its rank and its position, so a transposed or mis-offset
// implementation cannot pass by accident.
//
// Every test is bounded by a watchdog alarm: a collective call that fails to
// agree across ranks must surface as a FAILURE, never as a hung job. Any
// rank's failure fails the whole program.
#define _POSIX_C_SOURCE 200809L

#include "h5cpp_test.hpp"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "h5cpp/h5cpp_mpi.hpp"

namespace {

const char* kPath = "test_pbasic.h5";

// Wall-clock budget for the whole program; a stall becomes a failed test.
const unsigned kWatchdogSeconds = 120;

int g_me     = 0;
int g_nprocs = 1;

extern "C" void on_watchdog(int)
{
    std::fprintf(stderr,
                 "test_pbasic: WATCHDOG fired on rank %d "
                 "(likely a collective call that did not agree)\n", g_me);
    std::fflush(stderr);
    _exit(2);
}

/// Local extents chosen so that no two ranks agree.
std::size_t local_rows(int base)
{
    return static_cast<std::size_t>(base + g_me);
}

/// Partition boundaries, computed independently of h5c/h5cpp.
std::vector<long long> expected_partition(std::size_t nlocal)
{
    std::vector<long long> part(static_cast<std::size_t>(g_nprocs) + 1, 0);
    long long mine = static_cast<long long>(nlocal);

    MPI_Allgather(&mine, 1, MPI_LONG_LONG, part.data() + 1, 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD);
    for (int r = 0; r < g_nprocs; ++r) {
        part[static_cast<std::size_t>(r) + 1] += part[static_cast<std::size_t>(r)];
    }
    return part;
}

/// Runs `fn` and reports the h5cpp status it threw, or H5C_OK if it did not.
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

/// The written boundaries must match the ones we computed ourselves.
///
/// This goes through partition() and local_block() rather than opening
/// "<path>/__partition__": the accessors are the supported way to reach the
/// layout, so the test exercises what users are meant to call.
void check_partition(h5cpp::parallel_file& f, const std::string& path,
                     std::size_t nlocal)
{
    const std::vector<long long> want = expected_partition(nlocal);
    const std::size_t            len  = want.size();

    const std::vector<std::int64_t> got = f.partition(path);
    H5CPP_ASSERT(got.size() == len,
                 "%s partition length: got %lu want %lu", path.c_str(),
                 static_cast<unsigned long>(got.size()),
                 static_cast<unsigned long>(len));

    for (std::size_t r = 0; r < got.size() && r < len; ++r) {
        H5CPP_ASSERT(static_cast<long long>(got[r]) == want[r],
                     "%s partition[%lu]: got %lld want %lld", path.c_str(),
                     static_cast<unsigned long>(r),
                     static_cast<long long>(got[r]), want[r]);
    }

    /* And this rank's own slice must agree with those boundaries. */
    const h5cpp::block_extent mine = f.local_block(path);
    H5CPP_ASSERT(mine.offset == static_cast<std::size_t>(want[g_me]),
                 "%s local offset: got %lu want %lld", path.c_str(),
                 static_cast<unsigned long>(mine.offset), want[g_me]);
    H5CPP_ASSERT(mine.count == nlocal,
                 "%s local rows: got %lu want %lu", path.c_str(),
                 static_cast<unsigned long>(mine.count),
                 static_cast<unsigned long>(nlocal));
}

// ---------------------------------------------------------------------------

void test_modes(h5cpp::parallel_file& f)
{
    H5CPP_ASSERT(f.is_collective(), "collective is the default");
    H5CPP_ASSERT(f.comm() != MPI_COMM_NULL, "comm() returned MPI_COMM_NULL");

    f.set_collective(false);
    H5CPP_ASSERT(!f.is_collective(), "independent was not selected");

    // Independent transfers must still round-trip.
    const std::size_t nlocal = local_rows(1);
    std::vector<double> src(nlocal);
    for (std::size_t i = 0; i < nlocal; ++i) {
        src[i] = 3.5 * (g_me + 1) + static_cast<double>(i);
    }
    f.write("/dist/indep", src);
    H5CPP_ASSERT(f.read<double>("/dist/indep") == src,
                 "independent round-trip");

    f.set_collective(true);
    H5CPP_ASSERT(f.is_collective(), "collective was not restored");
}

void test_1d(h5cpp::parallel_file& f)
{
    const std::size_t            nlocal = local_rows(4);  // 4, 5, 6, ...
    const std::vector<long long> part   = expected_partition(nlocal);
    const std::size_t            total  = static_cast<std::size_t>(part[g_nprocs]);

    std::vector<double> src(nlocal);
    for (std::size_t i = 0; i < nlocal; ++i) {
        src[i] = 1000.0 * g_me + static_cast<double>(i) + 0.25;
    }

    f.write("/dist/one", src);
    check_partition(f, "/dist/one", nlocal);

    const h5cpp::distributed_info meta = f.info("/dist/one");
    H5CPP_ASSERT(meta.local.dims == h5cpp::shape{nlocal},
                 "1d local shape");
    H5CPP_ASSERT(meta.local.count == nlocal, "1d local count");
    H5CPP_ASSERT(meta.global.dims == h5cpp::shape{total}, "1d global shape");
    H5CPP_ASSERT(meta.global.type == H5C_F64, "1d type");

    H5CPP_ASSERT(f.read<double>("/dist/one") == src, "1d round-trip");

    // The block must sit at this rank's offset, not at the file start.
    std::vector<double> whole(total);
    f.read_replicated("/dist/one/data", whole, {total});
    const std::size_t base = static_cast<std::size_t>(part[g_me]);
    for (std::size_t i = 0; i < nlocal; ++i) {
        H5CPP_ASSERT(whole[base + i] == src[i], "1d global[%lu]: got %g want %g",
                     static_cast<unsigned long>(base + i), whole[base + i],
                     src[i]);
    }
}

void test_2d(h5cpp::parallel_file& f)
{
    const std::size_t            nlocal = local_rows(2);  // 2, 3, 4, ...
    const h5cpp::shape           ldims  = { nlocal, 3 };
    const std::vector<long long> part   = expected_partition(nlocal);
    const std::size_t            total  = static_cast<std::size_t>(part[g_nprocs]);

    // Row-major: element (i, j) lives at i*3 + j. Values encode both.
    std::vector<double> src(nlocal * 3);
    for (std::size_t i = 0; i < nlocal; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            src[i * 3 + j] = 100.0 * g_me + 10.0 * static_cast<double>(i) +
                             static_cast<double>(j);
        }
    }

    f.write("/dist/two", src, ldims);
    check_partition(f, "/dist/two", nlocal);

    const h5cpp::distributed_info meta = f.info("/dist/two");
    H5CPP_ASSERT(meta.local.dims == ldims, "2d local shape");
    H5CPP_ASSERT(meta.local.count == nlocal * 3, "2d local count");
    H5CPP_ASSERT(meta.global.dims == (h5cpp::shape{total, 3}),
                 "2d global shape");

    std::vector<double> got(nlocal * 3, 0.0);
    f.read_into("/dist/two", got, ldims);
    H5CPP_ASSERT(got == src, "2d round-trip");

    // A block placed at the wrong offset would show up here.
    std::vector<double> whole(total * 3);
    f.read_replicated("/dist/two/data", whole, {total, 3});
    const std::size_t base = static_cast<std::size_t>(part[g_me]);
    for (std::size_t i = 0; i < nlocal; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            const std::size_t k = (base + i) * 3 + j;
            H5CPP_ASSERT(whole[k] == src[i * 3 + j],
                         "2d global[%lu]: got %g want %g",
                         static_cast<unsigned long>(k), whole[k],
                         src[i * 3 + j]);
        }
    }
}

/// An empty rank still joins every collective call.
void test_zero_extent(h5cpp::parallel_file& f)
{
    const std::size_t  nlocal = (g_me == 0) ? 5 : 0;
    const h5cpp::shape ldims  = { nlocal, 3 };

    std::vector<std::int32_t> src(nlocal * 3);
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<std::int32_t>(7 + 2 * i);
    }

    f.write("/dist/sparse", src, ldims);
    check_partition(f, "/dist/sparse", nlocal);

    const h5cpp::distributed_info meta = f.info("/dist/sparse");
    H5CPP_ASSERT(meta.local.dims[0] == nlocal, "sparse local dims[0]");
    H5CPP_ASSERT(meta.global.dims[0] == 5, "sparse global dims[0] must be 5");

    // read() sizes itself from __partition__, so the empty rank gets nothing.
    const std::vector<std::int32_t> got = f.read<std::int32_t>("/dist/sparse");
    H5CPP_ASSERT(got.size() == nlocal * 3, "sparse local size: %lu",
                 static_cast<unsigned long>(got.size()));
    H5CPP_ASSERT(got == src, "sparse round-trip");
}

void test_interleaved(h5cpp::parallel_file& f)
{
    const std::size_t            n    = local_rows(3);  // 3, 4, 5, ...
    const std::vector<long long> part = expected_partition(n);
    const std::size_t total = static_cast<std::size_t>(part[g_nprocs]);

    std::vector<double> u(n), v(n), w(n);
    for (std::size_t i = 0; i < n; ++i) {
        u[i] = 100.0 * g_me + static_cast<double>(i);
        v[i] = 100.0 * g_me + static_cast<double>(i) + 0.5;
        w[i] = -(100.0 * g_me + static_cast<double>(i));
    }

    f.write_interleaved<double>("/dist/velocity", { u, v, w });
    check_partition(f, "/dist/velocity", n);

    const h5cpp::distributed_info meta = f.info("/dist/velocity");
    H5CPP_ASSERT(meta.local.dims == (h5cpp::shape{n, 3}),
                 "interleaved local shape");
    H5CPP_ASSERT(meta.global.dims == (h5cpp::shape{total, 3}),
                 "interleaved global shape");

    std::vector<double> gu(n, 12345.0), gv(n, 12345.0), gw(n, 12345.0);
    f.read_interleaved<double>("/dist/velocity", { gu, gv, gw });
    H5CPP_ASSERT(gu == u, "interleaved u round-trip");
    H5CPP_ASSERT(gv == v, "interleaved v round-trip");
    H5CPP_ASSERT(gw == w, "interleaved w round-trip");

    // The file really is [total, 3] interleaved, not three blocks.
    std::vector<double> whole(total * 3);
    f.read_replicated("/dist/velocity/data", whole, {total, 3});
    const std::size_t base = static_cast<std::size_t>(part[g_me]);
    for (std::size_t i = 0; i < n; ++i) {
        H5CPP_ASSERT(whole[(base + i) * 3 + 0] == u[i],
                     "interleaved u at row %lu",
                     static_cast<unsigned long>(base + i));
        H5CPP_ASSERT(whole[(base + i) * 3 + 1] == v[i],
                     "interleaved v at row %lu",
                     static_cast<unsigned long>(base + i));
        H5CPP_ASSERT(whole[(base + i) * 3 + 2] == w[i],
                     "interleaved w at row %lu",
                     static_cast<unsigned long>(base + i));
    }

    // Components of unequal length are caught in h5cpp, on every rank alike.
    std::vector<double> shortv(n > 0 ? n - 1 : 0);
    const h5c_status_t st = status_of([&] {
        f.write_interleaved<double>("/dist/ragged", { u, v, shortv });
    });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "ragged components: got %s", h5c_status_string(st));
    assert_same_status(st, "ragged components");

    // Zero local extent, expressed with empty views.
    const std::vector<double> none;
    f.write_interleaved<double>("/dist/nothing", { none, none, none });
    const h5cpp::distributed_info empty = f.info("/dist/nothing");
    H5CPP_ASSERT(empty.global.dims == (h5cpp::shape{0, 3}),
                 "empty interleaved global shape");
}

/// A disagreement across ranks must fail on EVERY rank with the same status,
/// without any rank walking into a collective HDF5 call alone. The watchdog
/// turns a deadlock into a failure.
void test_shape_mismatch(h5cpp::parallel_file& f)
{
    if (g_nprocs < 2) {
        return;  // nothing to disagree about
    }
    const std::vector<double> buf(2 * static_cast<std::size_t>(3 + g_me), 0.0);
    const h5cpp::shape ldims = { 2, static_cast<std::size_t>(3 + g_me) };

    h5c_status_t st = status_of([&] { f.write("/dist/bad", buf, ldims); });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "mismatched non-split dim: got %s, expected shape mismatch",
                 h5c_status_string(st));
    assert_same_status(st, "mismatched non-split dim");
    H5CPP_ASSERT(!f.exists("/dist/bad"),
                 "a rejected write must not leave '/dist/bad' behind");

    // A rank (dimensionality) disagreement must also be caught everywhere.
    st = status_of([&] {
        if (g_me == 0) {
            f.write("/dist/bad2", buf.data(), ldims);
        } else {
            f.write("/dist/bad2", buf.data(), h5cpp::shape{2});
        }
    });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "mismatched dimensionality: got %s", h5c_status_string(st));
    assert_same_status(st, "mismatched dimensionality");

    // A local buffer-length check fires before h5c is reached. Every rank does
    // it identically here, which is the only safe way to use a local check in
    // a collective program.
    st = status_of([&] {
        const std::vector<double> two(2);
        f.write("/dist/short", two, h5cpp::shape{2, 3});
    });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH,
                 "buffer length is checked: got %s", h5c_status_string(st));
    assert_same_status(st, "buffer length");
}

/// Reopening read-only exercises the read path against a closed file.
void test_reopen_read()
{
    h5cpp::parallel_file f(kPath, h5cpp::mode::read, MPI_COMM_WORLD,
                           MPI_INFO_NULL);
    const std::size_t nlocal = local_rows(4);

    const std::vector<double> got = f.read<double>("/dist/one");
    H5CPP_ASSERT(got.size() == nlocal, "reopen local size");
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double want = 1000.0 * g_me + static_cast<double>(i) + 0.25;
        H5CPP_ASSERT(got[i] == want, "reopen 1d[%lu]: got %g want %g",
                     static_cast<unsigned long>(i), got[i], want);
    }

    // Writing to a read-only file must fail identically on every rank.
    const h5c_status_t st = status_of([&] { f.write("/dist/nope", got); });
    H5CPP_ASSERT(st == H5C_ERR_STATE, "read-only write: got %s",
                 h5c_status_string(st));
    assert_same_status(st, "read-only write");

    f.close();
    f.close();  // idempotent
}

void test_move_semantics()
{
    h5cpp::parallel_file a(kPath, h5cpp::mode::read);
    h5cpp::parallel_file b(std::move(a));

    H5CPP_ASSERT(b.exists("/dist/one"), "moved-to file works");
    H5CPP_ASSERT(a.c_handle() == nullptr, "moved-from file is empty");
    b.close();
    // The destructor of the moved-from file must not double close.
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
        test_modes(f);
        test_1d(f);
        test_2d(f);
        test_zero_extent(f);
        test_interleaved(f);
        test_shape_mismatch(f);
        f.close();
    }
    test_reopen_read();
    test_move_semantics();

    alarm(0);

    // Any rank's failure must fail the whole test.
    int total = 0;
    MPI_Allreduce(&h5cpp_test_failures, &total, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    h5cpp_test_failures = total;
    if (g_me != 0) {
        // Only rank 0 prints the summary; every rank keeps the exit code.
        MPI_Finalize();
        return (total == 0) ? 0 : 1;
    }
    MPI_Finalize();
    return H5CPP_TEST_SUMMARY("test_pbasic");
}
