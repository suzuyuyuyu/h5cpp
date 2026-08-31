// Distributed read and write.
//
// Each rank owns a slice of the slowest-varying axis; h5c concatenates them
// into one dataset and records the rank boundaries beside it.
//
// MUST be run through the batch system, never on a login node:
//     sbatch example/run-parallel-example.sh
#include <cstdint>
#include <h5cpp/h5cpp_mpi.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int me = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int status = 0;
    try {
        const std::string path = "example_parallel.h5";

        // Deliberately unequal: rank r owns 2 + r rows. A rank may own 0.
        const std::size_t nlocal = static_cast<std::size_t>(2 + me);
        const h5cpp::shape dims = {nlocal, 3};  // dims[0] is the split axis

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < nprocs; ++i) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (me == i) std::cout << "rank " << me << ": local size = " << nlocal << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> local(nlocal * 3);
        for (std::size_t i = 0; i < local.size(); ++i) {
            local[i] = 100.0 * me + static_cast<double>(i);
        }

        // Every call below is collective: all ranks, same order, same path.
        // A bad argument on one rank becomes the same exception on all of
        // them rather than a deadlock, because h5c agrees the status first.
        {
            h5cpp::parallel_file f(path, h5cpp::mode::truncate);
            f.write("/coords", local, dims);

            // Where did this rank's block land? No need to touch
            // __partition__ directly.
            const h5cpp::block_extent mine = f.local_block("/coords");
            std::cout << "rank " << me << ": rows [" << mine.offset << ", "
                      << mine.offset + mine.count << ")\n"
                      << std::flush;

            f.close();
        }

        {
            h5cpp::parallel_file f(path, h5cpp::mode::read);

            // read() sizes itself from the stored boundaries.
            const std::vector<double> got = f.read<double>("/coords");
            if (got != local) {
                std::cerr << "rank " << me << ": round trip differs\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Both of these are COLLECTIVE, so every rank must call them.
            // Guarding the call itself with `if (me == 0)` would deadlock --
            // guard only the printing.
            const h5cpp::distributed_info meta = f.info("/coords");
            const std::vector<std::int64_t> part = f.partition("/coords");

            if (me == 0) {
                std::cout << "global shape: {" << meta.global.dims[0] << ", "
                          << meta.global.dims[1] << "} from " << nprocs
                          << " ranks\n";
                std::cout << "partition:";
                for (std::int64_t b : part) {
                    std::cout << ' ' << b;
                }
                std::cout << '\n';
            }
            f.close();
        }

        if (me == 0) {
            std::cout << "wrote and read " << path << '\n';
        }
    } catch (const h5cpp::error& e) {
        std::cerr << "rank " << me << " h5cpp error: " << e.what() << '\n';
        status = 1;
    }

    // Any rank's failure must fail the whole run.
    int total = 0;
    MPI_Allreduce(&status, &total, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return total;
}
