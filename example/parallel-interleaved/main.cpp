// Distributed vector field: the combination a solver actually needs.
//
// Components live in separate arrays (u, v, w) because that is what suits a
// GPU kernel, the points are split across ranks, and the file has to come out
// as one [total_n, 3] dataset so that XDMF sees a vector.
//
// h5cpp takes the components as views, so this rank's point count is derived
// rather than passed -- the one number the C API cannot check.
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
        const std::string path = "example_parallel_interleaved.h5";

        // Rank 0 deliberately owns nothing when there is more than one rank.
        // An empty subdomain is legal: the rank contributes no rows but still
        // takes part in every collective call, and empty views say so.
        const std::size_t nlocal =
            (nprocs > 1 && me == 0) ? 0u : static_cast<std::size_t>(2 + me);

        std::vector<double> u(nlocal), v(nlocal), w(nlocal);
        for (std::size_t i = 0; i < nlocal; ++i) {
            // Every value encodes its rank and position, so a mis-offset
            // write cannot look right by accident.
            u[i] = 1000.0 * me + static_cast<double>(i);
            v[i] = 2000.0 * me + static_cast<double>(i);
            w[i] = 3000.0 * me + static_cast<double>(i);
        }

        // ---- write ---------------------------------------------------
        {
            h5cpp::parallel_file f(path, h5cpp::mode::truncate);

            f.write_interleaved<double>("/fields/velocity", {u, v, w});

            // The dataset lives at "<path>/data"; the group also holds
            // __partition__. XDMF needs this attribute to treat the three
            // components as a vector.
            f.write_attr_str("/fields/velocity/data", "attribute_type", "Vector");

            const h5cpp::block_extent mine = f.local_block("/fields/velocity");
            std::cout << "rank " << me << ": " << mine.count
                      << " points at rows [" << mine.offset << ", "
                      << mine.offset + mine.count << ")\n"
                      << std::flush;

            f.close();
        }

        // ---- read back -----------------------------------------------
        {
            h5cpp::parallel_file f(path, h5cpp::mode::read);

            std::vector<double> gu(nlocal), gv(nlocal), gw(nlocal);
            f.read_interleaved<double>("/fields/velocity", {gu, gv, gw});

            if (gu != u || gv != v || gw != w) {
                std::cerr << "rank " << me << ": round trip differs\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Collective: every rank calls it, only rank 0 prints.
            const h5cpp::distributed_info meta = f.info("/fields/velocity");
            const std::vector<std::int64_t> part =
                f.partition("/fields/velocity");

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

        // Rank 0 shows the file really is one interleaved dataset, reading it
        // with the plain serial API.
        if (me == 0) {
            h5cpp::file f(path, h5cpp::mode::read);
            const std::vector<double> flat =
                f.read<double>("/fields/velocity/data");

            std::cout << "as stored:";
            for (std::size_t i = 0; i < flat.size() && i < 12; ++i) {
                std::cout << ' ' << flat[i];
            }
            std::cout << (flat.size() > 12 ? " ...\n" : "\n");
            f.close();

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
