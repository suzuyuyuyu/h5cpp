// Five visualization steps containing a tetra mesh and a particle cloud.
// MUST be run through the batch system, never on a login node:
//     sbatch example/run-parallel-example.sh
#include <h5cpp/h5cpp_viz.hpp>

#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int kSteps = 5;
constexpr std::size_t kCells = 4;
constexpr std::size_t kParticles = 6;
int g_me = 0;

void write_step(int step, double time) {
    char path[64];
    std::snprintf(path, sizeof path, "result/seq%06d.h5", step);
    h5cpp::viz writer(path, time);

    const std::size_t npoints = 4 * kCells;
    std::vector<double> nodes(npoints * 3), pressure(npoints), velocity(npoints * 3);
    std::vector<std::int32_t> connectivity(kCells * 4), subdomain(kCells);
    constexpr double unit[4][3] = {
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}
    };
    for (std::size_t cell = 0; cell < kCells; ++cell) {
        for (std::size_t k = 0; k < 4; ++k) {
            const std::size_t p = 4 * cell + k;
            const double x = static_cast<double>(cell) + unit[k][0];
            const double y = static_cast<double>(g_me) + unit[k][1];
            const double z = unit[k][2];
            nodes[3 * p] = x;
            nodes[3 * p + 1] = y;
            nodes[3 * p + 2] = z;
            pressure[p] = std::sin(x - 2.0 * time) * std::cos(0.5 * y);
            velocity[3 * p] = -y;
            velocity[3 * p + 1] = x;
            velocity[3 * p + 2] = 0.1 * std::sin(time);
            connectivity[4 * cell + k] = static_cast<std::int32_t>(p);
        }
        subdomain[cell] = g_me;
    }

    writer.begin_mesh({h5cpp::viz_kind::unstructured, "fluid", "Tetrahedron",
                       4, npoints, kCells});
    writer.write_nodes(nodes);
    writer.write_connectivity(connectivity);
    writer.write_point_data("Pressure", pressure);
    writer.write_point_data("Velocity", velocity, 3);
    writer.write_cell_data("SubdomainID", subdomain);

    std::vector<double> x(kParticles), y(kParticles), z(kParticles), radius(kParticles);
    for (std::size_t i = 0; i < kParticles; ++i) {
        const double s = static_cast<double>(i) / static_cast<double>(kParticles);
        x[i] = 0.5 + 3.0 * s + 0.3 * std::sin(time + s);
        y[i] = static_cast<double>(g_me) + 0.5 + 0.2 * std::cos(time + s);
        z[i] = 2.0 - 0.15 * time * (1.0 + s);
        radius[i] = 0.05 + 0.02 * s;
    }
    writer.begin_mesh({h5cpp::viz_kind::polydata, "particles", kParticles});
    writer.write_nodes<double>({x, y, z});
    writer.write_point_data("Radius", radius);

    if (writer.status() != H5C_OK) {
        throw h5cpp::error(writer.status(), "visualization write failed");
    }
    writer.close();
    if (g_me == 0) {
        std::cout << "wrote " << path << " (t = " << time << ")\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_me);

    int status = 0;
    try {
        int directory_error = 0;
        if (g_me == 0) {
            if (mkdir("result", 0777) != 0 && errno != EEXIST) {
                directory_error = 1;
            }
        }
        MPI_Bcast(&directory_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (directory_error != 0) {
            throw std::runtime_error("cannot create result directory");
        }
        for (int step = 0; step < kSteps; ++step) {
            write_step(step, 0.25 * step);
        }
    } catch (const std::exception& e) {
        std::cerr << "rank " << g_me << ": " << e.what() << '\n';
        status = 1;
    }

    int failed = 0;
    MPI_Allreduce(&status, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return failed;
}
