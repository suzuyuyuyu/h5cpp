// Parallel visualization wrapper test. Run only through the batch-system MPI
// test job. Asymmetric values make transposed and mis-offset writes visible.
#define _POSIX_C_SOURCE 200809L

#include "h5cpp_test.hpp"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <vector>

#include "h5cpp/h5cpp_viz.hpp"

namespace {

const char* kPath = "test_pviz.h5";
const unsigned kWatchdogSeconds = 120;
int g_me = 0;
int g_nprocs = 1;

extern "C" void on_watchdog(int) {
    std::fprintf(stderr, "test_pviz: WATCHDOG fired on rank %d\n", g_me);
    std::fflush(stderr);
    _exit(2);
}

template <class F>
h5c_status_t status_of(F&& fn) {
    try {
        fn();
    } catch (const h5cpp::error& e) {
        return e.status();
    }
    return H5C_OK;
}

void assert_same_status(h5c_status_t st, const char* what) {
    int mine = static_cast<int>(st), lo = 0, hi = 0;
    MPI_Allreduce(&mine, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&mine, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    H5CPP_ASSERT(lo == hi, "%s: ranks returned different statuses (%d..%d)",
                 what, lo, hi);
}

struct layout {
    std::size_t point_offset;
    std::size_t cell_offset;
    std::size_t total_points;
    std::size_t total_cells;
};

layout expected_layout(std::size_t np, std::size_t nc) {
    unsigned long long mine[2] = {
        static_cast<unsigned long long>(np),
        static_cast<unsigned long long>(nc)
    };
    std::vector<unsigned long long> all(static_cast<std::size_t>(g_nprocs) * 2);
    MPI_Allgather(mine, 2, MPI_UNSIGNED_LONG_LONG, all.data(), 2,
                  MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    layout out{0, 0, 0, 0};
    for (int r = 0; r < g_nprocs; ++r) {
        const std::size_t p = static_cast<std::size_t>(all[2 * r]);
        const std::size_t c = static_cast<std::size_t>(all[2 * r + 1]);
        if (r < g_me) {
            out.point_offset += p;
            out.cell_offset += c;
        }
        out.total_points += p;
        out.total_cells += c;
    }
    return out;
}

double value(int rank, std::size_t row, std::size_t comp, double family) {
    return family + 1000.0 * rank + 10.0 * static_cast<double>(row) +
           static_cast<double>(comp);
}

void check_offsets(h5cpp::viz& writer, std::size_t np, std::size_t nc,
                   const char* mesh) {
    const layout want = expected_layout(np, nc);
    const h5cpp::viz_offsets got = writer.offsets();
    H5CPP_ASSERT(got.point == want.point_offset,
                 "%s point offset: got %lu want %lu", mesh,
                 static_cast<unsigned long>(got.point),
                 static_cast<unsigned long>(want.point_offset));
    H5CPP_ASSERT(got.cell == want.cell_offset,
                 "%s cell offset: got %lu want %lu", mesh,
                 static_cast<unsigned long>(got.cell),
                 static_cast<unsigned long>(want.cell_offset));
}

void write_fluid(h5cpp::viz& writer) {
    const std::size_t np = (g_nprocs > 1 && g_me == 0)
                               ? 0u : static_cast<std::size_t>(5 + 2 * g_me);
    const std::size_t nc = (g_nprocs > 1 && g_me == 0)
                               ? 0u : static_cast<std::size_t>(1 + g_me);
    writer.begin_mesh({h5cpp::viz_kind::unstructured, "fluid", "Tetrahedron",
                       4, np, nc});
    check_offsets(writer, np, nc, "fluid");

    std::vector<double> x(np), y(np), z(np);
    for (std::size_t i = 0; i < np; ++i) {
        x[i] = value(g_me, i, 0, 10000.0);
        y[i] = value(g_me, i, 1, 10000.0);
        z[i] = value(g_me, i, 2, 10000.0);
    }
    writer.write_nodes<double>({x, y, z});

    std::vector<std::int32_t> conn(nc * 4);
    for (std::size_t c = 0; c < nc; ++c) {
        for (std::size_t k = 0; k < 4; ++k) {
            conn[c * 4 + k] = static_cast<std::int32_t>((2 * c + k) % np);
        }
    }
    writer.write_connectivity(conn);

    std::vector<double> pressure(np), stress(np * 6);
    std::vector<double> u(np), v(np), w(np);
    for (std::size_t i = 0; i < np; ++i) {
        pressure[i] = value(g_me, i, 0, 20000.0);
        u[i] = value(g_me, i, 0, 30000.0);
        v[i] = value(g_me, i, 1, 30000.0);
        w[i] = value(g_me, i, 2, 30000.0);
        for (std::size_t k = 0; k < 6; ++k) {
            stress[i * 6 + k] = value(g_me, i, k, 40000.0);
        }
    }
    writer.write_point_data("Pressure", pressure);
    writer.write_point_data<double>("Velocity", {u, v, w});
    writer.write_point_data("Stress", stress, 6);

    std::vector<std::int32_t> owner(nc);
    std::vector<float> cu(nc), cv(nc), cw(nc);
    std::vector<double> cell_stress(nc * 6);
    for (std::size_t c = 0; c < nc; ++c) {
        owner[c] = static_cast<std::int32_t>(100 * g_me + c);
        cu[c] = static_cast<float>(value(g_me, c, 0, 50000.0));
        cv[c] = static_cast<float>(value(g_me, c, 1, 50000.0));
        cw[c] = static_cast<float>(value(g_me, c, 2, 50000.0));
        for (std::size_t k = 0; k < 6; ++k) {
            cell_stress[c * 6 + k] = value(g_me, c, k, 60000.0);
        }
    }
    writer.write_cell_data("Owner", owner);
    writer.write_cell_data<float>("CellVelocity", {cu, cv, cw});
    writer.write_cell_data("CellStress", cell_stress, 6);

    // gather_components rejects this before h5c sees it: sticky status stays OK.
    std::vector<double> long_v(np + 1);
    h5c_status_t st = status_of([&] {
        writer.write_point_data<double>("Ragged", {u, long_v, w});
    });
    H5CPP_ASSERT(st == H5C_ERR_SHAPE_MISMATCH, "ragged views: got %s",
                 h5c_status_string(st));
    assert_same_status(st, "ragged views");
    H5CPP_ASSERT(writer.status() == H5C_OK,
                 "ragged views reached h5c before rejection");
    writer.write_point_data<double>("Ragged", {u, v, w});
}

void write_component_connectivity(h5cpp::viz& writer) {
    const std::size_t np = static_cast<std::size_t>(4 + g_me);
    const std::size_t nc = static_cast<std::size_t>(1 + g_me);
    writer.begin_mesh({h5cpp::viz_kind::unstructured, "component_mesh",
                       "Triangle", 3, np, nc});
    check_offsets(writer, np, nc, "component_mesh");

    std::vector<double> nodes(np * 3);
    for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            nodes[i * 3 + k] = value(g_me, i, k, 70000.0);
        }
    }
    writer.write_nodes(nodes);

    std::vector<std::int32_t> a(nc), b(nc), c(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        a[i] = static_cast<std::int32_t>(i % np);
        b[i] = static_cast<std::int32_t>((i + 2) % np);
        c[i] = static_cast<std::int32_t>((i + 3) % np);
    }
    writer.write_connectivity<std::int32_t>({a, b, c});
}

void write_particles(h5cpp::viz& writer) {
    const std::size_t np = static_cast<std::size_t>(3 + 3 * g_me);
    writer.begin_mesh({h5cpp::viz_kind::polydata, "particles", np});
    check_offsets(writer, np, 0, "particles");

    std::vector<double> nodes(np * 3), radius(np);
    for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t k = 0; k < 3; ++k) {
            nodes[i * 3 + k] = value(g_me, i, k, 80000.0);
        }
        radius[i] = value(g_me, i, 0, 90000.0);
    }
    writer.write_nodes(nodes);
    writer.write_point_data("Radius", radius);
}

void test_bad_connectivity(h5cpp::viz& writer) {
    const std::size_t np = static_cast<std::size_t>(4 + g_me);
    const std::size_t nc = static_cast<std::size_t>(1 + g_me);
    writer.begin_mesh({h5cpp::viz_kind::unstructured, "rejected", "Triangle",
                       3, np, nc});
    std::vector<double> nodes(np * 3, 0.0);
    writer.write_nodes(nodes);

    std::vector<std::int32_t> conn(nc * 3, 0);
    if (g_me == g_nprocs - 1) {
        conn[0] = static_cast<std::int32_t>(np);
    }
    const h5c_status_t st = status_of([&] { writer.write_connectivity(conn); });
    H5CPP_ASSERT(st == H5C_ERR_INVALID_ARG, "bad connectivity: got %s",
                 h5c_status_string(st));
    assert_same_status(st, "bad connectivity");
    H5CPP_ASSERT(writer.status() == H5C_ERR_INVALID_ARG,
                 "sticky status did not record bad connectivity");
}

void check_file() {
    if (g_me != 0) {
        return;
    }
    h5cpp::file f(kPath, h5cpp::mode::read);

    std::size_t fluid_points = 0, fluid_cells = 0;
    for (int r = 1; r < g_nprocs; ++r) {
        fluid_points += static_cast<std::size_t>(5 + 2 * r);
        fluid_cells += static_cast<std::size_t>(1 + r);
    }
    H5CPP_ASSERT(f.info("/fluid/geometry/nodes").dims ==
                     (h5cpp::shape{fluid_points, 3}), "fluid node shape");
    H5CPP_ASSERT(f.info("/fluid/point_data/Stress").dims ==
                     (h5cpp::shape{fluid_points, 6}), "point tensor shape");
    H5CPP_ASSERT(f.info("/fluid/cell_data/CellStress").dims ==
                     (h5cpp::shape{fluid_cells, 6}), "cell tensor shape");
    H5CPP_ASSERT(f.read_attr_str("/fluid/point_data/Velocity", "attribute_type") ==
                     "Vector", "point vector attribute");
    H5CPP_ASSERT(f.read_attr_str("/fluid/cell_data/CellStress", "attribute_type") ==
                     "Tensor6", "cell tensor attribute");
    H5CPP_ASSERT(f.exists("/component_mesh/geometry/connectivity"),
                 "component connectivity missing");
    const std::size_t particle_points =
        3 * static_cast<std::size_t>(g_nprocs) *
        static_cast<std::size_t>(g_nprocs + 1) / 2;
    H5CPP_ASSERT(f.info("/particles/geometry/nodes").dims ==
                     (h5cpp::shape{particle_points, 3}), "polydata node shape");
    H5CPP_ASSERT(!f.exists("/particles/geometry/connectivity"),
                 "polydata has connectivity");

    const std::vector<double> pressure =
        f.read<double>("/fluid/point_data/Pressure");
    const std::vector<double> velocity =
        f.read<double>("/fluid/point_data/Velocity");
    const std::vector<double> stress =
        f.read<double>("/fluid/point_data/Stress");
    std::size_t row = 0;
    for (int r = 1; r < g_nprocs; ++r) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(5 + 2 * r); ++i, ++row) {
            H5CPP_ASSERT(pressure[row] == value(r, i, 0, 20000.0),
                         "pressure[%lu]", static_cast<unsigned long>(row));
            for (std::size_t k = 0; k < 3; ++k) {
                H5CPP_ASSERT(velocity[row * 3 + k] == value(r, i, k, 30000.0),
                             "velocity[%lu][%lu]", static_cast<unsigned long>(row),
                             static_cast<unsigned long>(k));
            }
            for (std::size_t k = 0; k < 6; ++k) {
                H5CPP_ASSERT(stress[row * 6 + k] == value(r, i, k, 40000.0),
                             "point stress[%lu][%lu]",
                             static_cast<unsigned long>(row),
                             static_cast<unsigned long>(k));
            }
        }
    }

    const std::vector<std::int32_t> owner =
        f.read<std::int32_t>("/fluid/cell_data/Owner");
    const std::vector<float> cell_velocity =
        f.read<float>("/fluid/cell_data/CellVelocity");
    const std::vector<double> cell_stress =
        f.read<double>("/fluid/cell_data/CellStress");
    row = 0;
    for (int r = 1; r < g_nprocs; ++r) {
        for (std::size_t c = 0; c < static_cast<std::size_t>(1 + r); ++c, ++row) {
            H5CPP_ASSERT(owner[row] == static_cast<std::int32_t>(100 * r + c),
                         "owner[%lu]", static_cast<unsigned long>(row));
            for (std::size_t k = 0; k < 3; ++k) {
                H5CPP_ASSERT(cell_velocity[row * 3 + k] ==
                                 static_cast<float>(value(r, c, k, 50000.0)),
                             "cell velocity[%lu][%lu]",
                             static_cast<unsigned long>(row),
                             static_cast<unsigned long>(k));
            }
            for (std::size_t k = 0; k < 6; ++k) {
                H5CPP_ASSERT(cell_stress[row * 6 + k] ==
                                 value(r, c, k, 60000.0),
                             "cell stress[%lu][%lu]",
                             static_cast<unsigned long>(row),
                             static_cast<unsigned long>(k));
            }
        }
    }

    const std::vector<std::int32_t> fluid_conn =
        f.read<std::int32_t>("/fluid/geometry/connectivity");
    std::size_t point_base = 0, cell_base = 0;
    for (int r = 1; r < g_nprocs; ++r) {
        const std::size_t np = static_cast<std::size_t>(5 + 2 * r);
        const std::size_t nc = static_cast<std::size_t>(1 + r);
        for (std::size_t c = 0; c < nc; ++c) {
            for (std::size_t k = 0; k < 4; ++k) {
                const std::int32_t want = static_cast<std::int32_t>(
                    point_base + (2 * c + k) % np);
                H5CPP_ASSERT(fluid_conn[(cell_base + c) * 4 + k] == want,
                             "buffer connectivity[%lu][%lu]",
                             static_cast<unsigned long>(cell_base + c),
                             static_cast<unsigned long>(k));
            }
        }
        point_base += np;
        cell_base += nc;
    }

    const std::vector<std::int32_t> component_conn =
        f.read<std::int32_t>("/component_mesh/geometry/connectivity");
    point_base = cell_base = 0;
    for (int r = 0; r < g_nprocs; ++r) {
        const std::size_t np = static_cast<std::size_t>(4 + r);
        const std::size_t nc = static_cast<std::size_t>(1 + r);
        for (std::size_t c = 0; c < nc; ++c) {
            const std::size_t local[3] = {c % np, (c + 2) % np, (c + 3) % np};
            for (std::size_t k = 0; k < 3; ++k) {
                const std::int32_t want =
                    static_cast<std::int32_t>(point_base + local[k]);
                H5CPP_ASSERT(component_conn[(cell_base + c) * 3 + k] == want,
                             "component connectivity[%lu][%lu]",
                             static_cast<unsigned long>(cell_base + c),
                             static_cast<unsigned long>(k));
            }
        }
        point_base += np;
        cell_base += nc;
    }
    f.close();
}

void test_attribute_type() {
    H5CPP_ASSERT(h5cpp::attribute_type(1) == "Scalar", "Scalar mapping");
    H5CPP_ASSERT(h5cpp::attribute_type(3) == "Vector", "Vector mapping");
    H5CPP_ASSERT(h5cpp::attribute_type(6) == "Tensor6", "Tensor6 mapping");
    H5CPP_ASSERT(h5cpp::attribute_type(9) == "Tensor", "Tensor mapping");
    H5CPP_ASSERT(h5cpp::attribute_type(2).empty(), "unnamed mapping");
}

}  // namespace

H5CPP_TEST_MAIN_STATE;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_me);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);
    std::signal(SIGALRM, on_watchdog);
    alarm(kWatchdogSeconds);

    test_attribute_type();
    {
        h5cpp::viz writer(kPath, 1.25);
        write_fluid(writer);
        write_component_connectivity(writer);
        write_particles(writer);
        test_bad_connectivity(writer);
        writer.close();
        writer.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    check_file();
    MPI_Barrier(MPI_COMM_WORLD);

    alarm(0);
    int total = 0;
    MPI_Allreduce(&h5cpp_test_failures, &total, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    h5cpp_test_failures = total;
    if (g_me != 0) {
        MPI_Finalize();
        return total == 0 ? 0 : 1;
    }
    MPI_Finalize();
    return H5CPP_TEST_SUMMARY("test_pviz");
}
