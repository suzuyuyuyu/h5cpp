// h5cpp — parallel visualization output.
//
// This header is the thin C++ face of h5c/h5c_viz.h. Every operation except
// status() is collective: all ranks call it in the same order with the same
// mesh and field names. Local point and cell counts may differ and may be 0.
// Connectivity is rank-local and 0-origin; h5c adds the point offset.
#ifndef H5CPP_VIZ_HPP
#define H5CPP_VIZ_HPP

#include <h5c/h5c_viz.h>
#include <mpi.h>

#include "h5cpp/h5cpp.hpp"

#if !defined(H5C_HAVE_PARALLEL)
#error "h5cpp/h5cpp_viz.hpp requires an h5c built with H5C_ENABLE_PARALLEL=ON"
#endif

namespace h5cpp {

enum class viz_kind {
    unstructured = H5C_VIZ_UNSTRUCTURED,
    polydata = H5C_VIZ_POLYDATA
};

/// One rank's contribution to a mesh. Empty strings and zero topology size
/// select the defaults documented by h5c_viz_mesh_t.
struct viz_mesh {
    viz_kind kind = viz_kind::unstructured;
    std::string name;
    std::string topology;
    int nodes_per_element = 0;
    std::size_t num_points = 0;
    std::size_t num_cells = 0;

    viz_mesh() = default;
    viz_mesh(viz_kind k, std::size_t points, std::size_t cells = 0)
        : kind(k), num_points(points), num_cells(cells) {}
    viz_mesh(viz_kind k, std::string n, std::size_t points,
             std::size_t cells = 0)
        : kind(k), name(std::move(n)), num_points(points), num_cells(cells) {}
    viz_mesh(viz_kind k, std::string n, std::string topo, int nodes,
             std::size_t points, std::size_t cells)
        : kind(k), name(std::move(n)), topology(std::move(topo)),
          nodes_per_element(nodes), num_points(points), num_cells(cells) {}
};

/// Where this rank's mesh contribution starts in the file.
struct viz_offsets {
    std::size_t point = 0;
    std::size_t cell = 0;
};

/// Owns an h5c visualization writer. Move-only; the destructor closes without
/// throwing, while explicit close() reports close failures.
class viz {
   public:
    /// Creates and truncates `path`, stamping scheme_version and `time`.
    /// Collective over `comm`; the communicator must remain valid until close.
    viz(const std::string& path, double time,
        MPI_Comm comm = MPI_COMM_WORLD, MPI_Info info = MPI_INFO_NULL) {
        h5c_viz_t* raw = nullptr;
        detail::check(h5c_viz_open(path.c_str(), time, comm, info, &raw),
                      "viz_open '" + path + "'");
        handle_ = raw;
    }

    viz(viz&& other) noexcept
        : handle_(other.handle_), num_points_(other.num_points_),
          num_cells_(other.num_cells_), nodes_per_element_(other.nodes_per_element_) {
        other.handle_ = nullptr;
    }

    viz& operator=(viz&& other) noexcept {
        if (this != &other) {
            discard();
            handle_ = other.handle_;
            num_points_ = other.num_points_;
            num_cells_ = other.num_cells_;
            nodes_per_element_ = other.nodes_per_element_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    viz(const viz&) = delete;
    viz& operator=(const viz&) = delete;

    /// Closes without throwing. Use close() when you need to observe failures.
    ~viz() { discard(); }

    /// Closes and reports failure. Collective; further calls are no-ops.
    void close() {
        if (handle_ == nullptr) {
            return;
        }
        h5c_status_t st = h5c_viz_close(handle_);
        handle_ = nullptr;
        detail::check(st, "viz_close");
    }

    /// Starts or re-selects a mesh. Collective.
    void begin_mesh(const viz_mesh& mesh) {
        h5c_viz_mesh_t raw{};
        raw.kind = static_cast<h5c_viz_kind_t>(mesh.kind);
        raw.name = mesh.name.empty() ? nullptr : mesh.name.c_str();
        raw.topology = mesh.topology.empty() ? nullptr : mesh.topology.c_str();
        raw.nodes_per_element = mesh.nodes_per_element;
        raw.num_points = mesh.num_points;
        raw.num_cells = mesh.num_cells;

        num_points_ = num_cells_ = 0;
        nodes_per_element_ = 0;
        detail::check(h5c_viz_begin_mesh(handle_, &raw),
                      "viz_begin_mesh '" + mesh.name + "'");
        num_points_ = mesh.num_points;
        num_cells_ = mesh.num_cells;
        nodes_per_element_ = mesh.kind == viz_kind::polydata
                                 ? 1
                                 : (mesh.nodes_per_element > 0
                                        ? mesh.nodes_per_element
                                        : H5C_VIZ_DEFAULT_NODES_PER_ELEM);
    }

    viz_offsets offsets() const {
        viz_offsets out;
        detail::check(h5c_viz_offsets(handle_, &out.point, &out.cell),
                      "viz_offsets");
        return out;
    }

    /// First failure recorded by h5c, or H5C_OK. Local, not collective.
    h5c_status_t status() const noexcept { return h5c_viz_status(handle_); }

    template <class T>
    void write_nodes(const T* nodes) {
        detail::check(h5c_viz_write_nodes(handle_, nodes, type_of<T>::value),
                      "viz_write_nodes");
    }

    template <class T>
    void write_nodes(const std::vector<T>& nodes) {
        require_size("nodes", nodes.size(), num_points_ * 3);
        write_nodes(nodes.data());
    }

    template <class T>
    void write_nodes(const std::vector<span<const T>>& xyz) {
        std::vector<const T*> ptrs;
        const std::size_t n = detail::gather_components("nodes", xyz, ptrs);
        require_components("nodes", ptrs.size(), 3);
        require_size("nodes", n, num_points_);
        detail::check(h5c_viz_write_nodes_comps(
                          handle_, reinterpret_cast<const void* const*>(ptrs.data()),
                          type_of<T>::value),
                      "viz_write_nodes_comps");
    }

    template <class T>
    void write_connectivity(const T* connectivity) {
        detail::check(h5c_viz_write_connectivity(
                          handle_, connectivity, type_of<T>::value),
                      "viz_write_connectivity");
    }

    template <class T>
    void write_connectivity(const std::vector<T>& connectivity) {
        require_size("connectivity", connectivity.size(),
                     num_cells_ * static_cast<std::size_t>(nodes_per_element_));
        write_connectivity(connectivity.data());
    }

    /// Component arrays are packed because the C API only accepts interleaved
    /// connectivity. Their common length is the local cell count.
    template <class T>
    void write_connectivity(const std::vector<span<const T>>& components) {
        std::vector<const T*> ptrs;
        const std::size_t n = detail::gather_components(
            "connectivity", components, ptrs);
        require_components("connectivity", ptrs.size(),
                           static_cast<std::size_t>(nodes_per_element_));
        require_size("connectivity", n, num_cells_);
        std::vector<T> packed(n * ptrs.size());
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t c = 0; c < ptrs.size(); ++c) {
                packed[i * ptrs.size() + c] = ptrs[c][i];
            }
        }
        write_connectivity(packed.data());
    }

    template <class T>
    void write_point_data(const std::string& name, const T* data,
                          std::size_t ncomp = 1) {
        write_field(name, data, ncomp, false);
    }

    template <class T>
    void write_point_data(const std::string& name, const std::vector<T>& data,
                          std::size_t ncomp = 1) {
        require_size(name, data.size(), num_points_ * ncomp);
        write_point_data(name, data.data(), ncomp);
    }

    template <class T>
    void write_point_data(const std::string& name,
                          const std::vector<span<const T>>& components) {
        write_field_components(name, components, num_points_, false);
    }

    template <class T>
    void write_cell_data(const std::string& name, const T* data,
                         std::size_t ncomp = 1) {
        write_field(name, data, ncomp, true);
    }

    template <class T>
    void write_cell_data(const std::string& name, const std::vector<T>& data,
                         std::size_t ncomp = 1) {
        require_size(name, data.size(), num_cells_ * ncomp);
        write_cell_data(name, data.data(), ncomp);
    }

    template <class T>
    void write_cell_data(const std::string& name,
                         const std::vector<span<const T>>& components) {
        write_field_components(name, components, num_cells_, true);
    }

    /// Borrowed h5c handle; h5cpp keeps ownership.
    h5c_viz_t* c_handle() const noexcept { return handle_; }

   private:
    void discard() noexcept {
        if (handle_ != nullptr) {
            h5c_viz_close(handle_);  // status deliberately ignored in a destructor
            handle_ = nullptr;
        }
    }

    template <class T>
    void write_field(const std::string& name, const T* data,
                     std::size_t ncomp, bool cell) {
        const h5c_status_t st = cell
            ? h5c_viz_write_cell_data(handle_, name.c_str(), data,
                                      type_of<T>::value, ncomp)
            : h5c_viz_write_point_data(handle_, name.c_str(), data,
                                       type_of<T>::value, ncomp);
        detail::check(st, std::string("viz_write_") +
                              (cell ? "cell_data '" : "point_data '") + name + "'");
    }

    template <class T>
    void write_field_components(const std::string& name,
                                const std::vector<span<const T>>& components,
                                std::size_t expected, bool cell) {
        std::vector<const T*> ptrs;
        const std::size_t n = detail::gather_components(name, components, ptrs);
        require_size(name, n, expected);
        const h5c_status_t st = cell
            ? h5c_viz_write_cell_data_comps(
                  handle_, name.c_str(),
                  reinterpret_cast<const void* const*>(ptrs.data()),
                  type_of<T>::value, ptrs.size())
            : h5c_viz_write_point_data_comps(
                  handle_, name.c_str(),
                  reinterpret_cast<const void* const*>(ptrs.data()),
                  type_of<T>::value, ptrs.size());
        detail::check(st, std::string("viz_write_") +
                              (cell ? "cell_data_comps '" : "point_data_comps '") +
                              name + "'");
    }

    static void require_size(const std::string& what, std::size_t given,
                             std::size_t wanted) {
        if (given != wanted) {
            throw error(H5C_ERR_SHAPE_MISMATCH, "'" + what + "' holds " +
                            std::to_string(given) + " elements but needs " +
                            std::to_string(wanted));
        }
    }

    static void require_components(const std::string& what, std::size_t given,
                                   std::size_t wanted) {
        if (given != wanted) {
            throw error(H5C_ERR_SHAPE_MISMATCH, "'" + what + "' has " +
                            std::to_string(given) + " components but needs " +
                            std::to_string(wanted));
        }
    }

    h5c_viz_t* handle_ = nullptr;
    std::size_t num_points_ = 0;
    std::size_t num_cells_ = 0;
    int nodes_per_element_ = 0;
};

inline std::string attribute_type(std::size_t ncomp) {
    const char* value = h5c_viz_attribute_type(ncomp);
    return value == nullptr ? std::string{} : std::string(value);
}

}  // namespace h5cpp

#endif  // H5CPP_VIZ_HPP
