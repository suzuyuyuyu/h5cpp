// h5cpp — parallel (MPI) I/O.
//
// This header is the ONLY place h5cpp exposes <mpi.h>. Serial users include
// h5cpp/h5cpp.hpp and never pull in MPI. It wraps h5c/h5c_mpi.h and therefore
// exists only when h5c was built with H5C_ENABLE_PARALLEL=ON, which defines
// H5C_HAVE_PARALLEL. Nothing here calls HDF5 or MPI-IO directly.
//
// ---------------------------------------------------------------------------
// Storage layout (identical to h5c and h5fortran)
// ---------------------------------------------------------------------------
//
// Writing a distributed array to path P produces a GROUP:
//
//     P/data             all ranks' data concatenated along the split axis
//     P/__partition__    int64 rank boundaries, length nprocs + 1
//
// The array is split along the SLOWEST-VARYING axis, shape[0]. `shape`
// always describes the LOCAL block: shape[0] is this rank's extent and may
// be 0; every other extent must agree on every rank. Rank >= 1 is required,
// because a scalar has no axis to split.
//
// ---------------------------------------------------------------------------
// Collective discipline
// ---------------------------------------------------------------------------
//
// EVERY member function of parallel_file except the const accessors
// (is_collective(), comm(), exists(), c_handle(), hid()) is COLLECTIVE over
// the file's communicator: all ranks must call it, in the same order, with
// the same path. That includes the constructors and close().
//
// Argument validation is agreed across ranks inside h5c before any HDF5
// collective call, so a bad argument on one rank makes every rank throw an
// h5cpp::error with the same status() instead of deadlocking. Because the
// throw happens on all ranks, an ordinary try/catch around a collective call
// keeps the ranks in step -- but only if every rank handles it the same way.
#ifndef H5CPP_MPI_HPP
#define H5CPP_MPI_HPP

#include <h5c/h5c_mpi.h>
#include <mpi.h>

#include "h5cpp/h5cpp.hpp"

#if !defined(H5C_HAVE_PARALLEL)
#error "h5cpp/h5cpp_mpi.hpp requires an h5c built with H5C_ENABLE_PARALLEL=ON"
#endif

namespace h5cpp {

/// Local and global shape of a distributed dataset. `local` is this rank's
/// block as recorded in __partition__, `global` the full extent of "data".
struct distributed_info {
    dataset_info local;
    dataset_info global;
};

/// Where a rank's block sits along the split axis.
struct block_extent {
    std::size_t offset = 0;  ///< first row this rank owns
    std::size_t count = 0;   ///< number of rows it owns; may be 0
};

// ---------------------------------------------------------------------------
// parallel_file
// ---------------------------------------------------------------------------

/// Owns an h5c file handle opened for parallel I/O. Same RAII rules as
/// h5cpp::file: move-only, the destructor closes without throwing, explicit
/// close() throws.
///
/// It is a distinct type from h5cpp::file on purpose. The two differ in the
/// contract, not just the implementation: on a parallel_file, write() and
/// read() move only this rank's block and are collective, so accepting one
/// where a serial file is expected would silently change the meaning of the
/// call. Keeping them separate also keeps h5cpp.hpp free of <mpi.h>.
///
///     h5cpp::parallel_file f("out.h5", h5cpp::mode::truncate);
///     f.write("/coords", local, {nlocal, 3});   // every rank, same order
class parallel_file {
   public:
    /// Opens on MPI_COMM_WORLD with MPI_INFO_NULL. Collective.
    parallel_file(const std::string& path, mode m) {
        h5c_file_t* raw = nullptr;
        detail::check(h5c_popen(path.c_str(), static_cast<h5c_mode_t>(m), &raw), "popen '" + path + "'");
        handle_ = raw;
    }

    /// Opens on an explicit communicator. `info` carries MPI-IO hints and may
    /// be MPI_INFO_NULL. The communicator must stay valid until close().
    /// Collective over `comm`.
    parallel_file(const std::string& path, mode m, MPI_Comm comm, MPI_Info info = MPI_INFO_NULL) {
        h5c_file_t* raw = nullptr;
        detail::check(h5c_popen_comm(path.c_str(), static_cast<h5c_mode_t>(m), comm, info, &raw), "popen '" + path + "'");
        handle_ = raw;
    }

    parallel_file(parallel_file&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    parallel_file& operator=(parallel_file&& other) noexcept {
        if (this != &other) {
            discard();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    parallel_file(const parallel_file&) = delete;
    parallel_file& operator=(const parallel_file&) = delete;

    /// Closes without throwing. Use close() when you need to observe failures.
    ~parallel_file() { discard(); }

    /// Closes and reports failure. Collective: every rank must call it, or let
    /// the destructor do it, at the same point. Further calls are no-ops.
    void close() {
        if (handle_ == nullptr) {
            return;
        }
        h5c_status_t st = h5c_close(handle_);
        handle_ = nullptr;
        detail::check(st, "close");
    }

    // -- transfer mode ---------------------------------------------------

    /// Selects collective (the default) or independent transfers for every
    /// later parallel call on this file. h5cpp never switches modes implicitly,
    /// so all ranks must make the same choice. Collective.
    void set_collective(bool collective) {
        detail::check(h5c_pset_collective(handle_, collective ? 1 : 0), "set_collective");
    }

    bool is_collective() const noexcept {
        return h5c_pis_collective(handle_) != 0;
    }

    /// The file's communicator, or MPI_COMM_NULL if it is not parallel.
    /// Borrowed: h5cpp does not free it.
    MPI_Comm comm() const noexcept { return h5c_pcomm(handle_); }

    // -- writing ---------------------------------------------------------

    /// Writes this rank's block. `dims` is the LOCAL shape; dims[0] is the
    /// split extent and may be 0. Collective.
    template <class T>
    void write(const std::string& path, const T* data, const shape& dims, bool replace = false) {
        detail::check(
            h5c_pwrite(handle_, path.c_str(), data, type_of<T>::value, static_cast<int>(dims.size()), dims.empty() ? nullptr : dims.data(), replace ? H5C_WRITE_REPLACE : H5C_WRITE_DEFAULT),
            "pwrite '" + path + "'"
        );
    }

    /// Container overload; checks that the buffer length matches the local
    /// shape. An empty buffer is fine when the shape has a zero extent.
    template <class T>
    void write(const std::string& path, const std::vector<T>& data, const shape& dims, bool replace = false) {
        // Caught here rather than in type_of<bool>: std::vector<bool> has no
        // data() to instantiate against, so the guard must come first.
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        require_size(path, data.size(), element_count(dims));
        write<T>(path, data.data(), dims, replace);
    }

    /// One-dimensional shorthand: the vector is this rank's whole block.
    template <class T>
    void write(const std::string& path, const std::vector<T>& data, bool replace = false) {
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        write<T>(path, data.data(), shape{data.size()}, replace);
    }

    // -- reading ---------------------------------------------------------

    /// Reads this rank's block into storage the caller owns. `dims` is the
    /// LOCAL shape, exactly as for write(). Collective.
    template <class T>
    void read_into(const std::string& path, T* data, const shape& dims) {
        detail::check(
            h5c_pread(handle_, path.c_str(), data, type_of<T>::value, static_cast<int>(dims.size()), dims.empty() ? nullptr : dims.data()),
            "pread '" + path + "'"
        );
    }

    template <class T>
    void read_into(const std::string& path, std::vector<T>& data, const shape& dims) {
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        require_size(path, data.size(), element_count(dims));
        T* raw = data.data();
        read_into<T>(path, raw, dims);
    }

    /// Reads this rank's block, sizing the vector from __partition__.
    /// Collective (info() and the read both are).
    template <class T>
    std::vector<T> read(const std::string& path) {
        const dataset_info meta = info(path).local;
        std::vector<T> out(meta.count);
        read_into<T>(path, out.data(), meta.dims);
        return out;
    }

    /// Reads a plain, undistributed dataset -- every rank gets the whole
    /// thing. Useful for small global arrays and for inspecting "P/data"
    /// directly. This is the serial h5c read path, so it does not use the
    /// parallel transfer mode; call it on every rank all the same.
    template <class T>
    void read_replicated(const std::string& path, T* data, const shape& dims) {
        detail::check(
            h5c_read(handle_, path.c_str(), data, type_of<T>::value, static_cast<int>(dims.size()), dims.empty() ? nullptr : dims.data()),
            "read '" + path + "'"
        );
    }

    template <class T>
    void read_replicated(const std::string& path, std::vector<T>& data, const shape& dims) {
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        require_size(path, data.size(), element_count(dims));
        T* raw = data.data();
        read_replicated<T>(path, raw, dims);
    }

    // -- interleaved multi-component I/O ---------------------------------

    /// Writes `ncomp` component arrays of `n` local elements each as one
    /// [total_n, ncomp] distributed dataset. Collective.
    ///
    /// Components are passed as a container of views rather than one packed
    /// buffer because that is how solver data actually lives -- u, v, w are
    /// separate arrays -- and h5c does the packing:
    ///
    ///     f.write_interleaved<double>("/velocity", {u, v, w});
    ///
    /// The overload taking views derives `n` from them and rejects components
    /// of differing length, which is the mistake that otherwise surfaces as
    /// silently shuffled data. `n == 0` is legal: pass empty views.
    template <class T>
    void write_interleaved(const std::string& path, const std::vector<const T*>& comps, std::size_t n, bool replace = false) {
        detail::check(
            h5c_pwrite_interleaved(
                handle_, path.c_str(),
                reinterpret_cast<const void* const*>(comps.data()),
                comps.size(), n, type_of<T>::value,
                replace ? H5C_WRITE_REPLACE : H5C_WRITE_DEFAULT
            ),
            "pwrite_interleaved '" + path + "'"
        );
    }

    template <class T>
    void write_interleaved(const std::string& path, const std::vector<span<const T>>& comps, bool replace = false) {
        std::vector<const T*> ptrs;
        const std::size_t n = detail::gather_components(path, comps, ptrs);
        write_interleaved<T>(path, ptrs, n, replace);
    }

    /// Reads this rank's rows of an interleaved dataset back into separate
    /// component arrays. Collective.
    template <class T>
    void read_interleaved(const std::string& path, const std::vector<T*>& comps, std::size_t n) {
        detail::check(
            h5c_pread_interleaved(handle_, path.c_str(), reinterpret_cast<void* const*>(comps.data()), comps.size(), n, type_of<T>::value),
            "pread_interleaved '" + path + "'"
        );
    }

    template <class T>
    void read_interleaved(const std::string& path, const std::vector<span<T>>& comps) {
        std::vector<T*> ptrs;
        const std::size_t n = detail::gather_components(path, comps, ptrs);
        read_interleaved<T>(path, ptrs, n);
    }

    // -- attributes and strings ------------------------------------------

    /// Attributes and scalar strings on a parallel file.
    ///
    /// These are HDF5 metadata operations: every rank must call them, in the
    /// same order, with the SAME value. They are wrapped here rather than
    /// left to c_handle() because annotating a distributed dataset -- with
    /// "attribute_type" for XDMF, say -- is part of writing it.
    ///
    /// The dataset itself lives at "<path>/data", so that is the object to
    /// annotate, not the group.
    void write_attr_str(const std::string& obj_path, const std::string& name, const std::string& value) {
        detail::check(h5c_write_attr_str(handle_, obj_path.c_str(), name.c_str(), value.c_str()), "write_attr_str '" + obj_path + ":" + name + "'");
    }

    std::string read_attr_str(const std::string& obj_path, const std::string& name) {
        char* raw = nullptr;
        detail::check(h5c_read_attr_str(handle_, obj_path.c_str(), name.c_str(), &raw), "read_attr_str '" + obj_path + ":" + name + "'");
        std::string out = (raw != nullptr) ? raw : "";
        h5c_free_string(raw);
        return out;
    }

    template <class T>
    void write_attr_scalar(const std::string& obj_path, const std::string& name, T value) {
        detail::check(h5c_write_attr_scalar(handle_, obj_path.c_str(), name.c_str(), &value, type_of<T>::value), "write_attr_scalar '" + obj_path + ":" + name + "'");
    }

    template <class T>
    T read_attr_scalar(const std::string& obj_path, const std::string& name) {
        T value{};
        detail::check(h5c_read_attr_scalar(handle_, obj_path.c_str(), name.c_str(), &value, type_of<T>::value), "read_attr_scalar '" + obj_path + ":" + name + "'");
        return value;
    }

    /// Writes a numeric 1-D attribute. The count comes from the container, so
    /// it cannot disagree with the storage passed to h5c. Collective.
    template <class T>
    void write_attr_array(const std::string& obj_path, const std::string& name, const std::vector<T>& values) {
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        detail::check(h5c_write_attr_array(handle_, obj_path.c_str(), name.c_str(), values.data(), type_of<T>::value, values.size()), "write_attr_array '" + obj_path + ":" + name + "'");
    }

    /// Reads a numeric 1-D attribute, sizing the result from its stored length.
    /// Collective.
    template <class T>
    std::vector<T> read_attr_array(const std::string& obj_path, const std::string& name) {
        std::vector<T> out(attr_length(obj_path, name));
        detail::check(h5c_read_attr_array(handle_, obj_path.c_str(), name.c_str(), out.data(), type_of<T>::value, out.size()), "read_attr_array '" + obj_path + ":" + name + "'");
        return out;
    }

    /// Elements in an attribute: 1 for a scalar, n for an array. Collective.
    std::size_t attr_length(const std::string& obj_path, const std::string& name) {
        std::size_t count = 0;
        detail::check(h5c_attr_length(handle_, obj_path.c_str(), name.c_str(), &count), "attr_length '" + obj_path + ":" + name + "'");
        return count;
    }

    /// Local query, like exists(): never touches the sticky status.
    bool attr_exists(const std::string& obj_path, const std::string& name) const {
        return h5c_attr_exists(handle_, obj_path.c_str(), name.c_str()) != 0;
    }

    // -- metadata --------------------------------------------------------

    /// Local and global shape of a distributed dataset. Collective.
    distributed_info info(const std::string& path) {
        h5c_dataset_info_t local;
        h5c_dataset_info_t global;
        detail::check(h5c_pdataset_info(handle_, path.c_str(), &local, &global), "pdataset_info '" + path + "'");

        distributed_info out;
        out.local = convert(local);
        out.global = convert(global);
        return out;
    }

    /// Where this rank's block starts along the split axis, and how many rows
    /// it owns. This is what info() cannot tell you. Collective.
    ///
    /// Prefer this over reading "<path>/__partition__" yourself: h5c owns the
    /// layout, and going through it keeps the group-relative dataset name out
    /// of caller code.
    block_extent local_block(const std::string& path) {
        std::size_t offset = 0;
        std::size_t count = 0;
        detail::check(h5c_poffset(handle_, path.c_str(), &offset, &count), "poffset '" + path + "'");
        return block_extent{offset, count};
    }

    /// Every rank's boundaries: nprocs + 1 entries, starting at 0 and
    /// non-decreasing. Collective. Use local_block() when you only need your
    /// own slice.
    std::vector<std::int64_t> partition(const std::string& path) {
        std::size_t count = 0;
        detail::check(h5c_ppartition(handle_, path.c_str(), nullptr, 0, &count), "ppartition '" + path + "' (length)");

        std::vector<std::int64_t> out(count);
        detail::check(
            h5c_ppartition(handle_, path.c_str(), out.data(), out.size(), nullptr),
            "ppartition '" + path + "'"
        );
        return out;
    }

    /// Local query, not collective.
    bool exists(const std::string& path) const {
        return h5c_exists(handle_, path.c_str()) != 0;
    }

    // -- interop ---------------------------------------------------------

    /// Borrowed h5c handle; h5cpp keeps ownership.
    h5c_file_t* c_handle() const noexcept { return handle_; }

    /// Borrowed HDF5 id; do not close it.
    hid_t hid() const noexcept { return h5c_file_hid(handle_); }

   private:
    void discard() noexcept {
        if (handle_ != nullptr) {
            h5c_close(handle_);  // status deliberately ignored in a destructor
            handle_ = nullptr;
        }
    }

    static dataset_info convert(const h5c_dataset_info_t& raw) {
        dataset_info out;
        out.dims.assign(raw.dims, raw.dims + raw.rank);
        out.type = raw.type;
        out.count = raw.count;
        return out;
    }

    static void require_size(const std::string& path, std::size_t given, std::size_t wanted) {
        if (given != wanted) {
            throw error(H5C_ERR_SHAPE_MISMATCH, "buffer for '" + path + "' holds " + std::to_string(given) + " elements but the shape needs " + std::to_string(wanted));
        }
    }

    h5c_file_t* handle_ = nullptr;
};

}  // namespace h5cpp

#endif  // H5CPP_MPI_HPP
