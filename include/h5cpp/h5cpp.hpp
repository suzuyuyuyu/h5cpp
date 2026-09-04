// h5cpp — a modern C++ wrapper around h5c.
//
// h5cpp adds RAII, type safety and exceptions on top of h5c. It never calls
// HDF5 directly: everything goes through the h5c C API, which keeps parallel
// I/O available without depending on the official HDF5 C++ interface.
//
// Array conventions are inherited from h5c unchanged:
//   - buffers are FLAT, shapes are explicit;
//   - dimension order is ROW-MAJOR, shape.back() varies fastest;
//   - a Fortran array a(nx, ny) corresponds to shape {ny, nx}.
#ifndef H5CPP_HPP
#define H5CPP_HPP

#include <h5c/h5c.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#include <span>
#define H5CPP_HAS_STD_SPAN 1
#endif

namespace h5cpp {

// ---------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------

/// Thrown by every failing h5cpp operation. Carries the underlying h5c status
/// so callers can branch on the cause rather than parsing the message.
class error : public std::runtime_error {
   public:
    error(h5c_status_t status, std::string what)
        : std::runtime_error(std::move(what)), status_(status) {}

    h5c_status_t status() const noexcept { return status_; }

   private:
    h5c_status_t status_;
};

namespace detail {

/// Turns a non-OK status into an exception, attaching h5c's recorded message.
inline void check(h5c_status_t st, const std::string& context) {
    if (st == H5C_OK) {
        return;
    }
    const h5c_error_t* e = h5c_last_error();
    std::string msg = context + ": " + h5c_status_string(st);
    if (e != nullptr && e->message[0] != '\0') {
        msg += " (";
        msg += e->message;
        msg += ")";
    }
    throw error(st, std::move(msg));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// shapes and views
// ---------------------------------------------------------------------------

/// Extents of a dataset, slowest-varying first. Brace-initialisable:
/// `file.write("/v", data, {ny, nx})`.
using shape = std::vector<std::size_t>;

inline std::size_t element_count(const shape& s) {
    return std::accumulate(s.begin(), s.end(), std::size_t{1}, std::multiplies<std::size_t>());
}

#if defined(H5CPP_HAS_STD_SPAN)
template <class T>
using span = std::span<T>;
#else
/// Minimal non-owning view for C++17 builds. Deliberately tiny: it exists only
/// so the same signatures compile with and without std::span.
template <class T>
class span {
   public:
    using element_type = T;
    using size_type = std::size_t;

    constexpr span() noexcept : data_(nullptr), size_(0) {}
    constexpr span(T* p, size_type n) noexcept : data_(p), size_(n) {}

    template <class U, class = typename std::enable_if<std::is_convertible<U (*)[], T (*)[]>::value>::type>
    span(std::vector<U>& v) noexcept : data_(v.data()), size_(v.size()) {}

    template <class U, class = typename std::enable_if<std::is_convertible<const U (*)[], T (*)[]>::value>::type>
    span(const std::vector<U>& v) noexcept : data_(v.data()), size_(v.size()) {}

    constexpr T* data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr T* begin() const noexcept { return data_; }
    constexpr T* end() const noexcept { return data_ + size_; }
    constexpr T& operator[](size_type i) const { return data_[i]; }

   private:
    T* data_;
    size_type size_;
};
#endif

namespace detail {

/// Flattens component views to pointers and returns their common length.
/// Shared by the serial and parallel interleaved calls: catching a ragged
/// component list here is the whole reason those take views rather than a
/// bare pointer plus an unverifiable `n`.
template <class View, class Ptr>
inline std::size_t gather_components(const std::string& path, const std::vector<View>& comps, std::vector<Ptr>& ptrs) {
    if (comps.empty()) {
        throw error(H5C_ERR_INVALID_ARG, "interleaved '" + path + "' needs at least one component");
    }
    const std::size_t n = comps.front().size();
    ptrs.reserve(comps.size());
    for (const View& c : comps) {
        if (c.size() != n) {
            throw error(H5C_ERR_SHAPE_MISMATCH, "interleaved '" + path +
                                                    "': components have "
                                                    "different lengths (" +
                                                    std::to_string(n) + " and " + std::to_string(c.size()) + ")");
        }
        ptrs.push_back(c.data());
    }
    return n;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// datatype mapping
// ---------------------------------------------------------------------------

/// Maps a C++ type to its h5c type tag. Unsupported types fail to compile,
/// which is safer than a runtime check. The static_assert exists only to make
/// that failure readable; without it the error is an opaque "incomplete type".
template <class T>
struct type_of {
    static_assert(sizeof(T) == 0,
                  "h5cpp: unsupported element type. Use float, double, "
                  "std::int32_t, std::int64_t, or h5c_bool_t. Note that `bool` "
                  "is deliberately absent: std::vector<bool> is a bitset and "
                  "has no contiguous buffer to hand to HDF5 -- use "
                  "std::vector<h5c_bool_t> instead.");
};

template <>
struct type_of<float> {
    static constexpr h5c_type_t value = H5C_F32;
};
template <>
struct type_of<double> {
    static constexpr h5c_type_t value = H5C_F64;
};
template <>
struct type_of<std::int32_t> {
    static constexpr h5c_type_t value = H5C_I32;
};
template <>
struct type_of<std::int64_t> {
    static constexpr h5c_type_t value = H5C_I64;
};

// Booleans use h5c_bool_t (int8_t), never `bool`: std::vector<bool> is a
// bitset and has no contiguous buffer to hand to HDF5.
template <>
struct type_of<h5c_bool_t> {
    static constexpr h5c_type_t value = H5C_BOOL;
};

/// Shape and datatype of a stored dataset.
struct dataset_info {
    shape dims;
    h5c_type_t type = H5C_TYPE_UNKNOWN;
    std::size_t count = 0;

    int rank() const noexcept { return static_cast<int>(dims.size()); }
};

// ---------------------------------------------------------------------------
// file
// ---------------------------------------------------------------------------

enum class mode {
    read = H5C_READ,
    readwrite = H5C_READWRITE,
    truncate = H5C_TRUNCATE
};

/// Owns an h5c file handle. Move-only; the destructor closes without throwing.
class file {
   public:
    file(const std::string& path, mode m) {
        h5c_file_t* raw = nullptr;
        detail::check(h5c_open(path.c_str(), static_cast<h5c_mode_t>(m), &raw), "open '" + path + "'");
        handle_ = raw;
    }

    file(file&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }

    file& operator=(file&& other) noexcept {
        if (this != &other) {
            discard();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    file(const file&) = delete;
    file& operator=(const file&) = delete;

    /// Closes without throwing. Use close() when you need to observe failures.
    ~file() { discard(); }

    /// Closes and reports failure. Safe to call once; further calls are no-ops.
    ///
    /// h5c_close() reports only whether the close itself succeeded, so a
    /// throw here really does mean the file may not have been flushed. The
    /// sticky status that h5c keeps for C callers is redundant in h5cpp,
    /// because every failure has already surfaced as an exception.
    void close() {
        if (handle_ == nullptr) {
            return;
        }
        h5c_status_t st = h5c_close(handle_);
        handle_ = nullptr;
        detail::check(st, "close");
    }

    // -- writing ---------------------------------------------------------

    template <class T>
    void write(const std::string& path, const T* data, const shape& dims, bool replace = false) {
        detail::check(
            h5c_write(handle_, path.c_str(), data, type_of<T>::value, static_cast<int>(dims.size()), dims.empty() ? nullptr : dims.data(), replace ? H5C_WRITE_REPLACE : H5C_WRITE_DEFAULT),
            "write '" + path + "'"
        );
    }

    /// Container overload; checks that the buffer length matches the shape.
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

    /// One-dimensional shorthand.
    template <class T>
    void write(const std::string& path, const std::vector<T>& data, bool replace = false) {
        // Caught here rather than in type_of<bool>: std::vector<bool> has no
        // data() to instantiate against, so the guard must come first.
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        write<T>(path, data.data(), shape{data.size()}, replace);
    }

    template <class T>
    void write_scalar(const std::string& path, T value, bool replace = false) {
        write<T>(path, &value, shape{}, replace);
    }

    // -- reading ---------------------------------------------------------

    /// Reads into storage the caller already owns.
    template <class T>
    void read_into(const std::string& path, T* data, const shape& dims) {
        detail::check(
            h5c_read(handle_, path.c_str(), data, type_of<T>::value, static_cast<int>(dims.size()), dims.empty() ? nullptr : dims.data()),
            "read '" + path + "'"
        );
    }

    template <class T>
    void read_into(const std::string& path, std::vector<T>& data, const shape& dims) {
        // Caught here rather than in type_of<bool>: std::vector<bool> has no
        // data() to instantiate against, so the guard must come first.
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        require_size(path, data.size(), element_count(dims));
        T* raw = data.data();
        read_into<T>(path, raw, dims);
    }

    /// Reads a whole dataset, sizing the vector from the stored shape.
    template <class T>
    std::vector<T> read(const std::string& path) {
        dataset_info meta = info(path);
        std::vector<T> out(meta.count);
        read_into<T>(path, out.data(), meta.dims);
        return out;
    }

    template <class T>
    T read_scalar(const std::string& path) {
        T value{};
        read_into<T>(path, &value, shape{});
        return value;
    }

    // -- strings ---------------------------------------------------------

    /// Writes a scalar string dataset. Fixed-length by default, which is what
    /// h5fortran reads; pass vlen=true for a variable-length one, which it
    /// cannot.
    void write_string(const std::string& path, const std::string& value, bool replace = false, bool vlen = false) {
        const unsigned flags = replace ? H5C_WRITE_REPLACE : H5C_WRITE_DEFAULT;
        const h5c_status_t st =
            vlen ? h5c_write_string_vlen(handle_, path.c_str(), value.c_str(), flags)
                 : h5c_write_string(handle_, path.c_str(), value.c_str(), flags);
        detail::check(st, "write_string '" + path + "'");
    }

    /// Reads either representation. The std::string owns the result, so the
    /// h5c_free_string() the C API needs never reaches the caller.
    std::string read_string(const std::string& path) {
        char* raw = nullptr;
        detail::check(h5c_read_string(handle_, path.c_str(), &raw), "read_string '" + path + "'");
        std::string out = (raw != nullptr) ? raw : "";
        h5c_free_string(raw);
        return out;
    }

    // -- attributes ------------------------------------------------------

    /// `obj_path` may name a dataset, a group, or "/" for the root group.
    /// Attributes are written separately from the data they annotate.
    ///
    /// The string and numeric forms have distinct names on purpose: with a
    /// single overloaded name, a string literal would bind to the numeric
    /// template and fail with an unhelpful message.
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
    /// it cannot disagree with the storage passed to h5c.
    template <class T>
    void write_attr_array(const std::string& obj_path, const std::string& name, const std::vector<T>& values) {
        static_assert(!std::is_same<T, bool>::value,
                      "h5cpp: std::vector<bool> is a bitset and has no "
                      "contiguous buffer to hand to HDF5. Use "
                      "std::vector<h5c_bool_t> instead.");
        detail::check(h5c_write_attr_array(handle_, obj_path.c_str(), name.c_str(), values.data(), type_of<T>::value, values.size()), "write_attr_array '" + obj_path + ":" + name + "'");
    }

    /// Reads a numeric 1-D attribute, sizing the result from its stored length.
    template <class T>
    std::vector<T> read_attr_array(const std::string& obj_path, const std::string& name) {
        std::vector<T> out(attr_length(obj_path, name));
        detail::check(h5c_read_attr_array(handle_, obj_path.c_str(), name.c_str(), out.data(), type_of<T>::value, out.size()), "read_attr_array '" + obj_path + ":" + name + "'");
        return out;
    }

    /// Elements in an attribute: 1 for a scalar, n for an array.
    std::size_t attr_length(const std::string& obj_path, const std::string& name) {
        std::size_t count = 0;
        detail::check(h5c_attr_length(handle_, obj_path.c_str(), name.c_str(), &count), "attr_length '" + obj_path + ":" + name + "'");
        return count;
    }

    /// Local query, like exists(): never touches the sticky status.
    bool attr_exists(const std::string& obj_path, const std::string& name) const {
        return h5c_attr_exists(handle_, obj_path.c_str(), name.c_str()) != 0;
    }

    // -- interleaved multi-component I/O ---------------------------------

    /// Writes `ncomp` component arrays of `n` elements each as one
    /// [n, ncomp] dataset. Vector and tensor fields need this layout for
    /// XDMF to see them as vectors rather than as unrelated scalars.
    ///
    ///     f.write_interleaved<double>("/velocity", {u, v, w});
    ///
    /// The view overload derives `n` and rejects components of differing
    /// length -- the mistake that otherwise shows up as silently shuffled
    /// data, because the C API cannot verify `n`.
    template <class T>
    void write_interleaved(const std::string& path, const std::vector<const T*>& comps, std::size_t n, bool replace = false) {
        detail::check(
            h5c_write_interleaved(
                handle_, path.c_str(),
                reinterpret_cast<const void* const*>(comps.data()),
                comps.size(), n, type_of<T>::value,
                replace ? H5C_WRITE_REPLACE : H5C_WRITE_DEFAULT
            ),
            "write_interleaved '" + path + "'"
        );
    }

    template <class T>
    void write_interleaved(const std::string& path, const std::vector<span<const T>>& comps, bool replace = false) {
        std::vector<const T*> ptrs;
        const std::size_t n = detail::gather_components(path, comps, ptrs);
        write_interleaved<T>(path, ptrs, n, replace);
    }

    /// Scatters an interleaved dataset back into separate component arrays.
    template <class T>
    void read_interleaved(const std::string& path, const std::vector<T*>& comps, std::size_t n) {
        detail::check(
            h5c_read_interleaved(handle_, path.c_str(), reinterpret_cast<void* const*>(comps.data()), comps.size(), n, type_of<T>::value),
            "read_interleaved '" + path + "'"
        );
    }

    template <class T>
    void read_interleaved(const std::string& path, const std::vector<span<T>>& comps) {
        std::vector<T*> ptrs;
        const std::size_t n = detail::gather_components(path, comps, ptrs);
        read_interleaved<T>(path, ptrs, n);
    }

    /// Reads a single component. Only that component's bytes move: a strided
    /// read has no read-modify-write penalty, unlike a strided write.
    template <class T>
    std::vector<T> read_component(const std::string& path, std::size_t comp) {
        const dataset_info meta = info(path);
        if (meta.dims.size() != 2) {
            throw error(H5C_ERR_SHAPE_MISMATCH, "'" + path + "' is not an [n, ncomp] dataset");
        }
        std::vector<T> out(meta.dims[0]);
        detail::check(
            h5c_read_component(handle_, path.c_str(), out.data(), comp, meta.dims[0], type_of<T>::value),
            "read_component '" + path + "'"
        );
        return out;
    }

    // -- metadata --------------------------------------------------------

    dataset_info info(const std::string& path) {
        h5c_dataset_info_t raw;
        detail::check(h5c_dataset_info(handle_, path.c_str(), &raw), "info '" + path + "'");

        dataset_info out;
        out.dims.assign(raw.dims, raw.dims + raw.rank);
        out.type = raw.type;
        out.count = raw.count;
        return out;
    }

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

    static void require_size(const std::string& path, std::size_t given, std::size_t wanted) {
        if (given != wanted) {
            throw error(H5C_ERR_SHAPE_MISMATCH, "buffer for '" + path + "' holds " + std::to_string(given) + " elements but the shape needs " + std::to_string(wanted));
        }
    }

    h5c_file_t* handle_ = nullptr;
};

}  // namespace h5cpp

#endif  // H5CPP_HPP
