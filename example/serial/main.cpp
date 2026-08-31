// Serial read and write.
//
// The same story as h5c's serial example, but the C++ layer carries the type,
// sizes the buffers, and turns failures into exceptions.
#include <h5cpp/h5cpp.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    const std::string path = "example_serial.h5";

    try {
        // shape is ROW-MAJOR: the last extent varies fastest.
        // This is the C++ view of what Fortran declares as a(2, 3).
        const std::vector<double> values = { 1, 2, 3, 4, 5, 6 };
        const h5cpp::shape        dims   = { 3, 2 };

        // Note h5c_bool_t, not bool: std::vector<bool> is a bitset and has no
        // contiguous buffer. Using bool here is a compile error with an
        // explanatory message.
        const std::vector<h5c_bool_t> flags = { H5C_TRUE, H5C_FALSE,
                                                H5C_FALSE, H5C_TRUE };

        // ---- write --------------------------------------------------
        {
            h5cpp::file f(path, h5cpp::mode::truncate);

            // The container overload checks the buffer length against dims.
            f.write("/mesh/coords", values, dims);
            f.write_scalar<double>("/time", 0.125);
            f.write("/mesh/active", flags, {2, 2});
            f.write_string("/title", "example field");

            // Attributes are written separately from the data they annotate.
            f.write_attr_str("/mesh/coords", "units", "m");
            f.write_attr_scalar<double>("/", "time", 0.125);

            // close() is explicit here so a flush failure is observable.
            // The destructor would close too, but it never throws.
            f.close();
        }

        // ---- read back ----------------------------------------------
        {
            h5cpp::file f(path, h5cpp::mode::read);

            const h5cpp::dataset_info meta = f.info("/mesh/coords");
            std::cout << "/mesh/coords: rank=" << meta.rank()
                      << " dims={" << meta.dims[0] << ", " << meta.dims[1] << "}"
                      << " count=" << meta.count << '\n';

            // read<T>() sizes the vector from the stored shape.
            const std::vector<double> got = f.read<double>("/mesh/coords");
            std::cout << "values:";
            for (double x : got) {
                std::cout << ' ' << x;
            }
            std::cout << '\n';

            std::cout << "time: " << f.read_scalar<double>("/time") << '\n';
            std::cout << "title: " << f.read_string("/title") << '\n';
            std::cout << "units: "
                      << f.read_attr_str("/mesh/coords", "units") << '\n';
            std::cout << "root time attr: "
                      << f.read_attr_scalar<double>("/", "time") << '\n';
            std::cout << "exists /nope: " << std::boolalpha
                      << f.exists("/nope") << '\n';

            // A missing dataset throws, and the status says why.
            try {
                (void)f.read<double>("/not/here");
            } catch (const h5cpp::error& e) {
                std::cout << "missing dataset -> "
                          << h5c_status_string(e.status()) << '\n';
            }

            f.close();
        }
    } catch (const h5cpp::error& e) {
        std::cerr << "h5cpp error: " << e.what() << '\n';
        return 1;
    }

    std::cout << "wrote and read " << path << '\n';
    return 0;
}
