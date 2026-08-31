// Vector fields held as separate component arrays.
//
// Solvers keep u, v, w apart, but XDMF wants one [n, 3] dataset so that
// ParaView sees a vector. h5cpp takes the components as views, so the length
// mismatch that the C API cannot detect becomes an exception here.
#include <h5cpp/h5cpp.hpp>
#include <iostream>
#include <vector>

int main() {
    const std::string path = "example_serial_interleaved.h5";

    try {
        // Separate components, as a solver would hold them.
        std::vector<double> u = {1, 2, 3, 4};
        std::vector<double> v = {10, 20, 30, 40};
        std::vector<double> w = {100, 200, 300, 400};

        {
            h5cpp::file f(path, h5cpp::mode::truncate);

            // n is derived from the views; no chance to pass a wrong count.
            f.write_interleaved<double>("/fields/velocity", {u, v, w});
            f.write_attr_str("/fields/velocity", "attribute_type", "Vector");
            f.close();
        }

        {
            h5cpp::file f(path, h5cpp::mode::read);

            std::vector<double> gu(u.size()), gv(v.size()), gw(w.size());
            f.read_interleaved<double>("/fields/velocity", {gu, gv, gw});

            std::cout << "components:\n";
            for (std::size_t i = 0; i < gu.size(); ++i) {
                std::cout << "  " << gu[i] << ' ' << gv[i] << ' ' << gw[i]
                          << '\n';
            }

            // The file really is interleaved: read as a plain [n, 3] dataset
            // it gives u0 v0 w0 u1 v1 w1 ...
            const std::vector<double> flat = f.read<double>("/fields/velocity");
            std::cout << "as stored:";
            for (double x : flat) {
                std::cout << ' ' << x;
            }
            std::cout << '\n';

            // Only one component moves only its own bytes.
            const std::vector<double> only_v =
                f.read_component<double>("/fields/velocity", 1);
            std::cout << "v only:";
            for (double x : only_v) {
                std::cout << ' ' << x;
            }
            std::cout << '\n';

            f.close();
        }

        // Ragged components are caught before any I/O happens.
        try {
            h5cpp::file f(path, h5cpp::mode::readwrite);
            std::vector<double> shorter = {1, 2};
            f.write_interleaved<double>("/bad", {u, v, shorter});
        } catch (const h5cpp::error& e) {
            std::cout << "ragged components -> "
                      << h5c_status_string(e.status()) << '\n';
        }
    } catch (const h5cpp::error& e) {
        std::cerr << "h5cpp error: " << e.what() << '\n';
        return 1;
    }

    std::cout << "wrote and read " << path << '\n';
    return 0;
}
