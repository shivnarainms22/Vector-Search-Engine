#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "vector_index.h"

namespace py = pybind11;

PYBIND11_MODULE(_vsearch, m) {
    m.doc() = "CUDA-accelerated vector search engine";

    py::enum_<KernelType>(m, "KernelType")
        .value("NAIVE",     KernelType::NAIVE)
        .value("SMEM",      KernelType::SMEM)
        .value("COALESCED", KernelType::COALESCED)
        .value("WARP",      KernelType::WARP)
        .export_values();

    py::class_<VectorIndex>(m, "VectorIndex")
        .def(py::init<int>(), py::arg("dim"))
        .def("add", [](VectorIndex& self,
                       py::array_t<float, py::array::c_style | py::array::forcecast> vecs) {
            if (vecs.ndim() != 2)
                throw std::invalid_argument("vectors must be 2-D");
            if (vecs.shape(1) != self.dim())
                throw std::invalid_argument("vector dim mismatch");
            self.add(vecs.data(), static_cast<int>(vecs.shape(0)));
        }, py::arg("vectors"))
        .def("search", [](VectorIndex& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                          int k,
                          const std::string& kernel) {
            if (queries.ndim() != 2)
                throw std::invalid_argument("queries must be 2-D");
            if (queries.shape(1) != self.dim())
                throw std::invalid_argument("query dim mismatch");

            KernelType ktype;
            if      (kernel == "naive")     ktype = KernelType::NAIVE;
            else if (kernel == "smem")      ktype = KernelType::SMEM;
            else if (kernel == "coalesced") ktype = KernelType::COALESCED;
            else if (kernel == "warp")      ktype = KernelType::WARP;
            else throw std::invalid_argument("unknown kernel: " + kernel);

            int q = static_cast<int>(queries.shape(0));
            SearchResult res = self.search(queries.data(), q, k, ktype);

            auto dist_arr = py::array_t<float>({q, k});
            auto idx_arr  = py::array_t<int32_t>({q, k});
            std::copy(res.distances.begin(), res.distances.end(),
                      dist_arr.mutable_data());
            std::copy(res.indices.begin(), res.indices.end(),
                      idx_arr.mutable_data());
            return py::make_tuple(dist_arr, idx_arr);
        }, py::arg("queries"), py::arg("k"), py::arg("kernel") = "warp")
        .def_property_readonly("dim", &VectorIndex::dim)
        .def_property_readonly("n",   &VectorIndex::n);
}
