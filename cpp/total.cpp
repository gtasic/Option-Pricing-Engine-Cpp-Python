#include "total.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(finance, m) {
    m.doc() = "Pricing & Greeks";

    // Structs (init vide + champs)
    py::class_<BS_parametres>(m, "BS_parametres")
      .def(py::init<double,double,double,double,double>())
      .def_readwrite("S0", &BS_parametres::S0)
      .def_readwrite("K",  &BS_parametres::K)
      .def_readwrite("T",  &BS_parametres::T)
      .def_readwrite("r",  &BS_parametres::r)
      .def_readwrite("sigma", &BS_parametres::sigma);

    py::class_<MC_parametres>(m, "MC_parametres")
      .def(py::init<>(double,double,double,double,double,int,int))
      .def_readwrite("nb_simulations", &MC_parametres::nb_simulations)
      .def_readwrite("nb_paths", &MC_parametres::nb_paths)
      .def_readwrite("S0", &MC_parametres::S0)
      .def_readwrite("K",  &MC_parametres::K)
      .def_readwrite("T",  &MC_parametres::T)
      .def_readwrite("r",  &MC_parametres::r)
      .def_readwrite("sigma", &MC_parametres::sigma);

    py::class_<tree_parametres>(m, "tree_parametres")
      .def(py::init<>(double,double,double,double,double,int))
      .def_readwrite("S0", &tree_parametres::S0)
      .def_readwrite("K",  &tree_parametres::K)
      .def_readwrite("T",  &tree_parametres::T)
      .def_readwrite("r",  &tree_parametres::r)
      .def_readwrite("sigma", &tree_parametres::sigma)
      .def_readwrite("N",  &tree_parametres::N);

    py::class_<D>(m, "D")
      .def_readonly("d1", &D::d1)
      .def_readonly("d2", &D::d2);

    // Utils
    m.def("inv_sqrt_2pi", static_cast<double(*)()>(&inv_sqrt_2pi));
    m.def("norm_pdf",     static_cast<double(*)(double)>(&norm_pdf));
    m.def("norm_cdf",     static_cast<double(*)(double)>(&norm_cdf));

    // Black–Scholes (cast explicite)
    m.def("call_price", static_cast<double(*)(BS_parametres)>(&call_price));
    m.def("put_price",  static_cast<double(*)(BS_parametres)>(&put_price));
    m.def("call_delta", static_cast<double(*)(BS_parametres)>(&call_delta));
    m.def("call_gamma", static_cast<double(*)(BS_parametres)>(&call_gamma));
    m.def("call_vega",  static_cast<double(*)(BS_parametres)>(&call_vega));
    m.def("call_theta", static_cast<double(*)(BS_parametres)>(&call_theta));
    m.def("call_rho",   static_cast<double(*)(BS_parametres)>(&call_rho));
    m.def("put_delta",  static_cast<double(*)(BS_parametres)>(&put_delta));
    m.def("put_gamma",  static_cast<double(*)(BS_parametres)>(&put_gamma));
    m.def("put_vega",   static_cast<double(*)(BS_parametres)>(&put_vega));
    m.def("put_theta",  static_cast<double(*)(BS_parametres)>(&put_theta));
    m.def("put_rho",    static_cast<double(*)(BS_parametres)>(&put_rho));
    m.def("theta_per_day", &theta_per_day);
    m.def("vega_per_1pct", &vega_per_1pct);

    m.def("monte_carlo_call",
          static_cast<double(*)(const MC_parametres&)>(&monte_carlo_call));
    m.def("tree",
          static_cast<double(*)(tree_parametres)>(&tree));
}
