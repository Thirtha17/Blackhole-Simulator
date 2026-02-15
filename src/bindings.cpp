#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "raytrace.hpp"

namespace py = pybind11;

PYBIND11_MODULE(bhsim, m) {
    m.doc() = "Black hole sky renderer (sky + BH disk + simple lensing)";

    m.def("render_sky_with_bh", &render_sky_with_bh,
          py::arg("width"),
          py::arg("height"),
          py::arg("fov_deg"),
          py::arg("bh_angular_radius_deg"),
          py::arg("lens_strength"),
          py::arg("sky_img"),
          py::arg("cam_yaw_deg") = 0.0,
          py::arg("cam_pitch_deg") = 6.0,
          py::arg("cam_distance_scale") = 1.0);
}
