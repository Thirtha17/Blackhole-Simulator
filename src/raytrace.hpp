#pragma once
#include <cstdint>
#include <pybind11/numpy.h>

pybind11::array_t<std::uint8_t> render_sky_with_bh(
    int width,
    int height,
    double fov_deg,
    double bh_angular_radius_deg,
    double lens_strength,
    pybind11::array_t<std::uint8_t> sky_img,
    double cam_yaw_deg,
    double cam_pitch_deg,
    double cam_distance_scale
);
