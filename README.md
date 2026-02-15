# Blackhole Simulator

A C++/Python black hole renderer using `pybind11`.

It builds a Python extension module (`bhsim`) from C++ code and renders a black-hole + accretion-disk frame from Python.

## Project Description

`Blackhole Simulator` is a black hole visualization project focused on physically inspired rendering with a fast C++ core and a simple Python interface.  
It traces light paths around a compact mass, samples emission from an accretion disk, and produces cinematic frames that show gravitational lensing, shadow formation, and relativistic intensity shifts.

The renderer is implemented in C++ for performance, exposed to Python through `pybind11`, and designed for quick experimentation with camera and lensing parameters. This makes it suitable for both educational demos and visual prototyping of black hole imagery.

## Project Structure

- `src/raytrace.cpp`: core renderer logic (C++)
- `src/raytrace.hpp`: renderer function declaration
- `src/bindings.cpp`: `pybind11` Python bindings
- `python/demo.py`: example Python script to render and display a frame
- `CMakeLists.txt`: build configuration

## Requirements

- C++17 compiler
- CMake >= 3.20
- Python (recommended: one virtualenv/conda env)
- `pybind11` available to CMake
- Python packages:
  - `numpy`
  - `matplotlib`

## Build

From repo root:

```bash
cmake -S . -B build \
  -DPython_EXECUTABLE=$(which python)
cmake --build build -j
```

Notes:

- Always configure and run with the **same Python interpreter** to avoid ABI mismatch.
- The extension is written into `build/` (e.g. `bhsim.cpython-312-darwin.so`).

## Run Demo

```bash
PYTHONPATH=build python python/demo.py
```

The demo opens a matplotlib window with the rendered frame.

## Python API

```python
import bhsim
img = bhsim.render_sky_with_bh(
    width=1400,
    height=780,
    fov_deg=60.0,
    bh_angular_radius_deg=4.8,
    lens_strength=0.55,
    sky_img=dummy_numpy_uint8_image
)
```

Return value:

- `numpy.ndarray` of shape `(height, width, 4)` in RGBA (`uint8`).

## Troubleshooting

### 1) `ImportError: Python version mismatch`

Reconfigure with the same Python you use to run:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(which python)
cmake --build build -j
```

### 2) `ModuleNotFoundError: No module named 'bhsim'`

Ensure module path is visible:

```bash
PYTHONPATH=build python python/demo.py
```

### 3) Missing Python packages

Install in your active environment:

```bash
python -m pip install numpy matplotlib
```

## License

This project is licensed under the MIT License. See `LICENSE`.
