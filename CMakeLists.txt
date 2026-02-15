cmake_minimum_required(VERSION 3.20)
project(bh_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(OpenMP)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
  OUTPUT_VARIABLE BHSIM_MODULE_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT BHSIM_MODULE_SUFFIX)
  set(BHSIM_MODULE_SUFFIX ".so")
endif()

add_library(bhsim MODULE
  src/bindings.cpp
  src/raytrace.cpp
)

target_link_libraries(bhsim PRIVATE pybind11::headers Python::Module)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bhsim PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(bhsim PRIVATE BHSIM_USE_OPENMP=1)
endif()
target_include_directories(bhsim PRIVATE ${Python_INCLUDE_DIRS})
target_compile_definitions(bhsim PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

set_target_properties(bhsim PROPERTIES
  PREFIX ""
  SUFFIX "${BHSIM_MODULE_SUFFIX}"
)
