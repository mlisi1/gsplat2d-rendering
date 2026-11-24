# SPDX-License-Identifier: Apache-2.0
# Consumer-side check of the installed C++ package:
#   cmake -S tests/cpp -B build-cpp -DCMAKE_PREFIX_PATH=<install prefix> && cmake --build build-cpp && build-cpp/smoke
cmake_minimum_required(VERSION 3.24)
project(surfel_raster_smoke LANGUAGES CXX)
find_package(surfel_raster REQUIRED)
add_executable(smoke smoke.cpp)
set_target_properties(smoke PROPERTIES CXX_STANDARD 17)
target_link_libraries(smoke PRIVATE surfel_raster::surfel_raster)
