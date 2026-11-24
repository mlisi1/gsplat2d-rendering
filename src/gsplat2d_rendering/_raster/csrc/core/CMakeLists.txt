# SPDX-License-Identifier: Apache-2.0
#
# Standalone build of the surfel rasterizer core, for C++ users (e.g.
# Gazebo plugins); no Python or torch needed.
#
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="86;87;120" [-DBUILD_SHARED_LIBS=ON]
#   cmake --build build && cmake --install build --prefix <prefix>
#
# then, in a consumer: find_package(surfel_raster) and link
# surfel_raster::surfel_raster.
cmake_minimum_required(VERSION 3.24)
project(surfel_raster VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
find_package(CUDAToolkit REQUIRED)

add_library(surfel_raster
  preprocess.cu binning.cu raster.cu render.cu scratch_arena.cu)
add_library(surfel_raster::surfel_raster ALIAS surfel_raster)
set_target_properties(surfel_raster PROPERTIES
  CXX_STANDARD 17 CUDA_STANDARD 17 POSITION_INDEPENDENT_CODE ON)
target_compile_options(surfel_raster PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_include_directories(surfel_raster PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(surfel_raster PUBLIC CUDA::cudart)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS surfel_raster EXPORT surfel_rasterTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES surfel_raster.h scratch_arena.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT surfel_rasterTargets NAMESPACE surfel_raster::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/surfel_raster)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/surfel_rasterConfig.cmake
  "include(CMakeFindDependencyMacro)\nfind_dependency(CUDAToolkit)\n"
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/surfel_rasterTargets.cmake\")\n")
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/surfel_rasterConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/surfel_rasterConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/surfel_rasterConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/surfel_raster)
