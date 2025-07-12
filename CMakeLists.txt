cmake_minimum_required(VERSION 3.30.4)
project(voxel_mapping VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75;86)

set(CCCL_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cccl/thrust
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cccl/cub
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cccl/libcudacxx/include
)

set(CUCO_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/submodules/cuco/include
)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

add_library(voxel_mapping SHARED
  src/voxel_mapping.cpp
  src/voxel_mapping.cu
  src/hash_map.cu
)

target_link_libraries(voxel_mapping PUBLIC
  CUDA::cudart
  Eigen3::Eigen
  spdlog::spdlog
  fmt::fmt
)

target_include_directories(voxel_mapping
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
  PRIVATE
    ${CUCO_INCLUDE_DIR}
    ${CCCL_INCLUDE_DIRS}
)

target_compile_options(voxel_mapping PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS voxel_mapping
  EXPORT export_voxel_mapping
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
    EXPORT export_voxel_mapping
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE voxel_mapping::
    DESTINATION share/${PROJECT_NAME}
)

configure_package_config_file(
  "cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "share/${PROJECT_NAME}"
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    DESTINATION "share/${PROJECT_NAME}"
)