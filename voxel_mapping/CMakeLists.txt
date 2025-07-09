cmake_minimum_required(VERSION 3.10)
project(voxel_mapping VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75;86)

add_library(voxel_mapping SHARED
  src/voxel_mapping.cpp
  src/voxel_mapping.cu
)

target_link_libraries(voxel_mapping PUBLIC
  CUDA::cudart
  Eigen3::Eigen
)

target_include_directories(voxel_mapping PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
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