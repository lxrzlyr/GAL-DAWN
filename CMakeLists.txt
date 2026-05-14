cmake_minimum_required(VERSION 3.16)

project(dawn VERSION 2.4.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(DAWN_BUILD_TESTS "Build and register DAWN tests" OFF)
option(DAWN_BUILD_CLI "Build the unified dawn command-line executable" ON)
option(DAWN_ENABLE_OPENMP "Enable OpenMP when available" ON)
option(DAWN_ENABLE_CUDA "Build CUDA targets" OFF)
option(DAWN_ENABLE_INSTALL "Enable install/export rules" OFF)
option(DAWN_ENABLE_WARNINGS "Enable compiler warnings for DAWN targets" ON)
option(DAWN_WARNINGS_AS_ERRORS "Treat DAWN warnings as errors" OFF)
option(DAWN_ENABLE_SANITIZERS "Enable ASan/UBSan for CPU targets" OFF)
option(DAWN_ENABLE_FORMAT_CHECK "Add clang-format check target" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(DAWN_PRODUCTION_TARGETS)

function(dawn_register_target target)
  list(APPEND DAWN_PRODUCTION_TARGETS ${target})
  set(DAWN_PRODUCTION_TARGETS "${DAWN_PRODUCTION_TARGETS}" PARENT_SCOPE)
endfunction()

function(dawn_apply_target_options target)
  if(DAWN_ENABLE_WARNINGS)
    if(MSVC)
      target_compile_options(${target} PRIVATE /W4)
      if(DAWN_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE /WX)
      endif()
    else()
      target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
      if(DAWN_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE -Werror)
      endif()
    endif()
  endif()

  if(DAWN_ENABLE_SANITIZERS AND NOT DAWN_ENABLE_CUDA)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${target} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=undefined)
      else()
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
      endif()
    endif()
  endif()
endfunction()

function(dawn_add_internal_includes target)
  target_include_directories(${target} PRIVATE
      "${PROJECT_SOURCE_DIR}/src")
endfunction()

add_library(dawn_common STATIC
    src/status.cpp)
target_include_directories(dawn_common PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_definitions(dawn_common PUBLIC
    DAWN_VERSION_STRING="${PROJECT_VERSION}")
dawn_register_target(dawn_common)

add_library(dawn_memory INTERFACE)
target_link_libraries(dawn_memory INTERFACE dawn_common)
dawn_register_target(dawn_memory)

add_library(dawn_graph STATIC
    src/graph.cpp
    src/legacy/matrix.cpp)
target_link_libraries(dawn_graph PUBLIC dawn_memory)
dawn_add_internal_includes(dawn_graph)
dawn_register_target(dawn_graph)

add_library(dawn_io STATIC
    src/io.cpp
    src/legacy/io.cpp)
target_link_libraries(dawn_io PUBLIC dawn_graph)
dawn_add_internal_includes(dawn_io)
dawn_register_target(dawn_io)

add_library(dawn_runtime STATIC
    src/runtime.cpp)
target_link_libraries(dawn_runtime PUBLIC dawn_io dawn_common dawn_memory)
dawn_register_target(dawn_runtime)

set(DAWN_KERNELS_CPU_SOURCES
    src/cpu_kernels.cpp)

add_library(dawn_kernels_cpu STATIC
    ${DAWN_KERNELS_CPU_SOURCES})
target_link_libraries(dawn_kernels_cpu PUBLIC dawn_graph)
dawn_add_internal_includes(dawn_kernels_cpu)
dawn_register_target(dawn_kernels_cpu)

set(DAWN_ALGORITHMS_CPU_SOURCES
    src/algorithms.cpp)

add_library(dawn_algorithms_cpu STATIC
    ${DAWN_ALGORITHMS_CPU_SOURCES})
target_link_libraries(dawn_algorithms_cpu PUBLIC dawn_kernels_cpu dawn_runtime)
dawn_add_internal_includes(dawn_algorithms_cpu)
dawn_register_target(dawn_algorithms_cpu)

set(DAWN_CUDA_HOST_OPENMP_OPTIONS "")
set(DAWN_PACKAGE_HAS_OPENMP OFF)
if(DAWN_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(DAWN_PACKAGE_HAS_OPENMP ON)
    foreach(target dawn_common dawn_memory dawn_graph dawn_io dawn_kernels_cpu dawn_algorithms_cpu)
      target_compile_definitions(${target} PUBLIC DAWN_HAS_OPENMP=1)
      target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endforeach()
    set(DAWN_CUDA_HOST_OPENMP_OPTIONS "-Xcompiler=-fopenmp")
    message(STATUS "DAWN: OpenMP found. CPU targets use OpenMP.")
  else()
    target_compile_definitions(dawn_common PUBLIC DAWN_HAS_OPENMP=0)
    message(STATUS "DAWN: OpenMP not found. CPU targets use single-thread fallback.")
  endif()
else()
  target_compile_definitions(dawn_common PUBLIC DAWN_HAS_OPENMP=0)
  message(STATUS "DAWN: OpenMP disabled. CPU targets use single-thread fallback.")
endif()

if(DAWN_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(DAWN_CUDA_ARCHITECTURES
      "86"
      CACHE STRING "CUDA architectures used by DAWN CUDA targets")

  add_library(dawn_kernels_cuda STATIC
      src/legacy/apsp_gpu.cu
      src/legacy/bfs_gpu.cu
      src/legacy/cc_gpu.cu
      src/legacy/mssp_gpu.cu
      src/legacy/sssp_gpu.cu)
  target_link_libraries(dawn_kernels_cuda PUBLIC dawn_io CUDA::cudart)
  dawn_add_internal_includes(dawn_kernels_cuda)
  target_compile_definitions(dawn_kernels_cuda PUBLIC DAWN_HAS_CUDA=1)
  set_target_properties(dawn_kernels_cuda PROPERTIES
      CUDA_ARCHITECTURES "${DAWN_CUDA_ARCHITECTURES}")
  dawn_register_target(dawn_kernels_cuda)

  add_library(dawn_algorithms_cuda INTERFACE)
  target_link_libraries(dawn_algorithms_cuda INTERFACE dawn_kernels_cuda)
  dawn_register_target(dawn_algorithms_cuda)
else()
  add_library(dawn_kernels_cuda INTERFACE)
  add_library(dawn_algorithms_cuda INTERFACE)
  dawn_register_target(dawn_kernels_cuda)
  dawn_register_target(dawn_algorithms_cuda)
  target_compile_definitions(dawn_common PUBLIC DAWN_HAS_CUDA=0)
  message(STATUS "DAWN: CUDA disabled. Configure with -DDAWN_ENABLE_CUDA=ON to build CUDA targets.")
endif()

foreach(target ${DAWN_PRODUCTION_TARGETS})
  get_target_property(target_type ${target} TYPE)
  if(NOT target_type STREQUAL "INTERFACE_LIBRARY")
    dawn_apply_target_options(${target})
  endif()
endforeach()

if(DAWN_BUILD_CLI)
  add_executable(dawn_cli src/dawn.cpp)
  set_target_properties(dawn_cli PROPERTIES OUTPUT_NAME dawn)
  target_link_libraries(dawn_cli PRIVATE dawn_algorithms_cpu)
  dawn_apply_target_options(dawn_cli)
endif()

if(DAWN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(validation)
endif()

if(DAWN_ENABLE_FORMAT_CHECK)
  add_custom_target(dawn_format_check
      COMMAND ${CMAKE_COMMAND}
          -DDAWN_FORMAT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
          -P "${PROJECT_SOURCE_DIR}/cmake/clang-format.cmake"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

if(DAWN_ENABLE_INSTALL)
  set(DAWN_INSTALL_LIBRARIES
      dawn_algorithms_cpu
      dawn_kernels_cpu
      dawn_runtime
      dawn_io
      dawn_graph
      dawn_memory
      dawn_common)
  configure_package_config_file(
      "${PROJECT_SOURCE_DIR}/cmake/dawnConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/dawnConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/dawn")
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/dawnConfigVersion.cmake"
      VERSION "${PROJECT_VERSION}"
      COMPATIBILITY SameMajorVersion)

  install(TARGETS ${DAWN_INSTALL_LIBRARIES}
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  if(TARGET dawn_cli)
    install(TARGETS dawn_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  endif()
  install(DIRECTORY include/dawn DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
      PATTERN "algorithm" EXCLUDE
      PATTERN "adapters" EXCLUDE)
  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/dawnConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/dawnConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/dawn")
endif()
