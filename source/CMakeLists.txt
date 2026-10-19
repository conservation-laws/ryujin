##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2026 by the ryujin authors
##

configure_file(
  ${CMAKE_SOURCE_DIR}/include/ryujin/base/compile_time_options.h.in
  ${CMAKE_BINARY_DIR}/include/ryujin/base/compile_time_options.h
  )

#
# Common and equation-independent source files and target:
#

set(COMMON_SOURCE_FILES
  base/mpi_ensemble.cc
  base/simd.cc
  base/version_info.cc
  discretization/discretization.cc
  discretization/mesh_adaptor.cc
  discretization/offline_data.cc
  discretization/time_integrator.cc
  driver/time_loop.cc
  geometries/geometry_library.cc
  linear_algebra/multicomponent_vector.cc
  linear_algebra/sparsity_pattern.cc
  output/error_evaluation.cc
  output/quantities.cc
  output/vtu_output.cc
  )

set_property(SOURCE base/version_info.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

add_library(obj_common OBJECT ${COMMON_SOURCE_FILES})
target_include_directories(obj_common PUBLIC
  ${CMAKE_BINARY_DIR}/include/
  ${CMAKE_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
deal_ii_setup_target(obj_common)
target_link_libraries(obj_common ${EXTERNAL_TARGETS})

#
# Equation-specific source files and compilation targets:
#

set(OBJECT_TARGETS)
set(EQUATION_DEFINITIONS)
macro(setup_equation EQUATION)
  #
  # obj_${EQUATION} object library: equation specific code:
  #

  add_subdirectory(${EQUATION})

  list(APPEND OBJECT_TARGETS obj_${EQUATION})
endmacro()

file(GLOB _files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS */CMakeLists.txt)
foreach(_file ${_files})
  get_filename_component(_directory "${_file}" DIRECTORY)

  string(TOUPPER "${_directory}" _directory_upper)
  set(EQUATION_${_directory_upper} ${DEFAULT_ENABLE_EQUATIONS} CACHE BOOL "Enable the equation »${_directory}« compute module")

  if(EQUATION_${_directory_upper})
    setup_equation("${_directory}")
    list(APPEND EQUATION_DEFINITIONS RYUJIN_WITH_EQUATION_${_directory_upper})
  endif()
endforeach()

#
# The library of all enabled equations and the entry point of the
# executable:
#

add_library(obj_dispatch OBJECT
  driver/equation_library.cc
  driver/run.cc
  )
target_compile_definitions(obj_dispatch PRIVATE ${EQUATION_DEFINITIONS})
deal_ii_setup_target(obj_dispatch)
target_link_libraries(obj_dispatch obj_common ${EXTERNAL_TARGETS})

#
# The executable
#

add_executable(ryujin main.cc)
deal_ii_setup_target(ryujin)
target_link_libraries(ryujin obj_common obj_dispatch ${OBJECT_TARGETS} ${EXTERNAL_TARGETS})

set_target_properties(ryujin PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/run"
  )

install(TARGETS ryujin DESTINATION ${CMAKE_INSTALL_BINDIR})
