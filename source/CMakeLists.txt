##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2022 by the ryujin authors
##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )

#
# Common and equation-independent source files and target:
#

set(COMMON_SOURCE_FILES
  discretization.cc
  multicomponent_vector.cc
  offline_data.cc
  simd.cc
  sparse_matrix_simd.cc
  )

add_library(obj_common OBJECT ${COMMON_SOURCE_FILES})
target_include_directories(obj_common PUBLIC
  ${CMAKE_BINARY_DIR}/source/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
deal_ii_setup_target(obj_common)

#
# Common, equation-dependent source files:
#

set(DEPENDENT_SOURCE_FILES
  hyperbolic_module.cc
  initial_values.cc
  postprocessor.cc
  quantities.cc
  time_integrator.cc
  time_loop.cc
  vtu_output.cc
  )

set_property(SOURCE time_loop.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

#
# Equation-specific source files and compilation targets:
#

macro(setup_equation EQUATION)
  #
  # obj_${EQUATION} object library: equation specific code:
  #

  add_subdirectory(${EQUATION})

  #
  # obj_${EQUATION}_dependent object library: equation dependent common code:
  #

  add_library(obj_${EQUATION}_dependent OBJECT ${DEPENDENT_SOURCE_FILES})
  deal_ii_setup_target(obj_${EQUATION}_dependent)

  # Include the obj_${EQUATION} objects with PRIVATE visibility
  target_link_libraries(obj_${EQUATION}_dependent PRIVATE obj_${EQUATION})

  if(WITH_LIKWID)
    target_link_libraries(obj_${EQUATION}_dependent PUBLIC likwid likwid-hwloc likwid-lua)
  endif()
  # For GNU libstdc++ we have to make sure to also link against libstdc++fs.
  if(LIBSTDCPP)
    target_link_libraries(obj_${EQUATION}_dependent PUBLIC stdc++fs)
  endif()
endmacro()

setup_equation("euler")
setup_equation("euler_aeos")
setup_equation("shallow_water")
