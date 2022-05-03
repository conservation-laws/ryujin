##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2021 by the ryujin authors
##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# Detect whether we are building against GCC's libstdc++.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  #include <iostream>
  #ifndef __GLIBCXX__
  #error NOT LIBSTDCPP
  #endif
  int main() {}"
  LIBSTDCPP
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
deal_ii_setup_target(obj_common)

#
# Common, equation-dependent source files:
#

set(DEPENDENT_SOURCE_FILES
  hyperbolic_module.cc
  initial_values.cc
  main.cc
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
# Equation-specific source files:
#

if("${INSTANTIATED_EQUATIONS}" STREQUAL "")
  set(INSTANTIATED_EQUATIONS
    "euler"
    )
endif()

foreach(EQUATION ${INSTANTIATED_EQUATIONS})
  # define obj_${EQUATION} object library:
  add_subdirectory(${EQUATION})

  add_executable(${EQUATION} ${DEPENDENT_SOURCE_FILES})
  deal_ii_setup_target(${EQUATION})
  target_link_libraries(${EQUATION}
    obj_common obj_${EQUATION} ${OpenMP_CXX_LIBRARIES}
    )

  set_property(TARGET ${EQUATION}
    PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run
    )

  if(LIKWID_PERFMON)
    target_link_libraries(${EQUATION} likwid likwid-hwloc likwid-lua)
  endif()

  # For GNU libstdc++ we have to make sure to also link against libstdc++fs.
  if(LIBSTDCPP)
    target_link_libraries(${EQUATION} stdc++fs)
  endif()

  install(TARGETS ${EQUATION} DESTINATION ${CMAKE_INSTALL_BINDIR})
endmacro()

setup_equation("euler")
setup_equation("shallow_water")
