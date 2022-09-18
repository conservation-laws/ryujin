##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2022 by the ryujin authors
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
# Equation-specific source files:
#

macro(setup_equation EQUATION)
  # define obj_${EQUATION} object library:
  add_subdirectory(${EQUATION})

  add_executable(${EQUATION} ${DEPENDENT_SOURCE_FILES})
  deal_ii_setup_target(${EQUATION})
  target_link_libraries(${EQUATION}
    obj_common obj_${EQUATION} ${OpenMP_CXX_LIBRARIES}
    )

  if(WITH_LIKWID)
    target_link_libraries(${EQUATION} likwid likwid-hwloc likwid-lua)
  endif()

  # For GNU libstdc++ we have to make sure to also link against libstdc++fs.
  if(LIBSTDCPP)
    target_link_libraries(${EQUATION} stdc++fs)
  endif()

  set_target_properties(${EQUATION} PROPERTIES
    RUNTIME_OUTPUT_NAME "ryujin-${DIM}d-${EQUATION}"
    )

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/run/${EQUATION}/ryujin"
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:${EQUATION}>
      "${CMAKE_BINARY_DIR}/run/${EQUATION}/ryujin"
      )

  add_custom_target(symlink_${EQUATION} ALL
    DEPENDS ${EQUATION} "${CMAKE_BINARY_DIR}/run/${EQUATION}/ryujin"
    COMMENT "updating ${EQUATION} symlink"
    )

  install(TARGETS ${EQUATION} DESTINATION ${CMAKE_INSTALL_BINDIR})
endmacro()

setup_equation("euler")
if(NOT "${DIM}" EQUAL "3")
  setup_equation("shallow_water")
endif()
