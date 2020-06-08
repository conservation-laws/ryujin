##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 by the ryujin authors
##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  )

add_executable(ryujin
  discretization.cc
  initial_values.cc
  limiter.cc
  main.cc
  offline_data.cc
  postprocessor.cc
  problem_description.cc
  riemann_solver.cc
  simd.cc
  sparse_matrix_simd.cc
  time_loop.cc
  time_step.cc
  )

deal_ii_setup_target(ryujin)

set_property(SOURCE time_loop.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

set_property(TARGET ryujin
  PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run
  )

if(LIKWID_PERFMON)
  target_link_libraries(ryujin likwid likwid-hwloc likwid-lua)
endif()

# FIXME: This is wrong if we link against libc++.
target_link_libraries(ryujin stdc++fs)
