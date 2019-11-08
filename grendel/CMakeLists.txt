INCLUDE_DIRECTORIES(
  ${CMAKE_BINARY_DIR}/grendel/
  )

ADD_LIBRARY(grendel SHARED
  discretization.cc
  initial_values.cc
  limiter.cc
  matrix_communicator.cc
  offline_data.cc
  postprocessor.cc
  problem_description.cc
  riemann_solver.cc
  simd.cc
  time_step.cc
  )
DEAL_II_SETUP_TARGET(grendel)

IF(LIKWID_PERFMON)
  TARGET_LINK_LIBRARIES(grendel likwid likwid-hwloc likwid-lua)
ENDIF()

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )
