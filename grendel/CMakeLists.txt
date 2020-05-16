include_directories(
  ${CMAKE_BINARY_DIR}/grendel/
  )

add_library(grendel SHARED
  discretization.cc
  initial_values.cc
  limiter.cc
  offline_data.cc
  postprocessor.cc
  problem_description.cc
  riemann_solver.cc
  simd.cc
  sparse_matrix_simd.cc
  time_step.cc
  )
deal_ii_setup_target(grendel)

if(LIKWID_PERFMON)
  target_link_libraries(grendel likwid likwid-hwloc likwid-lua)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )
