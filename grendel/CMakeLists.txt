ADD_LIBRARY(grendel SHARED
  discretization.cc
  initial_values.cc
  matrix_communicator.cc
  offline_data.cc
  problem_description.cc
  riemann_solver.cc
  schlieren_postprocessor.cc
  time_step.cc
  )
DEAL_II_SETUP_TARGET(grendel)

IF(USE_SIMD)
  SET_PROPERTY(TARGET grendel APPEND PROPERTY COMPILE_DEFINITIONS USE_SIMD)
ENDIF()
