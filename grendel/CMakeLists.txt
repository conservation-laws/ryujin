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

IF(USE_SIMD)
  SET_PROPERTY(SOURCE offline_data.cc APPEND PROPERTY COMPILE_DEFINITIONS USE_SIMD)
ENDIF()

IF(USE_CUSTOM_POW)
  SET_PROPERTY(TARGET grendel APPEND PROPERTY COMPILE_DEFINITIONS USE_CUSTOM_POW)
ENDIF()

SET_PROPERTY(TARGET grendel APPEND PROPERTY COMPILE_DEFINITIONS
  DIM=${DIM}
  NUMBER=${NUMBER}
  )
