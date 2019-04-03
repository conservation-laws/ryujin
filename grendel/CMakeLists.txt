ADD_LIBRARY(grendel SHARED
  discretization.cc
  initial_values.cc
  offline_data.cc
  problem_description.cc
  riemann_solver.cc
  schlieren_postprocessor.cc
  time_step.cc
  )
DEAL_II_SETUP_TARGET(grendel)
