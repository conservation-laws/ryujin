ADD_LIBRARY(grendel SHARED
  discretization.cc
  offline_data.cc
  problem_description.cc
  riemann_solver.cc
  time_step.cc
  )
DEAL_II_SETUP_TARGET(grendel)
