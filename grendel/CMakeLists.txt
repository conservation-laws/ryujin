ADD_LIBRARY(grendel SHARED
  discretization.cc
  riemann_solver.cc
  offline_data.cc
  time_step.cc
  )
DEAL_II_SETUP_TARGET(grendel)
