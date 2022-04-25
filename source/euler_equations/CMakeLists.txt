##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2021 by the ryujin authors
##

add_library(euler_equations OBJECT
  limiter.cc
  postprocessor.cc
  hyperbolic_system.cc
  riemann_solver.cc
  )
deal_ii_setup_target(euler_equations)
