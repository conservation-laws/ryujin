##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2022 by the ryujin authors
##

add_library(obj_shallow_water OBJECT
  limiter.cc
  postprocessor.cc
  hyperbolic_system.cc
  riemann_solver.cc
  )
deal_ii_setup_target(obj_shallow_water)
target_include_directories(obj_shallow_water PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
