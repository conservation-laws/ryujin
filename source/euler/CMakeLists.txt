##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2021 by the ryujin authors
##

add_library(obj_euler OBJECT
  limiter.cc
  postprocessor.cc
  hyperbolic_system.cc
  riemann_solver.cc
  )
## FIXME:
target_include_directories(obj_euler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
deal_ii_setup_target(obj_euler)
