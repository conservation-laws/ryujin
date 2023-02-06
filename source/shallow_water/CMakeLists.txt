##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2022 by the ryujin authors
##

add_library(obj_shallow_water OBJECT
  limiter.cc
  hyperbolic_system.cc
  riemann_solver.cc
  )
target_link_libraries(obj_shallow_water PUBLIC obj_common)
deal_ii_setup_target(obj_shallow_water)
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_shallow_water PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
