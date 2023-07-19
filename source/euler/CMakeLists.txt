##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2022 by the ryujin authors
##

add_library(obj_euler OBJECT
  limiter.cc
  riemann_solver.cc
  )
set_target_properties(obj_euler PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(obj_euler obj_common)
deal_ii_setup_target(obj_euler)
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_euler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
