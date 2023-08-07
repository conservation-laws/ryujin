##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2023 by the ryujin authors
##

add_library(obj_euler OBJECT
  initial_state_library.cc
  limiter.cc
  riemann_solver.cc
  )
set_target_properties(obj_euler PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_euler)
target_link_libraries(obj_euler obj_common ${EXTERNAL_TARGETS})
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_euler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
