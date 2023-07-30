##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2023 by the ryujin authors
##

add_library(obj_navier_stokes OBJECT
  initial_state_library.cc
  parabolic_solver.cc
  )
set_target_properties(obj_navier_stokes PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(obj_navier_stokes obj_common)
deal_ii_setup_target(obj_navier_stokes)
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_navier_stokes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
