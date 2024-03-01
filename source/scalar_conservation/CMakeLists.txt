##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2024 by the ryujin authors
##

add_library(obj_scalar_conservation OBJECT
  flux_library.cc
  initial_state_library.cc
  limiter.cc
  riemann_solver.cc
  )
set_target_properties(obj_scalar_conservation PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(obj_scalar_conservation obj_common)
deal_ii_setup_target(obj_scalar_conservation)
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_scalar_conservation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
