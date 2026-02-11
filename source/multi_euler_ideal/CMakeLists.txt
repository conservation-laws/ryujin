##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2023 - 2025 by the ryujin authors
## Copyright (C) 2025 by Triad National Security, LLC
##

add_library(obj_multi_euler_ideal OBJECT
  equation_dispatch.cc
  initial_state_library.cc
  limiter.cc
  riemann_solver.cc
  )
set_target_properties(obj_multi_euler_ideal PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_multi_euler_ideal)
target_link_libraries(obj_multi_euler_ideal obj_common ${EXTERNAL_TARGETS})
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_multi_euler_ideal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
