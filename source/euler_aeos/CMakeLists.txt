##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2024 by the ryujin authors
##

add_library(obj_euler_aeos OBJECT
  equation_of_state_library.cc
  initial_state_library.cc
  limiter.cc
  riemann_solver.cc
  )
set_target_properties(obj_euler_aeos PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_euler_aeos)
target_link_libraries(obj_euler_aeos obj_common ${EXTERNAL_TARGETS})
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_euler_aeos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
