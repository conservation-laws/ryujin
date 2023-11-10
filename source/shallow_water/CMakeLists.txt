##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2023 by the ryujin authors
##

add_library(obj_shallow_water OBJECT
  initial_state_library.cc
  )
  #limiter.cc
  #riemann_solver.cc
set_target_properties(obj_shallow_water PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_shallow_water)
target_link_libraries(obj_shallow_water obj_common ${EXTERNAL_TARGETS})
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_shallow_water PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
