##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2023 - 2026 by the ryujin authors
##

add_library(obj_scalar_conservation OBJECT
  equation.cc
  flux_library.cc
  hyperbolic_module.cc
  initial_state_library.cc
  initial_values.cc
  solution_transfer.cc
  )
set_target_properties(obj_scalar_conservation PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(obj_scalar_conservation obj_common)
deal_ii_setup_target(obj_scalar_conservation)
