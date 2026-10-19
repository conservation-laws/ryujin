##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2023 - 2026 by the ryujin authors
##

add_library(object_scalar_conservation OBJECT
  equation.cc
  flux_library.cc
  hyperbolic_module.cc
  initial_state_library.cc
  initial_values.cc
  solution_transfer.cc
  )
set_target_properties(object_scalar_conservation PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(object_scalar_conservation object_core)
deal_ii_setup_target(object_scalar_conservation)
