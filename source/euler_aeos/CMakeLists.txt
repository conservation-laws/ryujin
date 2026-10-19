##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2023 - 2026 by the ryujin authors
##

add_library(object_euler_aeos OBJECT
  equation.cc
  equation_of_state_library.cc
  hyperbolic_module.cc
  initial_state_library.cc
  initial_values.cc
  solution_transfer.cc
  )
set_target_properties(object_euler_aeos PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(object_euler_aeos)
target_link_libraries(object_euler_aeos object_core ${EXTERNAL_TARGETS})
