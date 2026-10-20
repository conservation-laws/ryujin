##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2025 - 2026 by the ryujin authors
##

add_library(object_euler_barotropic OBJECT
  barotropic_equation_of_state_library.cc
  hyperbolic_module.cc
  initial_state_library.cc
  initial_values.cc
  solution_transfer.cc
  )
set_target_properties(object_euler_barotropic PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(object_euler_barotropic)
target_link_libraries(object_euler_barotropic object_core ${EXTERNAL_TARGETS})
