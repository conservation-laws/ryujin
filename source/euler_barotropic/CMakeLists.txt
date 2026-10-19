##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2025 - 2026 by the ryujin authors
##

add_library(obj_euler_barotropic OBJECT
  barotropic_equation_of_state_library.cc
  equation_dispatch.cc
  hyperbolic_module.cc
  initial_state_library.cc
  initial_values.cc
  solution_transfer.cc
  )
set_target_properties(obj_euler_barotropic PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_euler_barotropic)
target_link_libraries(obj_euler_barotropic obj_common ${EXTERNAL_TARGETS})
