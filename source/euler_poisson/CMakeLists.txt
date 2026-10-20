##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

add_library(object_euler_poisson OBJECT
  electrostatic_configuration_library.cc
  equation.cc
  equation_euler_aeos.cc
  equation_euler_barotropic.cc
  parabolic_module.cc
  parabolic_module_euler_aeos.cc
  parabolic_module_euler_barotropic.cc
  )
set_target_properties(object_euler_poisson PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(object_euler_poisson)
target_link_libraries(object_euler_poisson object_core ${EXTERNAL_TARGETS})
