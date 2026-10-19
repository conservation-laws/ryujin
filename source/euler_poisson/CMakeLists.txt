##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

if(NOT EQUATION_EULER)
  message(FATAL_ERROR
    "The equation »euler_poisson« requires the equation »euler« to be enabled")
endif()

add_library(obj_euler_poisson OBJECT
  electrostatic_configuration_library.cc
  equation.cc
  parabolic_module.cc
  )
set_target_properties(obj_euler_poisson PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_euler_poisson)
target_link_libraries(obj_euler_poisson obj_common ${EXTERNAL_TARGETS})
