##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 - 2026 by the ryujin authors
##

if(NOT EQUATION_EULER)
  message(FATAL_ERROR
    "The equation »navier_stokes« requires the equation »euler« to be enabled")
endif()

add_library(obj_navier_stokes OBJECT
  equation.cc
  parabolic_module.cc
  )
set_target_properties(obj_navier_stokes PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_navier_stokes)
target_link_libraries(obj_navier_stokes obj_common ${EXTERNAL_TARGETS})
