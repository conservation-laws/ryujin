##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

if(NOT EQUATION_EULER_AEOS)
  message(FATAL_ERROR
    "The equation »euler_poisson_aeos« requires the equation »euler_aeos« to be enabled")
endif()

if(NOT EQUATION_EULER_POISSON)
  message(FATAL_ERROR
    "The equation »euler_poisson_aeos« requires the equation »euler_poisson« to be enabled")
endif()

add_library(obj_euler_poisson_aeos OBJECT
  equation_dispatch.cc
  parabolic_module.cc
  )
set_target_properties(obj_euler_poisson_aeos PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_euler_poisson_aeos)
target_link_libraries(obj_euler_poisson_aeos obj_common ${EXTERNAL_TARGETS})
