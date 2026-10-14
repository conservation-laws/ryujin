##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

if(NOT EQUATION_EULER_BAROTROPIC)
  message(FATAL_ERROR
    "The equation »euler_poisson_barotropic« requires the equation »euler_barotropic« to be enabled")
endif()

if(NOT EQUATION_EULER_POISSON)
  message(FATAL_ERROR
    "The equation »euler_poisson_barotropic« requires the equation »euler_poisson« to be enabled")
endif()

add_library(obj_euler_poisson_barotropic OBJECT
  equation_dispatch.cc
  parabolic_module.cc
  )
set_target_properties(obj_euler_poisson_barotropic PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_euler_poisson_barotropic)
target_link_libraries(obj_euler_poisson_barotropic obj_common ${EXTERNAL_TARGETS})
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_euler_poisson_barotropic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
