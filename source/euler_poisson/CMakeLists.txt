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

#
# The Euler-Poisson equations for the hyperbolic descriptions of euler_aeos
# and euler_barotropic, if enabled:
#

foreach(_variant aeos barotropic)
  string(TOUPPER "${_variant}" _variant_upper)
  if(EQUATION_EULER_${_variant_upper})
    add_library(obj_euler_poisson_${_variant} OBJECT
      equation_euler_${_variant}.cc
      parabolic_module_euler_${_variant}.cc
      )
    set_target_properties(obj_euler_poisson_${_variant} PROPERTIES LINKER_LANGUAGE CXX)
    deal_ii_setup_target(obj_euler_poisson_${_variant})
    target_link_libraries(obj_euler_poisson_${_variant} obj_common ${EXTERNAL_TARGETS})
    list(APPEND OBJECT_TARGETS obj_euler_poisson_${_variant})
  endif()
endforeach()
set(OBJECT_TARGETS ${OBJECT_TARGETS} PARENT_SCOPE)
