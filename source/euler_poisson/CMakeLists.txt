##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

if(NOT EQUATION_EULER)
  message(FATAL_ERROR
    "The equation »euler_poisson« requires the equation »euler« to be enabled")
endif()

add_library(object_euler_poisson OBJECT
  electrostatic_configuration_library.cc
  equation.cc
  parabolic_module.cc
  )
set_target_properties(object_euler_poisson PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(object_euler_poisson)
target_link_libraries(object_euler_poisson object_core ${EXTERNAL_TARGETS})

#
# The Euler-Poisson equations for the hyperbolic descriptions of euler_aeos
# and euler_barotropic, if enabled:
#

foreach(_variant aeos barotropic)
  string(TOUPPER "${_variant}" _variant_upper)
  if(EQUATION_EULER_${_variant_upper})
    add_library(object_euler_poisson_${_variant} OBJECT
      equation_euler_${_variant}.cc
      parabolic_module_euler_${_variant}.cc
      )
    set_target_properties(object_euler_poisson_${_variant} PROPERTIES LINKER_LANGUAGE CXX)
    deal_ii_setup_target(object_euler_poisson_${_variant})
    target_link_libraries(object_euler_poisson_${_variant} object_core ${EXTERNAL_TARGETS})
    list(APPEND OBJECT_TARGETS object_euler_poisson_${_variant})
  endif()
endforeach()
set(OBJECT_TARGETS ${OBJECT_TARGETS} PARENT_SCOPE)
