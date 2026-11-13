##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

add_library(object_scalar_diffusion OBJECT
  parabolic_module.cc
  )
set_target_properties(object_scalar_diffusion PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(object_scalar_diffusion)
target_link_libraries(object_scalar_diffusion object_core ${EXTERNAL_TARGETS})
