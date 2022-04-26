##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2021 by the ryujin authors
##

add_library(navier_stokes_equations OBJECT
  dissipation_module.cc
  )
deal_ii_setup_target(navier_stokes_equations)
