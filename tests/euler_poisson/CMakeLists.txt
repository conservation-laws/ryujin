##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 - 2026 by the ryujin authors
##

set(EQUATION euler_poisson)

set(TEST_LIBRARIES libryujin)
set(TEST_TARGET ryujin)

# Tests of the Euler-Poisson equations for the barotropic Euler equations
# are marked with ".with_equation_euler_barotropic=on":
set(DEAL_II_WITH_EQUATION_EULER_BAROTROPIC ${EQUATION_EULER_BAROTROPIC})

if(TARGET object_${EQUATION})
  deal_ii_pickup_tests()
endif()
