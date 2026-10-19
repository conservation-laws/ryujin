##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 - 2024 by the ryujin authors
##

set(EQUATION navier_stokes)

set(TEST_LIBRARIES
  obj_common obj_${EQUATION}
  obj_euler
  )
set(TEST_TARGET ryujin)

if(TARGET obj_${EQUATION})
  deal_ii_pickup_tests()
endif()
