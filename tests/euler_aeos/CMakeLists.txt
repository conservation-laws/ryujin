##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 - 2024 by the ryujin authors
##

set(EQUATION euler_aeos)

set(TEST_LIBRARIES obj_common obj_${EQUATION})
set(TEST_TARGET ryujin)

if(TARGET obj_${EQUATION})
  deal_ii_pickup_tests()
endif()
