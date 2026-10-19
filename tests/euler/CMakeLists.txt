##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 - 2026 by the ryujin authors
##

set(EQUATION euler)

set(TEST_LIBRARIES libryujin)
set(TEST_TARGET ryujin)

if(TARGET object_${EQUATION})
  deal_ii_pickup_tests()
endif()
