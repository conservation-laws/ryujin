##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

set(TEST_LIBRARIES libryujin)

# The tests of the driver run the Euler equations:
if(TARGET object_euler)
  deal_ii_pickup_tests()
endif()
