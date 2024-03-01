##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2022 by the ryujin authors
##

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  )

set(TEST_LIBRARIES obj_common)
deal_ii_pickup_tests()
