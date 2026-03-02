##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2025 by the ryujin authors
## Copyright (C) 2025 by Triad National Security, LLC
##

set(EQUATION multi_euler_ideal)

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  ${CMAKE_SOURCE_DIR}/source/${EQUATION}
  ${CMAKE_SOURCE_DIR}/source/
  )

set(TEST_LIBRARIES obj_common obj_${EQUATION} obj_${EQUATION}_dependent)
set(TEST_TARGET ryujin)

if(TARGET obj_${EQUATION})
  deal_ii_pickup_tests()
endif()
