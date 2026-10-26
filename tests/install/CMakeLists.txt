##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# Install ryujin into the test directory, and build and run the example
# project in example/ against the installation:
#

add_test(NAME install/example
  COMMAND ${CMAKE_COMMAND}
    -DBINARY_DIR=${PROJECT_BINARY_DIR}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/example
    -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/example
    -DGENERATOR=${CMAKE_GENERATOR}
    -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/run_example.cmake
  )
