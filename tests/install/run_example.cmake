##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# Usage:
#   cmake -DBINARY_DIR=<ryujin build directory> -DSOURCE_DIR=<example>
#     -DTEST_DIR=<scratch directory> -DGENERATOR=<generator>
#     -DCXX_COMPILER=<compiler of ryujin>
#     -P run_example.cmake
#

macro(run_step _name)
  message(STATUS "${_name}")
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE _result)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${_name} failed: ${_result}")
  endif()
endmacro()

file(REMOVE_RECURSE ${TEST_DIR})
file(MAKE_DIRECTORY ${TEST_DIR}/run)

run_step("Build ryujin"
  ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ryujin
  )
run_step("Install ryujin"
  ${CMAKE_COMMAND} --install ${BINARY_DIR} --prefix ${TEST_DIR}/install
  )
run_step("Configure example"
  ${CMAKE_COMMAND} -G ${GENERATOR} -S ${SOURCE_DIR} -B ${TEST_DIR}/build
  -DRYUJIN_DIR=${TEST_DIR}/install -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
  )
run_step("Build example"
  ${CMAKE_COMMAND} --build ${TEST_DIR}/build
  )

file(COPY ${SOURCE_DIR}/euler.prm DESTINATION ${TEST_DIR}/run)
execute_process(COMMAND ${TEST_DIR}/build/example
  WORKING_DIRECTORY ${TEST_DIR}/run
  RESULT_VARIABLE _result
  )
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Run example failed: ${_result}")
endif()
