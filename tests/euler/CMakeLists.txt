set(EQUATION euler)

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  ${CMAKE_SOURCE_DIR}/source/${EQUATION}
  ${CMAKE_SOURCE_DIR}/source/
  )

set(TEST_LIBRARIES obj_common obj_${EQUATION} obj_${EQUATION}_dependent)

set(TEST_TARGET ryujin)

deal_ii_pickup_tests()
