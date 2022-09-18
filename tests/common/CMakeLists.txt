include_directories(
  ${CMAKE_BINARY_DIR}/source/
  )

set(TEST_LIBRARIES obj_common)
deal_ii_pickup_tests()
