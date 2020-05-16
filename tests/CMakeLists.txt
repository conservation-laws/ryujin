include_directories(
  ${CMAKE_SOURCE_DIR}/ryujin/
  ${CMAKE_SOURCE_DIR}/grendel/
  ${CMAKE_BINARY_DIR}/grendel/
  )

set(test_libraries grendel)

deal_ii_pickup_tests()
