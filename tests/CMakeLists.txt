include_directories(
  ${CMAKE_SOURCE_DIR}/ryujin/
  ${CMAKE_SOURCE_DIR}/ryujin/
  ${CMAKE_BINARY_DIR}/ryujin/
  )

set(test_libraries ryujin)

deal_ii_pickup_tests()
