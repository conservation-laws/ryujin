if(NOT ${CMAKE_VERSION} VERSION_LESS 3.15)
  include_directories(
    ${CMAKE_SOURCE_DIR}/source/
    ${CMAKE_BINARY_DIR}/source/
    )

  add_library(testdriver SHARED $<TARGET_OBJECTS:ryujin>)
  deal_ii_setup_target(testdriver)
  if(LIKWID_PERFMON)
    target_link_libraries(testdriver likwid likwid-hwloc likwid-lua)
  endif()
  if(LIBSTDCPP)
    target_link_libraries(testdriver stdc++fs)
  endif()

  set(TEST_TARGET ryujin)
  set(TEST_LIBRARIES testdriver)
  deal_ii_pickup_tests()
endif()
