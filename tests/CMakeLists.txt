if(NOT ${CMAKE_VERSION} VERSION_LESS 3.15)
  include_directories(
    ${CMAKE_SOURCE_DIR}/source/
    ${CMAKE_BINARY_DIR}/source/
    )

  add_library(tests SHARED $<TARGET_OBJECTS:ryujin>)
  deal_ii_setup_target(tests)
  if(LIKWID_PERFMON)
    target_link_libraries(tests likwid likwid-hwloc likwid-lua)
  endif()

  set(TEST_LIBRARIES tests)
  deal_ii_pickup_tests()
endif()
