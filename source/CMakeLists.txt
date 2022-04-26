##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2021 by the ryujin authors
##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )

include_directories(
  ${CMAKE_BINARY_DIR}/source/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# Detect whether we are building against GCC's libstdc++.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  #include <iostream>
  #ifndef __GLIBCXX__
  #error NOT LIBSTDCPP
  #endif
  int main() {}"
  LIBSTDCPP
  )

#
# Common and dependent source files:
#

set(COMMON_SOURCE_FILES
  discretization.cc
  simd.cc
  sparse_matrix_simd.cc
  )

set(DEPENDENT_SOURCE_FILES
  euler_module.cc
  initial_values.cc
  main.cc
  offline_data.cc
  quantities.cc
  time_integrator.cc
  time_loop.cc
  vtu_output.cc
  )

set_property(SOURCE time_loop.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

add_library(obj_common ${COMMON_SOURCE_FILES})
deal_ii_setup_target(obj_common)

#
# Equation-specific source files:
#

if("${INSTANTIATED_EQUATIONS}" STREQUAL "")
  set(INSTANTIATED_EQUATIONS
    "euler"
    )
endif()

foreach(EQUATION ${INSTANTIATED_EQUATIONS})
  # define obj_${EQUATION} object library:
  add_subdirectory(${EQUATION})

  add_library(obj_dependent_${EQUATION}
    ${DEPENDENT_SOURCE_FILES}
    )
  target_include_directories(obj_dependent_${EQUATION} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/${EQUATION}
    )
  deal_ii_setup_target(obj_dependent_${EQUATION})

  add_executable(${EQUATION}
    $<TARGET_OBJECTS:obj_common>
    $<TARGET_OBJECTS:obj_dependent_${EQUATION}>
    $<TARGET_OBJECTS:obj_${EQUATION}>
    )
  deal_ii_setup_target(${EQUATION})
  target_link_libraries(${EQUATION} ${OpenMP_CXX_LIBRARIES})

  set_property(TARGET ${EQUATION}
    PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run
    )

  if(LIKWID_PERFMON)
    target_link_libraries(${EQUATION} likwid likwid-hwloc likwid-lua)
  endif()

  # For GNU libstdc++ we have to make sure to also link against libstdc++fs.
  if(LIBSTDCPP)
    target_link_libraries(${EQUATION} stdc++fs)
  endif()

  install(TARGETS ${EQUATION}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endforeach()
