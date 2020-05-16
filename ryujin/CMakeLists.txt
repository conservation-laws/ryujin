##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 by the ryujin authors
##

include_directories(
  ${CMAKE_SOURCE_DIR}/ryujin/
  ${CMAKE_SOURCE_DIR}/grendel/
  ${CMAKE_BINARY_DIR}/grendel/
  )

add_executable(ryujin
  ryujin.cc
  timeloop.cc
  )

deal_ii_setup_target(ryujin)
target_link_librarieS(ryujin grendel)
target_link_librarieS(ryujin stdc++fs)

set_property(TARGET ryujin
  PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/run
  )

set_property(TARGET ryujin APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )
