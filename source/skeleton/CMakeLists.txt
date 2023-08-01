##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2023 by the ryujin authors
##

add_library(obj_skeleton OBJECT
  initial_state_library.cc
  )
set_target_properties(obj_skeleton PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(obj_skeleton obj_common ${EXTERNAL_TARGETS})
deal_ii_setup_target(obj_skeleton)
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_skeleton PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
