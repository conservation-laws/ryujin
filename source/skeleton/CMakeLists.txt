##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2024 by the ryujin authors
##

add_library(obj_skeleton OBJECT
  initial_state_library.cc
  )
set_target_properties(obj_skeleton PROPERTIES LINKER_LANGUAGE CXX)
deal_ii_setup_target(obj_skeleton)
target_link_libraries(obj_skeleton obj_common ${EXTERNAL_TARGETS})
# Propagate the current source directory with PUBLIC visibility
target_include_directories(obj_skeleton PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
