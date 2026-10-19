##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# Set up a target of a project using ryujin: compile and link with the
# deal.II configuration of the build type that ryujin was built with, and
# link against ryujin.
#
#   ryujin_setup_target(<target>)
#
# The macro works for ryujin found with find_package(ryujin), as well as
# for ryujin included with add_subdirectory(). The latter requires a call
# to find_package(deal.II) in the project itself.
#

macro(ryujin_setup_target _target)
  if(NOT TARGET ryujin::ryujin)
    ryujin_include_target_config()
  endif()
  get_property(_ryujin_build GLOBAL PROPERTY RYUJIN_BUILD_TYPE)
  string(TOUPPER "${_ryujin_build}" _ryujin_build)
  deal_ii_setup_target(${_target} ${_ryujin_build})
  target_link_libraries(${_target} ryujin::ryujin)
endmacro()
