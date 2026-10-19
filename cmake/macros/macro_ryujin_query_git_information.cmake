##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# A wrapper around deal_ii_query_git_information() that ensures we are
# querying the repository located at the current source directory for git
# information.
#
#   ryujin_query_git_information()
#

function(ryujin_query_git_information)
  set(CMAKE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  deal_ii_query_git_information()
  foreach(_var GIT_BRANCH GIT_REVISION GIT_SHORTREV GIT_TIMESTAMP GIT_TAG)
    set(${_var} "${${_var}}" PARENT_SCOPE)
  endforeach()
endfunction()
