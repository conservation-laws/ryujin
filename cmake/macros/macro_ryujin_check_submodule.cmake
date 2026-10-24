##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# Abort with an error message if the git submodule <name> located in the
# current source directory has not been checked out, i.e., its directory
# is missing or empty. If the submodule is only needed for an optional
# feature, the name of the controlling option can be passed as second
# argument and is mentioned in the error message.
#
#   ryujin_check_submodule(<name> [<option>])
#

function(ryujin_check_submodule _name)
  file(GLOB _files ${CMAKE_CURRENT_SOURCE_DIR}/${_name}/*)
  if("${_files}" STREQUAL "")
    set(_message
      "The ${_name} git submodule is missing. Run\n"
      "  $ git submodule update --init ${_name}\n"
      )
    if(NOT "${ARGV1}" STREQUAL "")
      list(APPEND _message "or configure with -D${ARGV1}=OFF\n")
    endif()
    message(FATAL_ERROR ${_message})
  endif()
endfunction()
