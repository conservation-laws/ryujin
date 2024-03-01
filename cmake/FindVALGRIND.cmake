##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2024 by the ryujin authors
##

# FIXME: we are currently not linking in the runtime bits. I doesn't seem
#        to be necessary for callgrind

find_path(VALGRIND_INCLUDE_DIR valgrind/callgrind.h
  PATH_SUFFIXES include
  )

find_package_handle_standard_args(VALGRIND DEFAULT_MSG
  VALGRIND_INCLUDE_DIR
  )

if(VALGRIND_FOUND AND NOT TARGET Valgrind::Valgrind)
  add_library(Valgrind::Valgrind INTERFACE IMPORTED)
  target_include_directories(Valgrind::Valgrind SYSTEM INTERFACE ${VALGRIND_INCLUDE_DIR})
endif()
