##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2024 by the ryujin authors
##

find_path(LIKWID_INCLUDE_DIR likwid.h
  PATH_SUFFIXES include
  )

find_library(LIKWID_LIBRARY
  NAMES likwid
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

find_library(LIKWID_HWLOC_LIBRARY
  NAMES likwid-hwloc
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

find_library(LIKWID_LUA_LIBRARY
  NAMES likwid-lua
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

find_package_handle_standard_args(LIKWID DEFAULT_MSG
  LIKWID_LIBRARY LIKWID_HWLOC_LIBRARY LIKWID_LUA_LIBRARY LIKWID_INCLUDE_DIR
  )

if(LIKWID_FOUND AND NOT TARGET Likdwid::Likwid)
  add_library(Likwid::Likwid INTERFACE IMPORTED)
  target_link_libraries(Likwid::Likwid INTERFACE
    ${LIKWID_LIBRARY} ${LIKWID_HWLOC_LIBRARY} ${LIKWID_LUA_LIBRARY}
    )
  target_include_directories(Likwid::Likwid SYSTEM INTERFACE ${LIKWID_INCLUDE_DIR})
  target_compile_definitions(Likwid::Likwid INTERFACE "LIKWID_PERFMON")
endif()
