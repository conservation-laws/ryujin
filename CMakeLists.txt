##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 - 2022 by the ryujin authors
##

get_filename_component(_source "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_build  "${CMAKE_BINARY_DIR}" REALPATH)
if("${_source}" STREQUAL "${_build}")
  message(FATAL_ERROR
    "\nRefusing to configure the project in the source directory. This "
    "operation would globber important files. You need to configure in a "
    "separate build directory. It is easiest to simply invoke $ make, which "
    "will configure ryujin in the directory ./build.\n"
    "You will need to clean up ./CMakeCache.txt and ./CMakeFiles by hand."
    )
endif()

set(RYUJIN_VERSION 2.1.1)

cmake_minimum_required(VERSION 3.11)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(deal.II 9.3 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffp-contract=fast" CACHE STRING
  "Flags used by the compiler during release builds."
  )

deal_ii_initialize_cached_variables()
deal_ii_query_git_information()

if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR
    "Need a deal.II library with mpi and p4est support enabled."
    )
endif()

project(ryujin CXX)

#
# Options:
#

set(DIM "2" CACHE STRING "The dimension")
set(NUMBER "double" CACHE STRING "The principal floating point type")

option(CHECK_BOUNDS "Enable debug code paths that check limiter bounds" OFF)
option(DEBUG_OUTPUT "Enable detailed time-step output" OFF)
option(DENORMALS_ARE_ZERO "Set the \"denormals are zero\" and \"flush to zero\" bits in the MXCSR register" ON)
option(FORCE_DEAL_II_SPARSE_MATRIX "Always use dealii::SparseMatrix instead of TrilinosWrappers::SparseMatrix for assembly" OFF)

option(WITH_ASYNC_MPI_EXCHANGE "Use synchronous MPI communication" OFF)
option(WITH_CALLGRIND "Compile and link against the valgrind/callgrind instrumentation library" OFF)
option(WITH_CUSTOM_POW "Use custom serial pow implementation" ON)
option(WITH_LIKWID "Compile and link against the likwid instrumentation library" OFF)
option(WITH_DOXYGEN "Build documentation with doxygen" OFF)

find_package(OpenMP)
option(WITH_OPENMP "Enable threading support via OpenMP" ${OpenMP_FOUND})

set(ORDER_FINITE_ELEMENT "1" CACHE STRING "Order of finite elements")
set(ORDER_MAPPING "1" CACHE STRING "Order of mapping")
set(ORDER_QUADRATURE "2" CACHE STRING "Order of quadrature")

#
# Set up compiler flags:
#

#string(APPEND DEAL_II_CXX_FLAGS " -Wfatal-errors")

string(APPEND DEAL_II_CXX_FLAGS " -std=c++17")

if(WITH_OPENMP)
  find_package(OpenMP REQUIRED)
  string(APPEND DEAL_II_CXX_FLAGS " ${OpenMP_CXX_FLAGS}")
  string(APPEND DEAL_II_LINKER_FLAGS " ${OpenMP_CXX_FLAGS}")
  #
  # Work around a bug with brew on darwin where libomp.dylib gets silently
  # dropped from the link line even though we try to add the library with a
  # proper target_link_libraries() call...
  #
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    string(APPEND DEAL_II_LINKER_FLAGS " ${OpenMP_CXX_LIBRARIES}")
  endif()
endif()

if(WITH_LIKWID)
  add_definitions(-DLIKWID_PERFMON)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(APPEND DEAL_II_CXX_FLAGS " -fdiagnostics-color=always")
  string(REPLACE "-Qunused-arguments" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")
  string(APPEND DEAL_II_CXX_FLAGS " -Wno-missing-braces -Wno-unknown-pragmas")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND DEAL_II_CXX_FLAGS
    " -Xclang -fcolor-diagnostics -Qunused-arguments"
    )
  string(APPEND DEAL_II_CXX_FLAGS_DEBUG
    " -fsanitize=address -fsanitize-address-use-after-scope"
    )
  string(APPEND DEAL_II_LINKER_FLAGS_DEBUG
    " -fsanitize=address -fsanitize-address-use-after-scope"
    )
  string(APPEND DEAL_II_CXX_FLAGS " -Wno-undefined-var-template")
endif()

#
# Detection
#

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
# Respect user overrides performed via CMAKE_CXX_FLAGS/CMAKE_EXE_LINKER_FLAGS
#

string(APPEND DEAL_II_CXX_FLAGS " $ENV{CXXFLAGS} ${CMAKE_CXX_FLAGS}")
string(APPEND DEAL_II_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG}")
string(APPEND DEAL_II_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS_RELEASE}")
string(APPEND DEAL_II_LINKER_FLAGS " $ENV{LDFLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_EXE_LINKER_FLAGS "")

#
# Set up the rest:
#

include(GNUInstallDirs)

add_subdirectory(source)
add_subdirectory(prm)

enable_testing()
add_subdirectory(tests)

IF(WITH_DOXYGEN)
  add_subdirectory(doc)
ENDIF()
