##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 by the ryujin authors
##

set(RYUJIN_VERSION 1.6.0)

cmake_minimum_required(VERSION 3.5)

find_package(deal.II 9.2 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

deal_ii_initialize_cached_variables()
deal_ii_query_git_information()

if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_THREADS OR NOT DEAL_II_WITH_P4EST)
  message(FATAL_ERROR
    "Need a deal.II library with threads, mpi and p4est support enabled."
    )
endif()

project(ryujin CXX)

#
# Set up compiler flags:
#

string(APPEND DEAL_II_CXX_FLAGS " -fopenmp")
string(APPEND DEAL_II_LINKER_FLAGS " -fopenmp -Wl,-as-needed")

string(APPEND DEAL_II_CXX_FLAGS_RELEASE " -O3 -ffp-contract=fast")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(APPEND DEAL_II_CXX_FLAGS " -fdiagnostics-color=always")
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
endif()

#
# External library and feature configuration:
#

option(CALLGRIND
  "Compile and link against the valgrind/callgrind instrumentation library" OFF
  )

option(LIKWID_PERFMON
  "Compile and link against the likwid instrumentation library" OFF
  )

option(DOCUMENTATION
  "Build the documentation with doxygen" OFF
  )

#
# Options:
#

set(DIM "2" CACHE STRING "The dimension")

set(NUMBER "double" CACHE STRING "The principal floating point type")

option(USE_COMMUNICATION_HIDING "Issue MPI synchronization of ghost values early" ON)

option(USE_CUSTOM_POW "Use custom pow implementation" ON)

option(USE_SIMD "Use SIMD vectorization" ON)

option(CHECK_BOUNDS "Enable debug code paths that check limiter bounds" OFF)

option(DEBUG_OUTPUT "Enable detailed time-step output" OFF)

option(OBSESSIVE_INLINING "Also inline the Riemann solver and limiter calls" OFF)

#
# Set up the rest:
#

enable_testing()

add_subdirectory(source)

add_subdirectory(tests)

IF(DOCUMENTATION)
  add_subdirectory(doc)
ENDIF()
