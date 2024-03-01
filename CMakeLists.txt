##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2024 by the ryujin authors
##

#
# Sanity check:
#

get_filename_component(_source "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_build  "${CMAKE_BINARY_DIR}" REALPATH)
if("${_source}" STREQUAL "${_build}")
  message(FATAL_ERROR
    "\nRefusing to configure the project in the source directory. This "
    "operation would globber important files. You need to configure in a "
    "separate build directory. It is easiest to simply invoke $ make, which "
    "will configure ryujin in the directory ./build.\n"
    "You will need to clean up ./CMakeCache.txt and ./CMakeFiles by hand, "
    "or by running $ make cleanup_insource which will delete these files "
    "and directories for you"
    )
endif()

#
# Set up deal.II
#

set(RYUJIN_VERSION 2.1.1)

cmake_minimum_required(VERSION 3.15)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(deal.II 9.3 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})

set(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Choose the type of build, options are: Debug, Release"
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
# Compile-time options:
#

set(NUMBER "double" CACHE STRING "The principal floating point type")

option(ASYNC_MPI_EXCHANGE "Use synchronous MPI communication" OFF)
option(CHECK_BOUNDS "Enable debug code paths that check limiter bounds" OFF)
option(DEBUG_OUTPUT "Enable detailed time-step output" OFF)
option(DENORMALS_ARE_ZERO "Set the \"denormals are zero\" and \"flush to zero\" bits in the MXCSR register" ON)
option(FORCE_DEAL_II_SPARSE_MATRIX "Always use dealii::SparseMatrix instead of TrilinosWrappers::SparseMatrix for assembly" OFF)
option(SANITIZER "Enable address and UBSAN sanitizers for DEBUG build" OFF)

set(ORDER_FINITE_ELEMENT "1" CACHE STRING "Order of finite elements")
set(ORDER_MAPPING "1" CACHE STRING "Order of mapping")
set(ORDER_QUADRATURE "2" CACHE STRING "Order of quadrature")

#
# External packages:
#

option(WITH_DOXYGEN "Build documentation with doxygen" OFF)
option(WITH_LIKWID "Compile and link against the likwid instrumentation library" OFF)
option(WITH_VALGRIND "Compile and link against the valgrind/callgrind instrumentation library" OFF)

if("${WITH_OPENMP}" STREQUAL "")
  find_package(OpenMP QUIET)
endif()
option(WITH_OPENMP "Enable threading support via OpenMP" ${OpenMP_FOUND})

if("${WITH_EOSPAC}" STREQUAL "")
  find_package(EOSPAC QUIET)
endif()
option(WITH_EOSPAC "Add support for the EOSPAC suite" ${EOSPAC_FOUND})

#
# Set up compiler flags:
#

#string(APPEND DEAL_II_CXX_FLAGS " -Wfatal-errors")
string(APPEND DEAL_II_CXX_FLAGS " -std=c++17")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(APPEND DEAL_II_CXX_FLAGS " -fdiagnostics-color=always")
  string(REPLACE "-Qunused-arguments" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")
  string(APPEND DEAL_II_CXX_FLAGS " -Wno-missing-braces -Wno-unknown-pragmas")

  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0)
    string(APPEND DEAL_II_CXX_FLAGS " -Wno-overloaded-virtual")
  endif()


endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND DEAL_II_CXX_FLAGS
    " -Xclang -fcolor-diagnostics"
    )
endif()

if(SANITIZER)
  string(APPEND DEAL_II_CXX_FLAGS_DEBUG
    " -fsanitize=address,undefined,leak -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
    )
  string(APPEND DEAL_II_LINKER_FLAGS_DEBUG
    " -fsanitize=address,undefined,leak -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope"
    )
endif()

# Respect user overrides performed via CMAKE_CXX_FLAGS/CMAKE_EXE_LINKER_FLAGS
string(APPEND DEAL_II_CXX_FLAGS " $ENV{CXXFLAGS} ${CMAKE_CXX_FLAGS}")
string(APPEND DEAL_II_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG}")
string(APPEND DEAL_II_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS_RELEASE}")
string(APPEND DEAL_II_LINKER_FLAGS " $ENV{LDFLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_EXE_LINKER_FLAGS "")

#
# Feature configuration:
#

set(EXTERNAL_TARGETS)

if(WITH_OPENMP)
  find_package(OpenMP REQUIRED)
  list(APPEND EXTERNAL_TARGETS "OpenMP::OpenMP_CXX")
endif()

if(WITH_EOSPAC)
  find_package(EOSPAC REQUIRED)
  list(APPEND EXTERNAL_TARGETS "Eospac::Eospac6")
endif()

if(WITH_LIKWID)
  find_package(LIKWID REQUIRED)
  list(APPEND EXTERNAL_TARGETS "Likwid::Likwid")
endif()

if(WITH_VALGRIND)
  find_package(VALGRIND REQUIRED)
  list(APPEND EXTERNAL_TARGETS "Valgrind::Valgrind")
endif()

#
# Set up the rest:
#

include(GNUInstallDirs)

add_subdirectory(source)

install(FILES COPYING.md README.md
  DESTINATION ${CMAKE_INSTALL_DOCDIR}
  )

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/run/")
file(CREATE_LINK "${CMAKE_SOURCE_DIR}/prm" "${CMAKE_BINARY_DIR}/run/prm" SYMBOLIC)
install(DIRECTORY prm DESTINATION ${CMAKE_INSTALL_DOCDIR})

enable_testing()
add_subdirectory(tests)

IF(WITH_DOXYGEN)
  add_subdirectory(doc)
ENDIF()
