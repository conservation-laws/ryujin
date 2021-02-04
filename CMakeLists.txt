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

if(NOT ${DEAL_II_WITH_CXX17})
  string(APPEND DEAL_II_CXX_FLAGS " -std=c++17")
endif()

string(APPEND DEAL_II_CXX_FLAGS_RELEASE " -O3 -ffp-contract=fast")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(APPEND DEAL_II_CXX_FLAGS " -fdiagnostics-color=always")
  string(REPLACE "-Qunused-arguments" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")
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

option(DOCUMENTATION
  "Build the documentation with doxygen" OFF
  )

option(LIKWID_PERFMON
  "Compile and link against the likwid instrumentation library" OFF
  )

#
# Options:
#

set(DIM "2" CACHE STRING "The dimension")
set(NUMBER "double" CACHE STRING "The principal floating point type")

option(CHECK_BOUNDS "Enable debug code paths that check limiter bounds" OFF)
option(DEBUG_OUTPUT "Enable detailed time-step output" OFF)
option(DENORMALS_ARE_ZERO "Set the \"denormals are zero\" and \"flush to zero\" bits in the MXCSR register" OFF)
option(FORCE_DEAL_II_SPARSE_MATRIX "Always use dealii::SparseMatrix instead of TrilinosWrappers::SparseMatrix for assembly" OFF)
option(OBSESSIVE_INLINING "Also inline the Riemann solver and limiter calls" OFF)
option(USE_COMMUNICATION_HIDING "Issue MPI synchronization of ghost values early" ON)
option(USE_CUSTOM_POW "Use custom pow implementation" ON)
option(USE_SIMD "Use SIMD vectorization" ON)

set(ORDER_FINITE_ELEMENT "1" CACHE STRING "Order of finite elements")
set(ORDER_MAPPING "1" CACHE STRING "Order of mapping")
set(ORDER_QUADRATURE "2" CACHE STRING "Order of quadrature")
set(EQUATION_OF_STATE "EquationOfState::ideal_gas" CACHE STRING "Equation of state")
set(NEWTON_EPS_DOUBLE "1.e-10" CACHE STRING "EPS double for limiter Newton solver")
set(NEWTON_EPS_FLOAT "1.e-4" CACHE STRING "EPS float for limiter Newton solver")
set(NEWTON_MAX_ITER "2" CACHE STRING "Maximal number of iterations in limiter Newton solver")
set(RIEMANN_NEWTON_MAX_ITER "0" CACHE STRING "Maximal number of iterations in Riemann solver")
set(INDICATOR "Indicators::entropy_viscosity_commutator" CACHE STRING "Indicator for preliminary high-order method")
set(COMPUTE_SECOND_VARIATIONS "true" CACHE STRING "Use second variations")
set(ENTROPY "Entropy::harten" CACHE STRING "Entropy for entropy-viscosity commutator")
set(SMOOTHNESS_INDICATOR "SmoothnessIndicators::pressure" CACHE STRING "Quantity used in smoothness indicator")
set(SMOOTHNESS_INDICATOR_ALPHA_0 "0." CACHE STRING "alpha_0 factor of smoothness indicator")
set(SMOOTHNESS_INDICATOR_POWER "3" CACHE STRING "exponent of smoothness indicator")
set(LIMITER "Limiters::specific_entropy" CACHE STRING "Select limiter")
set(LIMITER_RELAX_BOUNDS "true" CACHE STRING "Relax limiter bounds")
set(LIMITER_RELAXATION_ORDER "3" CACHE STRING "Relaxation order for limiter bounds")
mark_as_advanced(
  ORDER_FINITE_ELEMENT ORDER_MAPPING ORDER_QUADRATURE
  EQUATION_OF_STATE
  NEWTON_EPS_DOUBLE NEWTON_EPS_FLOAT NEWTON_MAX_ITER
  RIEMANN_NEWTON_MAX_ITER
  INDICATOR COMPUTE_SECOND_VARIATIONS ENTROPY
  SMOOTHNESS_INDICATOR SMOOTHNESS_INDICATOR_ALPHA_0
  SMOOTHNESS_INDICATOR_POWER
  LIMITER LIMITER_RELAX_BOUNDS LIMITER_RELAXATION_ORDER
  )

#
# Set up the rest:
#

enable_testing()

add_subdirectory(source)

add_subdirectory(tests)

IF(DOCUMENTATION)
  add_subdirectory(doc)
ENDIF()
