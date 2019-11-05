SET(RYUJIN_VERSION 1.2.0)

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

#
# Find deal.II and setup up compiler flags:
#

FIND_PACKAGE(deal.II 9.0 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_QUERY_GIT_INFORMATION()

IF(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_THREADS OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR
    "Need a deal.II library with threads, mpi and p4est support enabled."
    )
ENDIF()

IF(CMAKE_CXX_COMPILER MATCHES "clang")
  STRING(APPEND DEAL_II_CXX_FLAGS
    " -Xclang -fcolor-diagnostics -Qunused-arguments"
    )
  STRING(APPEND DEAL_II_LINKER_FLAGS " -Wl,-as-needed")

  STRING(APPEND DEAL_II_CXX_FLAGS_RELEASE
    " -march=native -Ofast -ffp-contract=fast"
    )

  STRING(APPEND DEAL_II_CXX_FLAGS_DEBUG
    " -fsanitize=address -fsanitize-address-use-after-scope"
    )
  STRING(APPEND DEAL_II_LINKER_FLAGS_DEBUG
    " -fsanitize=address -fsanitize-address-use-after-scope"
    )

ELSE()
  STRING(REPLACE "-Xclang" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")
  STRING(REPLACE "-fcolor-diagnostics" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")
  STRING(REPLACE "-Qunused-arguments" "" DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS}")
ENDIF()


PROJECT(ryujin CXX)

IF(POLICY CMP0042)
  CMAKE_POLICY(SET CMP0042 NEW)
ENDIF()

#
# Options:
#

SET(DIM "2" CACHE STRING "The dimension")

SET(NUMBER "double" CACHE STRING "The principal floating point type")

OPTION(USE_SIMD "Use SIMD vectorization" ON)

OPTION(USE_CUSTOM_POW "Use custom pow implementation" OFF)

OPTION(CHECK_BOUNDS "Enable debug code paths that check limiter bounds" OFF)

OPTION(DEBUG_OUTPUT "Enable detailed time-step output" OFF)

#
# Set up the rest:
#

ADD_SUBDIRECTORY(grendel)
ADD_SUBDIRECTORY(ryujin)
ADD_SUBDIRECTORY(run)

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
