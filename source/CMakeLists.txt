##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2026 by the ryujin authors
##

configure_file(
  ${PROJECT_SOURCE_DIR}/include/ryujin/base/compile_time_options.h.in
  ${PROJECT_BINARY_DIR}/include/ryujin/base/compile_time_options.h
  )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ryujin
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
  )
install(FILES ${PROJECT_BINARY_DIR}/include/ryujin/base/compile_time_options.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ryujin/base
  )

# All object files end up in the shared library libryujin.so:
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# Equation-independent core source files and target:
#

set(CORE_SOURCE_FILES
  base/mpi_ensemble.cc
  base/simd.cc
  base/version_info.cc
  discretization/discretization.cc
  discretization/mesh_adaptor.cc
  discretization/offline_data.cc
  discretization/time_integrator.cc
  driver/checkpointing.cc
  driver/environment.cc
  driver/equation_library.cc
  driver/logger.cc
  driver/run.cc
  driver/scheduler.cc
  geometries/geometry_library.cc
  linear_algebra/multicomponent_vector.cc
  linear_algebra/sparsity_pattern.cc
  output/error_evaluation.cc
  output/quantities.cc
  output/vtu_output.cc
  )

set_property(SOURCE base/version_info.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

add_library(object_core OBJECT ${CORE_SOURCE_FILES})
target_include_directories(object_core PUBLIC
  ${PROJECT_BINARY_DIR}/include/
  ${PROJECT_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
deal_ii_setup_target(object_core)
target_link_libraries(object_core ${EXTERNAL_TARGETS})

#
# Equation-specific source files and compilation targets:
#

set(OBJECT_TARGETS)
set(EQUATION_DEFINITIONS)
macro(setup_equation EQUATION)
  #
  # object_${EQUATION} object library: equation specific code:
  #

  add_subdirectory(${EQUATION})

  list(APPEND OBJECT_TARGETS object_${EQUATION})
endmacro()

file(GLOB _files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS */CMakeLists.txt)
foreach(_file ${_files})
  get_filename_component(_directory "${_file}" DIRECTORY)

  string(TOUPPER "${_directory}" _directory_upper)
  set(EQUATION_${_directory_upper} ${DEFAULT_ENABLE_EQUATIONS} CACHE BOOL "Enable the equation »${_directory}« compute module")

  if(EQUATION_${_directory_upper})
    setup_equation("${_directory}")
    list(APPEND EQUATION_DEFINITIONS RYUJIN_WITH_EQUATION_${_directory_upper})
  endif()
endforeach()

set_property(SOURCE driver/equation_library.cc APPEND PROPERTY COMPILE_DEFINITIONS
  ${EQUATION_DEFINITIONS}
  )

#
# The shared library libryujin.so:
#

set(_objects)
foreach(_target object_core ${OBJECT_TARGETS})
  list(APPEND _objects $<TARGET_OBJECTS:${_target}>)
endforeach()

add_library(libryujin SHARED ${_objects})

deal_ii_setup_target(libryujin)
target_link_libraries(libryujin ${EXTERNAL_TARGETS})

target_compile_features(libryujin PUBLIC cxx_std_20)
target_include_directories(libryujin PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

set_target_properties(libryujin PROPERTIES
  OUTPUT_NAME ryujin
  EXPORT_NAME ryujin
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
  )

# The name of the target for projects that include ryujin with
# add_subdirectory(), identical to the imported target of
# find_package(ryujin):
add_library(ryujin::ryujin ALIAS libryujin)

install(TARGETS libryujin
  EXPORT ryujinTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

#
# The ryujin executable:
#

add_executable(ryujin main.cc)

deal_ii_setup_target(ryujin)
target_link_libraries(ryujin libryujin)

set_target_properties(ryujin PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/run"
  )

# Locate the installed libryujin.so unless it lands in a system directory:
if(NOT "${CMAKE_INSTALL_FULL_LIBDIR}" IN_LIST CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES)
  file(RELATIVE_PATH _libdir
    ${CMAKE_INSTALL_FULL_BINDIR} ${CMAKE_INSTALL_FULL_LIBDIR}
    )
  set_target_properties(ryujin PROPERTIES INSTALL_RPATH "$ORIGIN/${_libdir}")
endif()

install(TARGETS ryujin RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
