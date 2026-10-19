##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2020 - 2026 by the ryujin authors
##

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_time_options.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/compile_time_options.h
  )

#
# Common and equation-independent source files and target:
#

set(COMMON_SOURCE_FILES
  discretization.cc
  equation_dispatch.cc
  error_evaluation.cc
  mesh_adaptor.cc
  mpi_ensemble.cc
  multicomponent_vector.cc
  offline_data.cc
  quantities.cc
  simd.cc
  sparsity_pattern.cc
  time_integrator.cc
  time_loop.cc
  version_info.cc
  vtu_output.cc
  )

set_property(SOURCE version_info.cc APPEND PROPERTY COMPILE_DEFINITIONS
  RYUJIN_VERSION="${RYUJIN_VERSION}"
  RYUJIN_GIT_REVISION="${GIT_REVISION}"
  RYUJIN_GIT_SHORTREV="${GIT_SHORTREV}"
  )

add_library(obj_common OBJECT ${COMMON_SOURCE_FILES})
target_include_directories(obj_common PUBLIC
  ${CMAKE_BINARY_DIR}/source/
  ${CMAKE_SOURCE_DIR}/include/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
deal_ii_setup_target(obj_common)
target_link_libraries(obj_common ${EXTERNAL_TARGETS})

#
# Common, equation-dependent source files:
#

#
# Sources that depend on the hyperbolic description only. They are
# compiled once per hyperbolic description:
#
set(HYPERBOLIC_DEPENDENT_SOURCE_FILES
  hyperbolic_module.cc
  initial_values.cc
  solution_transfer.cc
  )

#
# Equation-specific source files and compilation targets:
#

set(OBJECT_TARGETS)
macro(setup_equation EQUATION)
  #
  # obj_${EQUATION} object library: equation specific code:
  #

  add_subdirectory(${EQUATION})

  #
  # obj_${EQUATION}_dependent object library: equation dependent common
  # code. Only an equation directory with a hyperbolic_description.h has
  # one. All other equation directories pair a parabolic description with
  # the hyperbolic description of another equation directory.
  #

  list(APPEND OBJECT_TARGETS obj_${EQUATION})

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${EQUATION}/hyperbolic_description.h)
    add_library(obj_${EQUATION}_dependent OBJECT
      ${HYPERBOLIC_DEPENDENT_SOURCE_FILES}
      )
    deal_ii_setup_target(obj_${EQUATION}_dependent)

    target_link_libraries(obj_${EQUATION}_dependent
      obj_${EQUATION} ${EXTERNAL_TARGETS}
      )

    list(APPEND OBJECT_TARGETS obj_${EQUATION}_dependent)
  endif()
endmacro()

file(GLOB _files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS */CMakeLists.txt)
foreach(_file ${_files})
  get_filename_component(_directory "${_file}" DIRECTORY)

  string(TOUPPER "${_directory}" _directory_upper)
  set(EQUATION_${_directory_upper} ${DEFAULT_ENABLE_EQUATIONS} CACHE BOOL "Enable the equation »${_directory}« compute module")

  if(EQUATION_${_directory_upper})
    setup_equation("${_directory}")
  endif()
endforeach()

#
# The executable
#

add_executable(ryujin main.cc)
deal_ii_setup_target(ryujin)
target_link_libraries(ryujin obj_common ${OBJECT_TARGETS} ${EXTERNAL_TARGETS})

set_target_properties(ryujin PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/run"
  )

install(TARGETS ryujin DESTINATION ${CMAKE_INSTALL_BINDIR})
