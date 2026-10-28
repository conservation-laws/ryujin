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
  driver/dashboard_text.cc
  driver/environment.cc
  driver/equation.cc
  driver/equation_euler.cc
  driver/equation_euler_aeos.cc
  driver/equation_euler_barotropic.cc
  driver/equation_euler_poisson.cc
  driver/equation_euler_poisson_aeos.cc
  driver/equation_euler_poisson_barotropic.cc
  driver/equation_navier_stokes.cc
  driver/equation_scalar_conservation.cc
  driver/equation_shallow_water.cc
  driver/logger.cc
  driver/parameter_editor.cc
  driver/parameter_model.cc
  driver/scheduler.cc
  driver/terminal.cc
  geometries/geometry_library.cc
  linear_algebra/multicomponent_vector.cc
  linear_algebra/sparsity_pattern.cc
  output/error_evaluation.cc
  output/quantities.cc
  output/vtu_output.cc
  )

if(WITH_FTXUI)
  list(APPEND CORE_SOURCE_FILES
    driver/dashboard.cc
    driver/dashboard_animation.cc
    driver/dashboard_theme.cc
    driver/parameter_editor_ui.cc
    driver/splash_screen.cc
    )
endif()

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
foreach(_equation
    euler euler_aeos euler_barotropic euler_poisson
    navier_stokes scalar_conservation shallow_water
    )
  add_subdirectory(${_equation})
  list(APPEND OBJECT_TARGETS object_${_equation})
endforeach()

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
