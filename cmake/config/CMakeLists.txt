##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

#
# The project configuration ryujinConfig.cmake: it is configured twice,
# for the build directory and for the installation, and determines the
# location of the project relative to its own location. The build
# directory mirrors the installation layout under ${CMAKE_BINARY_DIR}.
#

set(RYUJIN_PROJECT_CONFIG_RELDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ryujin)
set(_build_config_dir ${CMAKE_BINARY_DIR}/${RYUJIN_PROJECT_CONFIG_RELDIR})

#
# Targets:
#

export(EXPORT ryujinTargets
  NAMESPACE ryujin::
  FILE ${_build_config_dir}/ryujinTargets.cmake
  )
install(EXPORT ryujinTargets
  NAMESPACE ryujin::
  DESTINATION ${RYUJIN_PROJECT_CONFIG_RELDIR}
  )

#
# Find modules of the external libraries:
#

set(_modules
  ${CMAKE_SOURCE_DIR}/cmake/FindEOSPAC.cmake
  ${CMAKE_SOURCE_DIR}/cmake/FindLIKWID.cmake
  ${CMAKE_SOURCE_DIR}/cmake/FindNVTX.cmake
  )
file(COPY ${_modules} DESTINATION ${_build_config_dir}/modules)
install(FILES ${_modules} DESTINATION ${RYUJIN_PROJECT_CONFIG_RELDIR}/modules)

#
# Feature configuration:
#

set(FEATURE_CONFIGURATION_STRING "")
foreach(_feature
    DEAL_II_THREADS EOSPAC GDAL LIKWID NVTX OPENMP
    )
  if(WITH_${_feature})
    set(_value ON)
  else()
    set(_value OFF)
  endif()
  string(APPEND FEATURE_CONFIGURATION_STRING
    "set(RYUJIN_WITH_${_feature} ${_value})\n"
    )
endforeach()

file(GLOB _files RELATIVE ${CMAKE_SOURCE_DIR}/source CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/source/*/CMakeLists.txt
  )
foreach(_file ${_files})
  get_filename_component(_directory "${_file}" DIRECTORY)
  string(TOUPPER "${_directory}" _directory_upper)
  if(EQUATION_${_directory_upper})
    set(_value ON)
  else()
    set(_value OFF)
  endif()
  string(APPEND FEATURE_CONFIGURATION_STRING
    "set(RYUJIN_WITH_EQUATION_${_directory_upper} ${_value})\n"
    )
endforeach()

#
# Version and configuration files:
#

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${_build_config_dir}/ryujinConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion
  )

set(CONFIG_BUILD_DIR TRUE)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  ${_build_config_dir}/ryujinConfig.cmake
  @ONLY
  )

set(CONFIG_BUILD_DIR FALSE)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ryujinConfig.cmake
  @ONLY
  )

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/ryujinConfig.cmake
  ${_build_config_dir}/ryujinConfigVersion.cmake
  DESTINATION ${RYUJIN_PROJECT_CONFIG_RELDIR}
  )
