##
## SPDX-License-Identifier: MIT
## Copyright (C) 2020 by the ryujin authors
##

find_package(Doxygen REQUIRED)

find_path(MATHJAX_PATH MathJax.js PATHS
  ${MATHJAX_PATH} $ENV{MATHJAX_PATH}
  /usr/share/mathjax /usr/share/javascript/mathjax
  )

if(MATHJAX_PATH MATCHES "MATHJAX_PATH-NOTFOUND")
  set(MATHJAX_PATH "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/")
endif()

message(STATUS "Using MathJax path: »${MATHJAX_PATH}«")

file(GLOB_RECURSE DOXYGEN_INPUT
  ${CMAKE_SOURCE_DIR}/source/*.h
  ${CMAKE_SOURCE_DIR}/doc/headers/*.h
  )
string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT}")

file(GLOB_RECURSE DOXYGEN_DEPEND
  ${CMAKE_CURRENT_SOURCE_DIR}/options.dox.in
  ${CMAKE_SOURCE_DIR}/source/*.h
  ${CMAKE_SOURCE_DIR}/doc/headers/*.h
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/options.dox.in
  ${CMAKE_CURRENT_BINARY_DIR}/options.dox
  @ONLY
  )

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/doxygen.log
  COMMAND ${DOXYGEN_EXECUTABLE}
    ${CMAKE_CURRENT_BINARY_DIR}/options.dox
    > ${CMAKE_BINARY_DIR}/doxygen.log 2>&1
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${DOXYGEN_DEPEND}
  COMMENT "Generating documentation via doxygen."
  VERBATIM
  )

add_custom_target(doxygen ALL
  DEPENDS ${CMAKE_BINARY_DIR}/doxygen.log
  )

install(DIRECTORY
  ${CMAKE_CURRENT_BINARY_DIR}/html
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/html
  )
