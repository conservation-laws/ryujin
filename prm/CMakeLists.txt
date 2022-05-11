macro(setup_equation EQUATION)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/run/${EQUATION}")

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/run/${EQUATION}/prm"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${CMAKE_CURRENT_SOURCE_DIR}/${EQUATION}"
      "${CMAKE_BINARY_DIR}/run/${EQUATION}/prm"
      )

  add_custom_target(symlink_${EQUATION}_prm ALL
    DEPENDS ${EQUATION} "${CMAKE_BINARY_DIR}/run/${EQUATION}/prm"
    COMMENT "updating ${EQUATION} prm symlink"
    )

  install(DIRECTORY ${EQUATION} DESTINATION ${CMAKE_INSTALL_DOCDIR}/prm/)
endmacro()

setup_equation("euler")
if(NOT "${DIM}" EQUAL "3")
  setup_equation("shallow_water")
endif()
