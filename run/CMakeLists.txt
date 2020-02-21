SET(_files cylinder.prm validation.baseline validation.prm)
FOREACH(_file ${_files})
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
    ${CMAKE_CURRENT_BINARY_DIR}/${_file}
    COPYONLY
    )
ENDFOREACH()
