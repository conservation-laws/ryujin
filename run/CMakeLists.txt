ADD_CUSTOM_TARGET(generate_meshes)

FILE(GLOB _meshes ${CMAKE_CURRENT_SOURCE_DIR}/*.geo)
FOREACH(_mesh ${_meshes})
  GET_FILENAME_COMPONENT(_name ${_mesh} NAME_WE)

  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_name}.msh
    COMMAND gmsh -2 ${_mesh} -o ${CMAKE_CURRENT_BINARY_DIR}/${_name}.msh
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  ADD_CUSTOM_TARGET(generate_${_name}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_name}.msh
    )
  ADD_DEPENDENCIES(generate_meshes generate_${_name})
ENDFOREACH()

SET(_files disc.prm validation.baseline validation.prm wall.prm)
FOREACH(_file ${_files})
  CONFIGURE_FILE(
    ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
    ${CMAKE_CURRENT_BINARY_DIR}/${_file}
    COPYONLY
    )
ENDFOREACH()
