file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/run/")

file(GLOB _directories CONFIGURE_DEPENDS *)
foreach(_directory ${_directories})
  if(IS_DIRECTORY "${_directory}")
    get_filename_component(_equation "${_directory}" NAME)
    file(CREATE_LINK
      "${_directory}" "${CMAKE_BINARY_DIR}/run/${_equation}" SYMBOLIC
      )
  endif()
endforeach()
