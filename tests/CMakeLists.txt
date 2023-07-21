find_program(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )
mark_as_advanced(NUMDIFF_EXECUTABLE)

if(NOT "${NUMDIFF_EXECUTABLE}" MATCHES "NUMDIFF_EXECUTABLE-NOTFOUND")
  file(GLOB _files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS */CMakeLists.txt)
  foreach(_file ${_files})
    get_filename_component(_directory "${_file}" DIRECTORY)
    add_subdirectory("${_directory}")
  endforeach()
endif()
