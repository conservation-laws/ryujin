find_program(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )
mark_as_advanced(NUMDIFF_EXECUTABLE)

if(NOT ${CMAKE_VERSION} VERSION_LESS 3.15
   AND NOT "${NUMDIFF_EXECUTABLE}" MATCHES "NUMDIFF_EXECUTABLE-NOTFOUND")
  add_subdirectory("euler")
  if(NOT "${DIM}" EQUAL "3")
    add_subdirectory("shallow_water")
  endif()
endif()
