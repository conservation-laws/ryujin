find_program(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )
mark_as_advanced(NUMDIFF_EXECUTABLE)

if(NOT "${NUMDIFF_EXECUTABLE}" MATCHES "NUMDIFF_EXECUTABLE-NOTFOUND")
  add_subdirectory("common")
  add_subdirectory("euler")
  add_subdirectory("euler_aeos")
  add_subdirectory("shallow_water")
endif()
