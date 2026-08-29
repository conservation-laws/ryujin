##
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## Copyright (C) 2026 by the ryujin authors
##

set(NVTX_HINTS ENV CUDA_HOME ENV CUDA_PATH ENV CUDA_ROOT)
set(NVTX_PATHS /opt/cuda /usr/local/cuda)

find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
  HINTS ${NVTX_HINTS}
  PATHS ${NVTX_PATHS}
  PATH_SUFFIXES include
  )

#
# The NVTX v3 API is header only, but we also use the CUDA profiler API
# (cudaProfilerStart() and cudaProfilerStop()) which requires the CUDA
# runtime library:
#

find_library(NVTX_CUDART_LIBRARY
  NAMES cudart
  HINTS ${NVTX_HINTS}
  PATHS ${NVTX_PATHS}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

find_package_handle_standard_args(NVTX DEFAULT_MSG
  NVTX_CUDART_LIBRARY NVTX_INCLUDE_DIR
  )

if(NVTX_FOUND AND NOT TARGET Nvtx::Nvtx)
  add_library(Nvtx::Nvtx INTERFACE IMPORTED)
  #
  # The NVTX headers load the injection library of an attached tool with
  # dlopen(), so we have to link against libdl as well:
  #
  target_link_libraries(Nvtx::Nvtx INTERFACE ${NVTX_CUDART_LIBRARY} ${CMAKE_DL_LIBS})
  target_include_directories(Nvtx::Nvtx SYSTEM INTERFACE ${NVTX_INCLUDE_DIR})
endif()
