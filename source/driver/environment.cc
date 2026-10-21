//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/instrumentation.h>
#include <ryujin/driver/environment.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <algorithm>

namespace ryujin
{
  namespace
  {
    /**
     * Change rounding mode on X86-64 architecture: Denormals are flushed to
     * zero to avoid computing on denormals which can slow down computations
     * significantly.
     */
    void flush_denormals_to_zero()
    {
#if defined(DENORMALS_ARE_ZERO) && defined(__x86_64)
#define MXCSR_DAZ (1 << 6)  /* Enable denormals are zero mode */
#define MXCSR_FTZ (1 << 15) /* Enable flush to zero mode */

      unsigned int mxcsr = __builtin_ia32_stmxcsr();
      mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
      __builtin_ia32_ldmxcsr(mxcsr);
#endif
    }


    /**
     * Set up thread pools and obey thread limits:
     */
    void set_thread_limit(const MPI_Comm &mpi_communicator [[maybe_unused]])
    {
      unsigned int n_threads [[maybe_unused]] =
          dealii::MultithreadInfo::n_threads();

#ifdef WITH_OPENMP
      const unsigned int omp_thread_limit = omp_get_thread_limit();
      const unsigned int omp_max_threads = omp_get_max_threads();
      n_threads = std::min({omp_thread_limit, omp_max_threads, n_threads});
      omp_set_num_threads(n_threads);
#endif

#ifdef WITH_DEAL_II_THREADS
      dealii::MultithreadInfo::set_thread_limit(n_threads);
#else
      dealii::MultithreadInfo::set_thread_limit(1);
#endif
    }
  } // namespace


  Environment::Environment(int &argc, char **&argv)
  {
    flush_denormals_to_zero();

    LSAN_DISABLE;
    mpi_initialization_.emplace(argc, argv);
    set_thread_limit(mpi_communicator());
    LSAN_ENABLE

    LIKWID_MARKER_INIT;
    NVTX_MARKER_INIT;
#ifdef WITH_OPENMP
    RYUJIN_PRAGMA(omp parallel)
    {
      LIKWID_MARKER_THREADINIT;
    }
#endif
  }


  Environment::~Environment()
  {
    LIKWID_MARKER_CLOSE;
    NVTX_MARKER_CLOSE;
    LSAN_DISABLE;
  }
} // namespace ryujin
