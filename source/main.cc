//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#include <compile_time_options.h>

#include "equation_dispatch.h"
#include "instrumentation.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <filesystem>

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


/**
 * The main function
 */
int main(int argc, char *argv[])
{
  /*
   * Entry point into the ryujin executable. We first determine what
   * parameter file to use and then hand off everything to an
   * EquationDispatch instance.
   */

  flush_denormals_to_zero();

  LSAN_DISABLE;
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);
  set_thread_limit(mpi_communicator);
  LSAN_ENABLE

  LIKWID_MARKER_INIT;
#ifdef WITH_OPENMP
  RYUJIN_PRAGMA(omp parallel)
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::cout << "[INFO] initiating flux capacitor" << std::endl;
  }

  if (argc > 2) {
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      std::cout << "[ERROR] Invalid number of parameters. At most one argument "
                << "supported which has to be a parameter file." << std::endl;
    }

    LIKWID_MARKER_CLOSE;
    LSAN_DISABLE;
    return 1;
  }

  const auto executable_name = std::filesystem::path(argv[0]).filename();
  std::string parameter_file = executable_name.string() + ".prm";

  if (argc == 2) {
    parameter_file = argv[1];

    if (!std::filesystem::exists(parameter_file)) {
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
        std::cout << "[ERROR] The specified parameter file »" << parameter_file
                  << "« does not exist." << std::endl;
      }

      LIKWID_MARKER_CLOSE;
      LSAN_DISABLE;
      return 1;
    }
  }

  if (!std::filesystem::exists(parameter_file)) {
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
      std::cout << "[INFO] Default parameter file »" << parameter_file
                << "« not found.\n[INFO] Creating template parameter files for "
                << "you. Please modify and rename one of the templates to »"
                << parameter_file << "«." << std::endl;
      ryujin::EquationDispatch::create_parameter_files();
    }

    MPI_Barrier(mpi_communicator);

    LIKWID_MARKER_CLOSE;
    LSAN_DISABLE;
    return 1;
  }

  {
    ryujin::EquationDispatch equation_dispatch;
    equation_dispatch.dispatch(parameter_file, mpi_communicator);
  }

  LIKWID_MARKER_CLOSE;
  LSAN_DISABLE;
  return 0;
}
