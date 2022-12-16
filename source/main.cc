//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#include <compile_time_options.h>

#include "introspection.h"
//#include "time_loop.h"

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <filesystem>
#include <fstream>

int main(int argc, char *argv[])
{
#if 0
#if defined(DENORMALS_ARE_ZERO) && defined(__x86_64)
  /*
   * Change rounding mode on X86-64 architecture: Denormals are flushed to
   * zero to avoid computing on denormals which can slow down computations
   * significantly.
   */
#define MXCSR_DAZ (1 << 6)  /* Enable denormals are zero mode */
#define MXCSR_FTZ (1 << 15) /* Enable flush to zero mode */

  unsigned int mxcsr = __builtin_ia32_stmxcsr();
  mxcsr |= MXCSR_DAZ | MXCSR_FTZ;
  __builtin_ia32_ldmxcsr(mxcsr);
#endif

  LSAN_DISABLE

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

#ifdef WITH_OPENMP
  const unsigned int n_threads_omp = omp_get_thread_limit();
  const unsigned int n_threads_dealii = dealii::MultithreadInfo::n_threads();
  const unsigned int n_threads = std::min(n_threads_omp, n_threads_dealii);
  omp_set_num_threads(n_threads);
  dealii::MultithreadInfo::set_thread_limit(n_threads);
#else
  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "[INFO] OpenMP support disabled, set thread limit to one"
              << std::endl;
  dealii::MultithreadInfo::set_thread_limit(1);
#endif

  LSAN_ENABLE
  LIKWID_INIT

  ryujin::TimeLoop<DIM, NUMBER> time_loop(mpi_communicator);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "[INFO] initiating flux capacitor" << std::endl;

  AssertThrow(
      argc <= 2,
      dealii::ExcMessage("Invalid number of parameters. At most one argument "
                         "supported which has to be a parameter file"));

  const auto executable_name = std::filesystem::path(argv[0]).filename();
  dealii::ParameterAcceptor::initialize(
      argc == 2 ? argv[1] : executable_name.string() + ".prm");

  time_loop.run();

  LIKWID_CLOSE;
#endif

  return 0;
}
