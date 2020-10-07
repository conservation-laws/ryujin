//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#include <compile_time_options.h>

#include "introspection.h"
#include "time_loop.h"

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <omp.h>

#include <fstream>

int main (int argc, char *argv[])
{
  LSAN_DISABLE

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  omp_set_num_threads(dealii::MultithreadInfo::n_threads());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  RYUJIN_PARALLEL_REGION_BEGIN
  LIKWID_MARKER_THREADINIT;
  RYUJIN_PARALLEL_REGION_END
#endif

  LSAN_ENABLE

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  ryujin::TimeLoop<DIM, NUMBER> time_loop(mpi_communicator);

  if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "[INFO] initiating flux capacitor" << std::endl;

  AssertThrow(
      argc <= 2,
      dealii::ExcMessage("Invalid number of parameters. At most one argument "
                         "supported which has to be a parameter file"));

  dealii::ParameterAcceptor::initialize(argc == 2 ? argv[1] : "ryujin.prm");

  time_loop.run();

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
