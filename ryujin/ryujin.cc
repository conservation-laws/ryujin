#include "timeloop.h"

#include <compile_time_options.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/multithread_info.h>

#include <omp.h>

#ifdef LIKWID_PERFMON
  #include <likwid.h>
#endif

#include <fstream>

int main (int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  /* Set the number of OpenMP threads to whatever deal.II allows for TBB: */
  omp_set_num_threads(dealii::MultithreadInfo::n_threads());

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  ryujin::TimeLoop<DIM, NUMBER> time_loop(mpi_communicator);

  /* If necessary, create empty parameter file and exit: */
  dealii::ParameterAcceptor::initialize("ryujin.prm");

  time_loop.run();

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
