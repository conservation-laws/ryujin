#include "timeloop.h"

#include <deal.II/base/utilities.h>

#ifdef LIKWID_PERFMON
  #include <likwid.h>
#endif

#include <fstream>

int main (int argc, char *argv[])
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
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
