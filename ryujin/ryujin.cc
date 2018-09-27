#include "timeloop.h"

#include <deal.II/base/utilities.h>

#include <fstream>

int main (int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  ryujin::TimeLoop<DIM> time_loop(mpi_communicator);

  /*
   * If necessary, create empty parameter file and exit:
   */

  const auto filename = "ryujin.prm";
  if (!std::ifstream(filename)) {
    std::ofstream file(filename);
    dealii::ParameterAcceptor::prm.print_parameters(
        file, dealii::ParameterHandler::OutputStyle::Text);
    return 0;
  }

  time_loop.run();
  return 0;
}