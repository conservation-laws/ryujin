#include "cut_cell_primitives_helpers.h"

//
// Test MeshAlignment and CutCellDecomposition on a serial triangulation.
//

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  run_cases<dealii::Triangulation<dim>, dealii::Triangulation<dim>>();
}
