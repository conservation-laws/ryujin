#include "cut_cell_primitives_helpers.h"

//
// Test MeshAlignment and CutCellDecomposition on a
// parallel::distributed::Triangulation creating a
// parallel::fullydistributed::Triangulation: the output must be identical
// for any number of MPI ranks and to the serial output of
// cut_cell_primitives_01.
//

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  run_cases<dealii::parallel::distributed::Triangulation<dim>,
            dealii::parallel::fullydistributed::Triangulation<dim>>();
}
