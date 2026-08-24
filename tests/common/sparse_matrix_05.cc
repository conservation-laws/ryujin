#include <sparse_matrix.h>
#include <sparsity_pattern.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <chrono>

/*
 * Test distribute_local_to_global()
 */

int main(int argc, char *argv[])
{
  using namespace std::chrono_literals;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const auto mpi_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const auto n_mpi_processes =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  /*
   * Create a unit square twice globally refined and the bottom left
   * quadrant once more. Creating a mesh with 4 hanging nodes.
   */

  constexpr int dim = 2;

  dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  triangulation.begin_active()->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();
  triangulation.refine_global(1);

  dealii::FE_Q<dim> fe(1);

  /* Distribute DoFs and set up locally owned and relevant ranges: */

  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const dealii::IndexSet &locally_owned = dof_handler.locally_owned_dofs();

  auto locally_relevant =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.reinit(locally_owned, locally_relevant);
  dealii::DoFTools::make_hanging_node_constraints(dof_handler,
                                                  affine_constraints);
  affine_constraints.close();

  /* We should be consistent... */
  if (!affine_constraints.is_consistent_in_parallel(
          dealii::Utilities::MPI::all_gather(MPI_COMM_WORLD, locally_owned),
          locally_relevant,
          MPI_COMM_WORLD,
          true)) {
    std::cout << "Oh Nooo!" << std::endl;
    __builtin_trap();
  }

  dealii::DynamicSparsityPattern dsp;
  dsp.reinit(dof_handler.n_dofs(), dof_handler.n_dofs(), locally_relevant);
  dealii::DoFTools::make_sparsity_pattern(
      dof_handler, dsp, affine_constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned, MPI_COMM_WORLD, locally_relevant);

  /* Enlarge the locally relevant set to include all additional couplings: */

  dealii::IndexSet additional_dofs(dof_handler.n_dofs());
  for (auto &entry : dsp)
    if (!locally_relevant.is_element(entry.column())) {
      Assert(locally_owned.is_element(entry.row()), dealii::ExcInternalError());
      additional_dofs.add_index(entry.column());
    }
  additional_dofs.compress();
  locally_relevant.add_indices(additional_dofs);
  locally_relevant.compress();
  const auto n_locally_relevant = locally_relevant.n_elements();

  const auto partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  /* Create final sparsity pattern: */

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();
  ryujin::SparsityPattern<simd_width> sparsity_pattern(0, dsp, partitioner);
  const auto sparsity_pattern_view = sparsity_pattern.view();

  /* Create a sparse matrix: */

  ryujin::SparseMatrix<double, 1, simd_width> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);
  const auto sparse_matrix_view = sparse_matrix.view();

  /* Add a local contribution from all owning cells: */
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
    cell->get_dof_indices(dof_indices);

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        if (i == j)
          cell_matrix(i, j) = std::pow(10., mpi_rank);
        else
          cell_matrix(i, j) = -std::pow(10., mpi_rank);

    ryujin::distribute_local_to_global(
        cell_matrix, dof_indices, affine_constraints, sparse_matrix);
  }

  sparse_matrix_view.compress(dealii::VectorOperation::add);

  const auto print_matrix = [&]() {
    for (unsigned int i = 0; i < n_locally_relevant; ++i) {
      const auto i_global = partitioner->local_to_global(i);
      const unsigned int row_length = sparsity_pattern_view.row_length(i);
      const unsigned int *js = sparsity_pattern_view.columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {
        const auto j_global = partitioner->local_to_global(*js);
        std::cout << "(" << i_global << "," << j_global << ") "
                  << sparse_matrix_view.read_entry(i, col_idx) << std::endl;
      }
    }
  };

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\n\nSparse matrix contents:\n";
      std::cout << "Rank " << mpi_rank << std::endl;
      print_matrix();
    }
    std::this_thread::sleep_for(200ms);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
