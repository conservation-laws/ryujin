#include <sparse_matrix.h>
#include <sparsity_pattern.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/sparsity_tools.h>

#include <chrono>
#include <vector>

/*
 * Test SparseMatrix::update_ghost_rows() on the host and on the default
 * memory space: We populate the locally owned rows with a value that only
 * depends on the global row and column index, exchange the ghost rows over
 * both memory spaces, and compare the results.
 */

using HostSpace = dealii::MemorySpace::Host;
using DefaultSpace = dealii::MemorySpace::Default;

int main(int argc, char *argv[])
{
  using namespace std::chrono_literals;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const auto mpi_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const auto n_mpi_processes =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  constexpr int dim = 2;

  dealii::parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD);
  dealii::GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);

  dealii::FE_Q<dim> fe(1);

  dealii::DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const dealii::IndexSet &locally_owned = dof_handler.locally_owned_dofs();

  auto locally_relevant =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  dealii::AffineConstraints<double> affine_constraints;
  affine_constraints.reinit(locally_owned, locally_relevant);
  affine_constraints.close();

  dealii::DynamicSparsityPattern dsp;
  dsp.reinit(dof_handler.n_dofs(), dof_handler.n_dofs(), locally_relevant);
  dealii::DoFTools::make_sparsity_pattern(
      dof_handler, dsp, affine_constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned, MPI_COMM_WORLD, locally_relevant);

  const auto partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();

  /*
   * The sparsity pattern has to be accessible on both memory spaces:
   */
  ryujin::SparsityPattern<simd_width> sparsity_pattern(
      0,
      dsp,
      partitioner,
      /*symmetrize ghost range*/ true,
      ryujin::TransferPolicy::implicit_transfers);
  const auto sparsity_pattern_view = sparsity_pattern.view();

  ryujin::SparseMatrix<double, 1, simd_width> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);

  const unsigned int n_locally_owned = partitioner->locally_owned_size();
  const unsigned int n_locally_relevant = locally_relevant.n_elements();

  /* Populate the locally owned rows: */

  {
    const auto view = sparse_matrix.view();
    for (unsigned int i = 0; i < n_locally_owned; ++i) {
      const auto i_global = partitioner->local_to_global(i);
      const unsigned int row_length = sparsity_pattern_view.row_length(i);
      const unsigned int *js = sparsity_pattern_view.columns(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {
        const auto j_global = partitioner->local_to_global(*js);
        view.write_entry(1000. * i_global + j_global, i, col_idx);
      }
    }
  }

  /* Read back the ghost range of the matrix: */

  const auto read_ghost_rows = [&]() {
    std::vector<double> result;
    const auto &const_matrix = sparse_matrix;
    const auto view = const_matrix.view<HostSpace>();
    for (unsigned int i = n_locally_owned; i < n_locally_relevant; ++i) {
      const unsigned int row_length = sparsity_pattern_view.row_length(i);
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx)
        result.push_back(view.read_entry(i, col_idx));
    }
    return result;
  };

  /* Exchange ghost rows on the host memory space: */

  sparse_matrix.view<HostSpace>().update_ghost_rows();
  const auto host_result = read_ghost_rows();

  /* Exchange ghost rows on the default memory space: */

  sparse_matrix.view<HostSpace>().zero_out_ghost_rows();
  sparse_matrix.move_to_memory_space<DefaultSpace>();
  sparse_matrix.view<DefaultSpace>().update_ghost_rows();
  sparse_matrix.move_to_memory_space<HostSpace>();
  const auto default_result = read_ghost_rows();

  for (unsigned int p = 0; p < n_mpi_processes; ++p) {
    if (p == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\n\nGhost rows after update_ghost_rows():\n";
      std::cout << "Rank " << mpi_rank << std::endl;
      for (std::size_t k = 0; k < host_result.size(); ++k)
        std::cout << host_result[k] << " " << default_result[k]
                  << (host_result[k] == default_result[k] ? " ok" : " MISMATCH")
                  << std::endl;
    }
    std::this_thread::sleep_for(200ms);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
