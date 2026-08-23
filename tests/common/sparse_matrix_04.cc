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
 * Test SparseMatrix::compress(VectorOperation::add)
 */

namespace ryujin
{
  template <int simd_length>
  class Debug : public SparsityPattern<simd_length>
  {
  public:
    Debug(const unsigned int n_internal_dofs,
          const dealii::DynamicSparsityPattern &sparsity,
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &partitioner)
        : SparsityPattern<simd_length>(n_internal_dofs,
                                       sparsity,
                                       partitioner,
                                       /*symmetrize ghost range*/ false)
    {
    }

    void print()
    {
      std::stringstream ss;

      ss << "Receive targets:\n";
      for (const auto &[left, right] : this->receive_targets())
        ss << left << " : " << right << "\n";

      ss << "Send targets:\n";
      for (const auto &[left, right] : this->send_targets())
        ss << left << " : " << right << "\n";

      ss << "Entries to be sent:\n";
      for (const auto &[left, right] : this->entries_to_be_sent())
        ss << left << " : " << right << "\n";

      std::cout << ss.str() << std::endl;
    }
  };
} // namespace ryujin


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

  /*
   * Distribute DoFs, set up locally owned and relevant ranges, and partitioner.
   */

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

  /* Create temporary sparsity pattern: */
  dealii::DoFTools::make_sparsity_pattern(
      dof_handler, dsp, affine_constraints, false);
  dealii::SparsityTools::distribute_sparsity_pattern(
      dsp, locally_owned, MPI_COMM_WORLD, locally_relevant);

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\nPreliminary sparsity pattern (global numbering):\n";
      std::cout << "Rank " << mpi_rank << std::endl;
      dsp.print(std::cout);
    }
    std::this_thread::sleep_for(200ms);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /*
   * Enlarge the locally relevant set to include all additional couplings:
   */

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

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\nIndex sets (owned and extended relevant):\n";
      std::cout << "Rank " << mpi_rank << std::endl;
      locally_owned.print(std::cout);
      locally_relevant.print(std::cout);
    }
    std::this_thread::sleep_for(200ms);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  const auto partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  /*
   * Create final sparsity pattern:
   */

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();
  ryujin::Debug<simd_width> sparsity_pattern(0, dsp, partitioner);
  const auto sparsity_pattern_view = sparsity_pattern.view();
  const auto print_sparsity = [&]() {
    for (unsigned int i = 0; i < n_locally_relevant; ++i) {
      const auto i_global = partitioner->local_to_global(i);
      const unsigned int row_length = sparsity_pattern_view.row_length(i);
      const unsigned int *js = sparsity_pattern_view.columns(i);
      std::cout << "[" << i_global;
      for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx, ++js) {
        const auto j_global = partitioner->local_to_global(*js);
        std::cout << "," << j_global;
      }
      std::cout << "]" << std::endl;
    }
  };

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\nModified sparsity pattern (global numbering):\n";
      std::cout << "Rank " << mpi_rank << std::endl;
      print_sparsity();
    }
    std::this_thread::sleep_for(200ms);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\nExchange pattern:\n";
      std::cout << "Rank " << mpi_rank << std::endl;
      sparsity_pattern.print();
    }
    std::this_thread::sleep_for(200ms);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /*
   * Create a sparse matrix:
   */

  ryujin::SparseMatrix<double, 1, simd_width> sparse_matrix;
  sparse_matrix.reinit(sparsity_pattern);
  const auto sparse_matrix_view = sparse_matrix.view();

  for (unsigned int i = 0; i < n_locally_relevant; ++i) {
    const unsigned int row_length = sparsity_pattern_view.row_length(i);
    for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
      sparse_matrix_view.write_entry(std::pow(10., mpi_rank), i, col_idx);
    }
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

#if 0
  /*
   * Reference values computed with Trilinos matrix: Note that diagonal
   * entries might differ between the two matrices because we always store
   * the diagonal entry in our custom sparsity pattern.
   */

  dealii::TrilinosWrappers::SparsityPattern trilinos_sparsity_pattern;
  trilinos_sparsity_pattern.reinit(locally_owned, dsp, MPI_COMM_WORLD);
  dealii::TrilinosWrappers::SparseMatrix trilinos_sparse_matrix(
      trilinos_sparsity_pattern);

  for (const auto &it : dsp) {
    const auto i = it.row();
    const auto j = it.column();
    trilinos_sparse_matrix.add(i, j, std::pow(10., mpi_rank));
  }

  trilinos_sparse_matrix.compress(dealii::VectorOperation::add);

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank) {
      if (mpi_rank == 0)
        std::cout << "\n\n(Reference) sparse matrix contents:\n";
      std::cout << "Rank " << mpi_rank << "\n";
      trilinos_sparse_matrix.print(std::cout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}
