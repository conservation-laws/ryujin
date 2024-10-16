#include <sparse_matrix_simd.h>
#include <sparse_matrix_simd.template.h>

/*
 * A quick check that "send_targets" and "entries_to_be_sent" are set up
 * correctly. Note that the sparsity pattern we create is artificial.
 *
 * A consequence is that the "receive targets" are not set up correctly.
 * Some ranks expect to receive more data than what is actually sent. This
 * is due to the fact that we violate the symmetry assumption that a
 * nonzero m_ij/c_ij entry implies a corresponding nonzero m_ji/c_ji entry.
 */

namespace ryujin
{
  template <int simd_length>
  class Debug : public SparsityPatternSIMD<simd_length>
  {
  public:
    Debug(const unsigned int n_internal_dofs,
          const dealii::DynamicSparsityPattern &sparsity,
          const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
              &partitioner)
        : SparsityPatternSIMD<simd_length>(
              n_internal_dofs, sparsity, partitioner)
    {
    }

    void print()
    {
      std::stringstream ss;
      // ss << "Receive targets:\n";
      // for (const auto &[left, right] : this->receive_targets)
      //   ss << left << " : " << right << "\n";

      ss << "Send targets:\n";
      for (const auto &[left, right] : this->send_targets)
        ss << left << " : " << right << "\n";

      ss << "Entries to be sent:\n";
      for (const auto &[left, right] : this->send_targets)
        ss << left << " : " << right << "\n";

      std::cout << ss.str() << std::endl;
    }
  };
} // namespace ryujin


int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const auto mpi_rank =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  const auto n_mpi_processes =
      dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  AssertThrow(n_mpi_processes == 4, dealii::ExcMessage("set up for 4 ranks"));

  /* Set up locally owned and relevant index sets. */

  dealii::IndexSet locally_owned(16);
  dealii::IndexSet locally_relevant(16);

  switch (mpi_rank) {
  case 0:
    // import only
    locally_owned.add_range(0, 4);
    locally_relevant.add_range(0, 4);   // own
    locally_relevant.add_range(4, 6);   // from rank 1
    locally_relevant.add_range(8, 10);  // from rank 2
    locally_relevant.add_range(12, 14); // from rank 3
    break;
  case 1:
    locally_owned.add_range(4, 8);
    locally_relevant.add_range(4, 8);
    locally_relevant.add_range(8, 10);  // from rank 2
    locally_relevant.add_range(12, 14); // from rank 3
    break;
  case 2:
    locally_owned.add_range(8, 12);
    locally_relevant.add_range(8, 12);
    locally_relevant.add_range(4, 6);   // from rank 1
    locally_relevant.add_range(14, 16); // from rank 3
    break;
  case 3:
    // export only
    locally_owned.add_range(12, 16);
    locally_relevant.add_range(12, 16);
    break;
  default:
    __builtin_unreachable();
  }

  const auto partitioner =
      std::make_shared<dealii::Utilities::MPI::Partitioner>(
          locally_owned, locally_relevant, MPI_COMM_WORLD);

  /* Set up sparsity pattern: */

  dealii::DynamicSparsityPattern dsp(16, 16, locally_relevant);

  switch (mpi_rank) {
  case 0:
    dsp.add(0, 0);
    dsp.add(1, 1);
    dsp.add(2, 2);
    dsp.add(3, 3);
    dsp.add(8, 3);
    dsp.add(9, 3);
    dsp.add(3, 8);
    dsp.add(3, 9);
    dsp.add(3, 12);
    break;
  case 1:
    dsp.add(4, 4);
    dsp.add(5, 5);
    dsp.add(6, 6);
    dsp.add(7, 7);
    dsp.add(5, 8);
    dsp.add(5, 9);
    dsp.add(5, 12);
    dsp.add(5, 13);
    dsp.add(8, 5);
    dsp.add(9, 5);
    dsp.add(12, 5);
    dsp.add(13, 5);
    break;
  case 2:
    dsp.add(8, 8);
    dsp.add(9, 9);
    dsp.add(10, 10);
    dsp.add(11, 11);
    dsp.add(10, 4);
    dsp.add(10, 5);
    dsp.add(4, 10);
    dsp.add(5, 10);
    dsp.add(11, 14);
    dsp.add(11, 15);
    dsp.add(14, 11);
    dsp.add(15, 11);
    break;
  case 3:
    dsp.add(12, 12);
    dsp.add(13, 13);
    dsp.add(14, 14);
    dsp.add(15, 15);
    break;
  default:
    __builtin_unreachable();
  }
  dsp.compress();

  using VA = dealii::VectorizedArray<double>;
  constexpr auto simd_width = VA::size();
  ryujin::Debug<simd_width> sparsity_pattern_simd(
      /* vectorized internal range */ 0, dsp, partitioner);

  for (unsigned int i = 0; i < n_mpi_processes; ++i) {
    if (i == mpi_rank)
      sparsity_pattern_simd.print();
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
