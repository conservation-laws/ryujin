#ifndef MATRIX_COMMUNICATOR_TEMPLATE_H
#define MATRIX_COMMUNICATOR_TEMPLATE_H

#include "matrix_communicator.h"

#include <boost/range/irange.hpp>

namespace grendel
{
  using namespace dealii;


  template <int dim>
  MatrixCommunicator<dim>::MatrixCommunicator(
      const MPI_Comm &mpi_communicator,
      dealii::TimerOutput &computing_timer,
      const grendel::OfflineData<dim> &offline_data,
      dealii::SparseMatrix<double> &matrix)
      : mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , matrix_(matrix)
  {
  }


  template <int dim>
  void MatrixCommunicator<dim>::prepare()
  {
    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto &sparsity = offline_data_->sparsity_pattern();
    const auto &extended_sparsity = offline_data_->extended_sparsity_pattern();

    Assert(sparsity.is_compressed(), dealii::ExcInternalError());
    Assert(extended_sparsity.is_compressed(), dealii::ExcInternalError());

    indices_.reinit(sparsity);

    for (auto i : locally_relevant) {
      auto ejt = extended_sparsity.begin(i);
      unsigned int local_index = 0;

      for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
        for (; ejt->column() != jt->column(); ++ejt)
          local_index++;

        set_entry(indices_, jt, local_index);
      }
    }

    unsigned int n = sparsity.max_entries_per_row();
    n = Utilities::MPI::max(n, mpi_communicator_);

    matrix_temp_.resize(
        n,
        dealii::LinearAlgebra::distributed::Vector<double>(
            locally_owned, locally_relevant, mpi_communicator_));
  }


  template <int dim>
  void MatrixCommunicator<dim>::synchronize()
  {
    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_relevant = offline_data_->locally_relevant();
    const auto indices =
        boost::irange<unsigned int>(0, locally_relevant.n_elements());
    const auto &sparsity = offline_data_->sparsity_pattern();

    {
      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {
          const auto i = *it;

          if (!locally_owned.is_element(i))
            continue;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto jt_local_index = get_entry(indices_, jt);
            matrix_temp_[jt_local_index][i] = get_entry(matrix_, jt);
          }
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
    }

    for (auto &it : matrix_temp_)
      it.update_ghost_values();

    {
      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Translate the local index into a index set iterator:: */
        auto it = locally_relevant.at(locally_relevant.nth_index_in_set(*i1));
        for (; i1 < i2; ++i1, ++it) {
          const auto i = *it;

          if (locally_owned.is_element(i))
            continue;

          for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
            const auto jt_local_index = get_entry(indices_, jt);
            set_entry(matrix_, jt, matrix_temp_[jt_local_index][i]);
          }
        }
      };

      parallel::apply_to_subranges(
          indices.begin(), indices.end(), on_subranges, 4096);
    }
  }

} /* namespace grendel */

#endif /* MATRIX_COMMUNICATOR_TEMPLATE_H */
