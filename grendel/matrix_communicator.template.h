#ifndef MATRIX_COMMUNICATOR_TEMPLATE_H
#define MATRIX_COMMUNICATOR_TEMPLATE_H

#include "matrix_communicator.h"

#include <deal.II/lac/sparsity_tools.h>

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
    const auto &dof_handler = offline_data_->dof_handler();
    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_extended = offline_data_->locally_extended();
    const auto &sparsity = offline_data_->sparsity_pattern();

    Assert(sparsity.is_compressed(), dealii::ExcInternalError());

    /*
     * This is an evil hack:
     *
     * The following creates a temporary extended sparsity pattern with
     * global indices and full sparsity rows. This sparsity pattern is only
     * used to create a (globally stable) ordering of all degrees of
     * freedom of the stencil belonging to a degree of freedom. We use this
     * information to synchronize the "ghost layer" of local SparseMatrix
     * objects over all MPI ranks.
     */

    {
      const auto n_dofs = locally_owned.size();
      const auto dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

      DynamicSparsityPattern extended_sparsity(
          n_dofs, n_dofs, locally_extended);

      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
      for (auto cell : dof_handler.active_cell_iterators()) {
        /* iterate over locally owned cells and the ghost layer */
        if (cell->is_artificial())
          continue;

        cell->get_dof_indices(dof_indices);
        affine_constraints.add_entries_local_to_global(
            dof_indices, extended_sparsity, false);
      }

      SparsityTools::gather_sparsity_pattern(
          extended_sparsity,
          dof_handler.compute_locally_owned_dofs_per_processor(),
          mpi_communicator_,
          locally_extended);

      extended_sparsity.compress();

      indices_.reinit(sparsity);

      for (const auto i_global : locally_extended) {
        const auto i = locally_extended.index_within_set(i_global);

        for (auto jt = sparsity.begin(i); jt != sparsity.end(i); ++jt) {
          const auto j = jt->column();
          const auto j_global = locally_extended.nth_index_in_set(j);

          auto ejt = extended_sparsity.begin(i_global);
          unsigned int index = 0;
          for (; ejt->column() != j_global; ++ejt)
            index++;
          set_entry(indices_, jt, index);
        }
      }
    } /* end of hack */

    unsigned int n = sparsity.max_entries_per_row();
    n = Utilities::MPI::max(n, mpi_communicator_);

    matrix_temp_.resize(
        n,
        dealii::LinearAlgebra::distributed::Vector<double>(
            locally_owned, locally_extended, mpi_communicator_));
  }


  template <int dim>
  void MatrixCommunicator<dim>::synchronize()
  {
    const auto &locally_owned = offline_data_->locally_owned();
    const auto &locally_extended = offline_data_->locally_extended();
    const auto indices =
        boost::irange<unsigned int>(0, locally_extended.n_elements());
    const auto &sparsity = offline_data_->sparsity_pattern();

    {
      const auto on_subranges = [&](auto i1, const auto i2) {
        /* Translate the local index into a index set iterator:: */
        auto it = locally_extended.at(locally_extended.nth_index_in_set(*i1));
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
        auto it = locally_extended.at(locally_extended.nth_index_in_set(*i1));
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
