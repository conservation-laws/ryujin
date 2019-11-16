#ifndef LOCAL_INDEX_HANDLING_H
#define LOCAL_INDEX_HANDLING_H

#include <deal.II/dofs/dof_handler.h>

namespace grendel
{
  namespace DoFRenumbering
  {
    /**
     * Reorder indices:
     *
     * In order to traverse over multiple rows of a (to be constructed)
     * sparsity pattern simultaneously using SIMD instructions we reorder
     * all locally owned degrees of freedom to ensure that a local index
     * range [0, n_locally_internal_) \subset [0, n_locally_owned) is
     * available that
     *
     *  - contains no boundary dof
     *
     *  - contains no foreign degree of freedom
     *
     *  - has "standard" connectivity, i.e. 2, 8, or 26 neighboring DoFs
     *    (in 1, 2, 3D).
     *
     *  Returns the right boundary n_internal of the internal index range.
     */
    template <int dim>
    unsigned int internal_range(dealii::DoFHandler<dim> &dof_handler)
    {
      const auto &locally_owned = dof_handler.locally_owned_dofs();
      const auto n_locally_owned = locally_owned.n_elements();

      /* The locally owned index range has to be contiguous */

      Assert(locally_owned.is_contiguous() == true,
             dealii::ExcMessage(
                 "Need a contiguous set of locally owned indices."));

      /* Offset to translate from global to local index range */
      const auto offset = n_locally_owned != 0 ? *locally_owned.begin() : 0;

      const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
      std::vector<dealii::types::global_dof_index> local_dof_indices(
          dofs_per_cell);

      /*
       * First pass: Accumulate how many cells are associated with a
       * given degree of freedom and mark all degrees of freedom shared
       * with a different number of cells than 2, 4, or 8 with
       * numbers::invalid_dof_index:
       */

      std::vector<dealii::types::global_dof_index> new_order(n_locally_owned);

      for (auto cell : dof_handler.active_cell_iterators()) {
        if (cell->is_artificial())
          continue;

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          const auto &index = local_dof_indices[j];
          if (!locally_owned.is_element(index))
            continue;

          Assert(index - offset < n_locally_owned, dealii::ExcInternalError());
          new_order[index - offset] += 1;
        }
      }

      constexpr dealii::types::global_dof_index standard_number_of_neighbors =
          dim == 1 ? 2 : (dim == 2 ? 4 : 8);

      for (auto &it : new_order) {
        if (it == standard_number_of_neighbors)
          it = 0;
        else
          it = dealii::numbers::invalid_dof_index;
      }

      /* Second pass: Create renumbering. */

      dealii::types::global_dof_index index = offset;

      unsigned int n_locally_internal = 0;
      for (auto &it : new_order)
        if (it != dealii::numbers::invalid_dof_index) {
          it = index++;
          n_locally_internal++;
        }

      for (auto &it : new_order)
        if (it == dealii::numbers::invalid_dof_index)
          it = index++;

      dof_handler.renumber_dofs(new_order);

      return n_locally_internal;
    }
  } // namespace DoFRenumbering

} // namespace grendel

#endif /* LOCAL_INDEX_HANDLING_H */
