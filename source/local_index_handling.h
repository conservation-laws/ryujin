//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef LOCAL_INDEX_HANDLING_H
#define LOCAL_INDEX_HANDLING_H

#include <deal.II/base/partitioner.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

namespace ryujin
{
  /**
   * Given a @p partitioner the function converts an AffineConstraints
   * object @p affine_constraints from the usual global numbering used in
   * deal.II to an MPI-rank local numbering \f$[0,n]\f$, where \f$n\f$ is
   * the number of locally relevant degrees of freedom.
   *
   * @ingroup FiniteElement
   */
  template <typename Number>
  void transform_to_local_range(
      const dealii::Utilities::MPI::Partitioner &partitioner,
      dealii::AffineConstraints<Number> &affine_constraints)
  {
    affine_constraints.close();

    dealii::AffineConstraints<Number> temporary;

    for (auto line : affine_constraints.get_lines()) {
      /* translate into local index ranges: */
      line.index = partitioner.global_to_local(line.index);
      std::transform(line.entries.begin(),
                     line.entries.end(),
                     line.entries.begin(),
                     [&](auto entry) {
                       return std::make_pair(
                           partitioner.global_to_local(entry.first),
                           entry.second);
                     });

      temporary.add_line(line.index);
      temporary.add_entries(line.index, line.entries);
      temporary.set_inhomogeneity(line.index, line.inhomogeneity);
    }

    temporary.close();

    affine_constraints = std::move(temporary);
  }


  /**
   * Given a @p partitioner the function translates each element of a given
   * vector @p vector from the usual global numbering used in deal.II to an
   * MPI-rank local numbering \f$[0,n]\f$, where \f$n\f$ is the number of
   * locally relevant degrees of freedom.
   *
   * @ingroup FiniteElement
   */
  template <typename VECTOR>
  void transform_to_local_range(
      const dealii::Utilities::MPI::Partitioner &partitioner, VECTOR &vector)
  {
    std::transform(
        vector.begin(), vector.end(), vector.begin(), [&](auto index) {
          return partitioner.global_to_local(index);
        });
  }


  /**
   * The DoFRenumbering namespace contains a number of custom dof
   * renumbering functions.
   *
   * @ingroup FiniteElement
   */
  namespace DoFRenumbering
  {
    /**
     * Import the Cuthill McKee reordering from deal.II into the current
     * namespace.
     */
    using dealii::DoFRenumbering::Cuthill_McKee;

    /**
     * Reorder all export indices in the locally owned index range to the
     * start of the index range.
     *
     * This renumbering requires MPI communication in order to determine
     * the set of export indices.
     *
     * @ingroup FiniteElement
     */
    template <int dim>
    unsigned int export_indices_first(dealii::DoFHandler<dim> &dof_handler,
                                      const MPI_Comm &mpi_communicator)
    {
      using namespace dealii;

      const IndexSet &locally_owned = dof_handler.locally_owned_dofs();
      const auto n_locally_owned = locally_owned.n_elements();

      /* The locally owned index range has to be contiguous */
      Assert(locally_owned.is_contiguous() == true,
             dealii::ExcMessage(
                 "Need a contiguous set of locally owned indices."));

      /* Offset to translate from global to local index range */
      const auto offset = n_locally_owned != 0 ? *locally_owned.begin() : 0;

      IndexSet locally_relevant;
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

      /* Create a temporary MPI partitioner: */

      Utilities::MPI::Partitioner partitioner(
          locally_owned, locally_relevant, mpi_communicator);

      IndexSet export_indices(n_locally_owned);
      for (const auto &it : partitioner.import_indices()) {
        export_indices.add_range(it.first, it.second);
      }

      std::vector<dealii::types::global_dof_index> new_order(n_locally_owned);

      const unsigned int n_export_indices = export_indices.n_elements();
      unsigned int index_import = 0;
      unsigned int index_rest = n_export_indices;
      for (unsigned int i = 0; i < n_locally_owned; ++i) {
        if (export_indices.is_element(i)) {
          Assert(index_import < n_export_indices, dealii::ExcInternalError());
          new_order[i] = offset + index_import++;
        } else {
          Assert(index_rest < n_locally_owned, dealii::ExcInternalError());
          new_order[i] = offset + index_rest++;
        }
      }
      Assert(index_import == n_export_indices, dealii::ExcInternalError());
      Assert(index_rest == n_locally_owned, dealii::ExcInternalError());

      dof_handler.renumber_dofs(new_order);

      return n_export_indices;
    }


    /**
     * Reorder indices:
     *
     * In order to traverse over multiple rows of a (to be constructed)
     * sparsity pattern simultaneously using SIMD instructions we reorder
     * all locally owned degrees of freedom to ensure that a local index
     * range \f$[0, \text{n_locally_internal_}) \subset [0,
     * \text{n_locally_owned})\f$ is available that
     *  - contains no boundary dof
     *  - contains no foreign degree of freedom
     *  - has "standard" connectivity, i.e. 2, 8, or 26 neighboring DoFs
     *    (in 1, 2, 3D).
     *
     * Returns the right boundary n_internal of the internal index range.
     *
     * @ingroup FiniteElement
     */
    template <int dim>
    unsigned int internal_range(dealii::DoFHandler<dim> &dof_handler)
    {
      using namespace dealii;

      const auto &locally_owned = dof_handler.locally_owned_dofs();
      const auto n_locally_owned = locally_owned.n_elements();

      /* The locally owned index range has to be contiguous */

      Assert(locally_owned.is_contiguous() == true,
             dealii::ExcMessage(
                 "Need a contiguous set of locally owned indices."));

      /* Offset to translate from global to local index range */
      const auto offset = n_locally_owned != 0 ? *locally_owned.begin() : 0;

      using dof_type = dealii::types::global_dof_index;
      std::vector<dof_type> new_order(n_locally_owned);

      {
        /*
         * Set up a temporary sparsity pattern to determine connectivity:
         * (We do this with global numbering)
         */

        IndexSet locally_relevant;
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

        AffineConstraints<double> affine_constraints;
        affine_constraints.reinit(locally_relevant);
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                affine_constraints);

        DynamicSparsityPattern dsp(locally_relevant);
        DoFTools::make_sparsity_pattern(
            dof_handler, dsp, affine_constraints, false);

        /* Mark all non-standard degrees of freedom: */

        constexpr unsigned int standard_connectivity =
            dim == 1 ? 3 : (dim == 2 ? 9 : 27);

        for (unsigned int i = 0; i < n_locally_owned; ++i)
          if (dsp.row_length(offset + i) != standard_connectivity)
            new_order[i] = dealii::numbers::invalid_dof_index;

        /* Explicitly poison boundary degrees of freedom: */

        const unsigned int dofs_per_face = dof_handler.get_fe().dofs_per_face;
        std::vector<dof_type> local_face_dof_indices(dofs_per_face);

        for (auto &cell : dof_handler.active_cell_iterators()) {
          if (!cell->at_boundary())
            continue;
          for (auto f : dealii::GeometryInfo<dim>::face_indices()) {
            const auto face = cell->face(f);
            if (!face->at_boundary())
              continue;
            face->get_dof_indices(local_face_dof_indices);
            for (unsigned int j = 0; j < dofs_per_face; ++j) {
              const auto &index = local_face_dof_indices[j];
              if (!locally_owned.is_element(index))
                continue;
              Assert(index >= offset && index - offset < n_locally_owned,
                     dealii::ExcInternalError());
              new_order[index - offset] = dealii::numbers::invalid_dof_index;
            }
          }
        }
      }

      /* Second pass: Create renumbering. */

      dof_type index = offset;

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


  /**
   * The DoFTools namespace contains a number of custom dof tools
   * functions.
   *
   * @ingroup FiniteElement
   */
  namespace DoFTools
  {
    /** Import a function from deal.II into the current namespace. */
    using dealii::DoFTools::extract_locally_relevant_dofs;

    /** Import a function from deal.II into the current namespace. */
    using dealii::DoFTools::make_hanging_node_constraints;

    /** Import a function from deal.II into the current namespace. */
    using dealii::DoFTools::make_periodicity_constraints;

    /** Import a function from deal.II into the current namespace. */
    using dealii::DoFTools::make_sparsity_pattern;

    /**
     * Given an MPI @p partitioner, @p dof_handler, and constraints @p
     * affine_constraints this function creates a sparsity pattern that
     * uses MPI-rank local numbering of indices (in the interval
     * \f$[0,n]\f$, where \f$n\f$ is the number of locally relevant degrees
     * of freedom) instead of the usual global numberinf used in deal.II.
     *
     * @ingroup FiniteElement
     */
    template <int dim, typename Number, typename SPARSITY>
    void make_local_sparsity_pattern(
        const dealii::Utilities::MPI::Partitioner &partitioner,
        const dealii::DoFHandler<dim> &dof_handler,
        SPARSITY &dsp,
        const dealii::AffineConstraints<Number> &affine_constraints,
        bool keep_constrained)
    {
      const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
      std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);

      for (auto cell : dof_handler.active_cell_iterators()) {
        /* iterate over locally owned cells and the ghost layer */
        if (cell->is_artificial())
          continue;

        /* translate into local index ranges: */
        cell->get_dof_indices(dof_indices);
        transform_to_local_range(partitioner, dof_indices);

        affine_constraints.add_entries_local_to_global(
            dof_indices, dsp, keep_constrained);
      }
    }
  } // namespace DoFTools

} // namespace ryujin

#endif /* LOCAL_INDEX_HANDLING_H */
