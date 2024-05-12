//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <deal.II/base/partitioner.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

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
     * Reorder all (strides of) locally internal indices that contain
     * export indices to the start of the index range.
     *
     * This renumbering requires MPI communication in order to determine
     * the set of export indices.
     *
     * @ingroup FiniteElement
     */
    template <int dim>
    unsigned int export_indices_first(dealii::DoFHandler<dim> &dof_handler,
                                      const MPI_Comm &mpi_communicator,
                                      const unsigned int n_locally_internal,
                                      const std::size_t group_size)
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

      /*
       * First pass: reorder all strides containing export indices and mark
       * all other indices with numbers::invalid_dof_index:
       */

      unsigned int n_export_indices = 0;

      Assert(n_locally_internal <= n_locally_owned, dealii::ExcInternalError());

      for (unsigned int i = 0; i < n_locally_internal; i += group_size) {
        bool export_index_present = false;
        for (unsigned int j = 0; j < group_size; ++j) {
          if (export_indices.is_element(i + j)) {
            export_index_present = true;
            break;
          }
        }

        if (export_index_present) {
          Assert(n_export_indices % group_size == 0,
                 dealii::ExcInternalError());
          for (unsigned int j = 0; j < group_size; ++j) {
            new_order[i + j] = offset + n_export_indices++;
          }
        } else {
          for (unsigned int j = 0; j < group_size; ++j)
            new_order[i + j] = dealii::numbers::invalid_dof_index;
        }
      }

#if DEBUG
      unsigned int n_other = 0;
      for (unsigned int i = n_locally_internal; i < n_locally_owned; ++i)
        if (export_indices.is_element(i))
          n_other++;

      Assert(n_other + n_export_indices >= export_indices.n_elements(),
             dealii::ExcInternalError());
#endif

      unsigned int running_index = n_export_indices;

      /*
       * Second pass: append the rest:
       */

      for (unsigned int i = 0; i < n_locally_internal; i += group_size) {
        if (new_order[i] == dealii::numbers::invalid_dof_index) {
          for (unsigned int j = 0; j < group_size; ++j) {
            Assert(new_order[i + j] == dealii::numbers::invalid_dof_index,
                   dealii::ExcInternalError());
            new_order[i + j] = offset + running_index++;
          }
        }
      }

      Assert(running_index == n_locally_internal, dealii::ExcInternalError());

      for (unsigned int i = n_locally_internal; i < n_locally_owned; i++) {
        new_order[i] = offset + running_index++;
      }

      Assert(running_index == n_locally_owned, dealii::ExcInternalError());

      dof_handler.renumber_dofs(new_order);

      Assert(n_export_indices % group_size == 0, dealii::ExcInternalError());
      Assert(n_export_indices <= n_locally_internal,
             dealii::ExcInternalError());
      return n_export_indices;
    }


    /**
     * Reorder all strides of inconsistent locally internal indices to the
     * end of the locally internal index range.
     *
     * @ingroup FiniteElement
     */
    template <int dim>
    unsigned int
    inconsistent_strides_last(dealii::DoFHandler<dim> &dof_handler,
                              const dealii::DynamicSparsityPattern &sparsity,
                              const unsigned int n_locally_internal,
                              const std::size_t group_size)
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

      std::vector<dealii::types::global_dof_index> new_order(n_locally_owned);

      /*
       * First pass: keep all strides with consistent row length at the
       * beginning of the locally internal index range and mark all other
       * indices with numbers::invalid_dof_index:
       */

      unsigned int n_consistent_range = 0;

      Assert(n_locally_internal <= n_locally_owned, dealii::ExcInternalError());

      for (unsigned int i = 0; i < n_locally_internal; i += group_size) {

        bool stride_is_consistent = true;
        const auto group_row_length = sparsity.row_length(offset + i);
        for (unsigned int j = 0; j < group_size; ++j) {
          if (group_row_length != sparsity.row_length(offset + i + j)) {
            stride_is_consistent = false;
            break;
          }
        }

        if (stride_is_consistent) {
          for (unsigned int j = 0; j < group_size; ++j) {
            new_order[i + j] = offset + n_consistent_range++;
          }
        } else {
          for (unsigned int j = 0; j < group_size; ++j)
            new_order[i + j] = dealii::numbers::invalid_dof_index;
        }
      }

      /*
       * Second pass: append the rest:
       */

      unsigned int running_index = n_consistent_range;

      for (unsigned int i = 0; i < n_locally_internal; i += group_size) {
        if (new_order[i] == dealii::numbers::invalid_dof_index) {
          for (unsigned int j = 0; j < group_size; ++j) {
            Assert(new_order[i + j] == dealii::numbers::invalid_dof_index,
                   dealii::ExcInternalError());
            new_order[i + j] = offset + running_index++;
          }
        }
      }

      Assert(running_index == n_locally_internal, dealii::ExcInternalError());

      for (unsigned int i = n_locally_internal; i < n_locally_owned; i++) {
        new_order[i] = offset + running_index++;
      }

      Assert(running_index == n_locally_owned, dealii::ExcInternalError());

      dof_handler.renumber_dofs(new_order);

      Assert(n_consistent_range % group_size == 0, dealii::ExcInternalError());
      Assert(n_consistent_range <= n_locally_internal,
             dealii::ExcInternalError());
      return n_consistent_range;
    }


    /**
     * Reorder indices:
     *
     * In order to traverse over multiple rows of a (to be constructed)
     * sparsity pattern simultaneously using SIMD instructions we reorder
     * all locally owned degrees of freedom to ensure that a local index
     * range \f$[0, \text{n_locally_internal_}) \subset [0,
     * \text{n_locally_owned})\f$ is available that groups dofs with same
     * stencil size in groups of multiples of @p group_size
     *
     * Returns the right boundary n_internal of the internal index range.
     *
     * @ingroup FiniteElement
     */
    template <int dim>
    unsigned int internal_range(dealii::DoFHandler<dim> &dof_handler,
                                const dealii::DynamicSparsityPattern &sparsity,
                                const std::size_t group_size)
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
      dof_type current_index = offset;

      /*
       * Sort degrees of freedom into a map grouped by stencil size. Write
       * out dof indices into the new_order vector in groups of group_size
       * and with same stencil size.
       */

      std::map<unsigned int, std::set<dof_type>> bins;

      for (unsigned int i = 0; i < n_locally_owned; ++i) {
        const dof_type index = i;
        const unsigned int row_length = sparsity.row_length(offset + index);
        bins[row_length].insert(index);

        if (bins[row_length].size() == group_size) {
          for (const auto &index : bins[row_length])
            new_order[index] = current_index++;
          bins.erase(row_length);
        }
      }

      unsigned int n_locally_internal = current_index - offset;

      /* Write out the rest. */

      for (const auto &entries : bins) {
        Assert(entries.second.size() > 0, ExcInternalError());
        for (const auto &index : entries.second)
          new_order[index] = current_index++;
      }
      Assert(current_index == offset + n_locally_owned, ExcInternalError());

      dof_handler.renumber_dofs(new_order);

      Assert(n_locally_internal % group_size == 0, ExcInternalError());
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
     * Given a @p dof_handler, and constraints @p affine_constraints this
     * function creates an extended sparsity pattern that also includes
     * locally relevant to locally relevant couplings.
     *
     * @ingroup FiniteElement
     */
    template <int dim, typename Number, typename SPARSITY>
    void make_extended_sparsity_pattern(
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

        cell->get_dof_indices(dof_indices);
        affine_constraints.add_entries_local_to_global(
            dof_indices, dsp, keep_constrained);
      }
    }


    /**
     * Given a @p dof_handler, and constraints @p affine_constraints this
     * function creates an extended sparsity pattern for the discontinuous
     * Galerkin formulation that also includes locally relevant to locally
     * relevant couplings.
     *
     * @ingroup FiniteElement
     */
    template <int dim, typename Number, typename SPARSITY>
    void make_extended_sparsity_pattern_dg(
        const dealii::DoFHandler<dim> &dof_handler,
        SPARSITY &dsp,
        const dealii::AffineConstraints<Number> &affine_constraints,
        bool keep_constrained)
    {
      Assert(affine_constraints.n_constraints() == 0,
             dealii::ExcMessage("I don't think constraints make sense for dG"));

      const auto &fe = dof_handler.get_fe();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
      std::vector<dealii::types::global_dof_index> neighbor_dof_indices(
          dofs_per_cell);

      /*
       * We collect all coupling dof indices on a face and store the result
       * in a vector.
       */
      std::vector<dealii::types::global_dof_index> coupling_indices;
      std::vector<dealii::types::global_dof_index> neighbor_coupling_indices;

      /* we iterate over locally owned cells and the ghost layer */
      for (auto cell : dof_handler.active_cell_iterators()) {
        if (cell->is_artificial())
          continue;

        cell->get_dof_indices(dof_indices);

        affine_constraints.add_entries_local_to_global(
            dof_indices, dsp, keep_constrained);

        for (const auto f_index : cell->face_indices()) {
          const auto &face = cell->face(f_index);

          /* Skip faces without neighbors... */
          const bool has_neighbor =
              !face->at_boundary() || cell->has_periodic_neighbor(f_index);
          if (!has_neighbor)
            continue;

          /* Avoid artificial cells: */
          const auto neighbor_cell =
              cell->neighbor_or_periodic_neighbor(f_index);
          if (neighbor_cell->is_artificial())
            continue;

          const unsigned int f_index_neighbor =
              cell->has_periodic_neighbor(f_index)
                  ? cell->periodic_neighbor_of_periodic_neighbor(f_index)
                  : cell->neighbor_of_neighbor(f_index);

          neighbor_cell->get_dof_indices(neighbor_dof_indices);

          /*
           * Construct all couplings between current and neighbor cell with
           * DoFs located at the boundary:
           */

          coupling_indices.resize(0);
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            if (fe.has_support_on_face(i, f_index))
              coupling_indices.push_back(dof_indices[i]);

          neighbor_coupling_indices.resize(0);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            if (fe.has_support_on_face(j, f_index_neighbor))
              neighbor_coupling_indices.push_back(neighbor_dof_indices[j]);

          affine_constraints.add_entries_local_to_global(
              coupling_indices,
              neighbor_coupling_indices,
              dsp,
              keep_constrained);
        }
      }
    }


  } // namespace DoFTools

} // namespace ryujin
