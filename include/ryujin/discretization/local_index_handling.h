//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/linear_algebra/sparsity_pattern.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <map>
#include <vector>

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
     * Compute a Cuthill McKee ordering of the locally owned index range
     * from the locally owned block of the sparsity pattern @p sparsity.
     *
     * The function returns the ordering as a vector @p order of global
     * indices: order[k] is the locally owned index that is placed at the
     * k-th position of the locally owned index range. The functions
     * export_indices_first() and internal_range() reorder such a vector
     * further.
     *
     * @ingroup FiniteElement
     */
    inline std::vector<dealii::types::global_dof_index>
    Cuthill_McKee(const DynamicSparsityPattern &sparsity,
                  const dealii::IndexSet &locally_owned)
    {
      using namespace dealii;

      /* The locally owned index range has to be contiguous */
      Assert(locally_owned.is_contiguous() == true,
             dealii::ExcMessage(
                 "Need a contiguous set of locally owned indices."));

      const auto n_locally_owned = locally_owned.n_elements();
      if (n_locally_owned == 0)
        return {};

      /* Offset to translate from global to local index range */
      const auto offset = *locally_owned.begin();

      /* Copy the locally owned block in local numbering: */
      dealii::DynamicSparsityPattern local_sparsity(n_locally_owned,
                                                    n_locally_owned);
      std::vector<types::global_dof_index> columns;
      for (unsigned int i = 0; i < n_locally_owned; ++i) {
        columns.clear();
        for (const auto column : sparsity.columns(offset + i))
          if (column >= offset && column < offset + n_locally_owned)
            columns.push_back(column - offset);
        local_sparsity.add_entries(i, columns.begin(), columns.end(), true);
      }

      std::vector<types::global_dof_index> new_indices(n_locally_owned);
      SparsityTools::reorder_Cuthill_McKee(local_sparsity, new_indices);

      std::vector<types::global_dof_index> order(n_locally_owned);
      for (unsigned int i = 0; i < n_locally_owned; ++i)
        order[new_indices[i]] = offset + i;

      return order;
    }


    /**
     * Reorder all (strides of) locally internal indices that contain
     * export indices to the start of the index range.
     *
     * The function operates on an ordering @p order of the locally owned
     * index range (see Cuthill_McKee()) and moves all strides of @p
     * warp_size consecutive positions in [0, @p n_locally_internal) that
     * contain an element of @p export_indices to the front. The relative
     * order of strides is preserved. Returns the number of positions
     * occupied by these strides.
     *
     * @ingroup FiniteElement
     */
    inline unsigned int
    export_indices_first(std::vector<dealii::types::global_dof_index> &order,
                         const dealii::IndexSet &export_indices,
                         const unsigned int n_locally_internal,
                         const std::size_t warp_size)
    {
      Assert(n_locally_internal <= order.size(), dealii::ExcInternalError());
      Assert(n_locally_internal % warp_size == 0, dealii::ExcInternalError());

      const auto contains_export_index = [&](const unsigned int i) {
        for (unsigned int j = 0; j < warp_size; ++j)
          if (export_indices.is_element(order[i + j]))
            return true;
        return false;
      };

      std::vector<dealii::types::global_dof_index> new_order;
      new_order.reserve(order.size());

      /* First pass: all strides containing export indices: */
      for (unsigned int i = 0; i < n_locally_internal; i += warp_size)
        if (contains_export_index(i))
          new_order.insert(new_order.end(), &order[i], &order[i] + warp_size);

      const unsigned int n_export_indices = new_order.size();

      /* Second pass: append the rest: */
      for (unsigned int i = 0; i < n_locally_internal; i += warp_size)
        if (!contains_export_index(i))
          new_order.insert(new_order.end(), &order[i], &order[i] + warp_size);

      new_order.insert(
          new_order.end(), order.begin() + n_locally_internal, order.end());

      order = std::move(new_order);
      return n_export_indices;
    }


    /**
     * Reorder all strides of inconsistent locally internal indices to the
     * end of the locally internal index range.
     *
     * @ingroup FiniteElement
     */
    template <int structdim, int dim>
    unsigned int
    inconsistent_strides_last(dealii::DoFHandler<structdim, dim> &dof_handler,
                              const DynamicSparsityPattern &sparsity,
                              const unsigned int n_locally_internal,
                              const std::size_t warp_size)
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

      for (unsigned int i = 0; i < n_locally_internal; i += warp_size) {

        bool stride_is_consistent = true;
        const auto warp_row_length = sparsity.row_length(offset + i);
        for (unsigned int j = 0; j < warp_size; ++j) {
          if (warp_row_length != sparsity.row_length(offset + i + j)) {
            stride_is_consistent = false;
            break;
          }
        }

        if (stride_is_consistent) {
          for (unsigned int j = 0; j < warp_size; ++j) {
            new_order[i + j] = offset + n_consistent_range++;
          }
        } else {
          for (unsigned int j = 0; j < warp_size; ++j)
            new_order[i + j] = dealii::numbers::invalid_dof_index;
        }
      }

      /*
       * Second pass: append the rest:
       */

      unsigned int running_index = n_consistent_range;

      for (unsigned int i = 0; i < n_locally_internal; i += warp_size) {
        if (new_order[i] == dealii::numbers::invalid_dof_index) {
          for (unsigned int j = 0; j < warp_size; ++j) {
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

      Assert(n_consistent_range % warp_size == 0, dealii::ExcInternalError());
      Assert(n_consistent_range <= n_locally_internal,
             dealii::ExcInternalError());
      return n_consistent_range;
    }


    /**
     * Reorder indices:
     *
     * In order to traverse over multiple rows of a (to be constructed)
     * sparsity pattern simultaneously (using SIMD instructions, or a warp
     * of threads on the device) we reorder all locally owned degrees of
     * freedom to ensure that a local index range \f$[0,
     * \text{n_locally_internal_}) \subset [0,
     * \text{n_locally_owned})\f$ is available that groups dofs with same
     * stencil size in "warps" of @p warp_size consecutive indices.
     *
     * The function operates on an ordering @p order of the locally owned
     * index range (see Cuthill_McKee()). Returns the right boundary
     * n_internal of the internal index range.
     *
     * @ingroup FiniteElement
     */
    inline unsigned int
    internal_range(std::vector<dealii::types::global_dof_index> &order,
                   const DynamicSparsityPattern &sparsity,
                   const std::size_t warp_size)
    {
      using dof_type = dealii::types::global_dof_index;
      std::vector<dof_type> new_order;
      new_order.reserve(order.size());

      /*
       * Sort degrees of freedom into a map grouped by stencil size. Write
       * out dof indices into the new_order vector in warps of warp_size
       * consecutive indices with same stencil size.
       */

      std::map<unsigned int, std::vector<dof_type>> bins;

      for (const auto index : order) {
        const unsigned int row_length = sparsity.row_length(index);
        auto &bin = bins[row_length];
        bin.push_back(index);

        if (bin.size() == warp_size) {
          new_order.insert(new_order.end(), bin.begin(), bin.end());
          bins.erase(row_length);
        }
      }

      const unsigned int n_locally_internal = new_order.size();

      /* Write out the rest. */

      for (const auto &[row_length, bin] : bins)
        new_order.insert(new_order.end(), bin.begin(), bin.end());
      Assert(new_order.size() == order.size(), dealii::ExcInternalError());

      order = std::move(new_order);

      Assert(n_locally_internal % warp_size == 0, dealii::ExcInternalError());
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
     * Same as FiniteElement::has_support_on_face() but with the correct
     * answer for simplices: deal.II does not override the default
     * implementation for FE_SimplexP and FE_SimplexDGP, which always returns
     * true. A nodal shape function on a simplex vanishes on a face if and
     * only if its support point is not located on that face.
     *
     * @ingroup FiniteElement
     */
    template <int structdim, int dim>
    bool has_support_on_face(const dealii::FiniteElement<structdim, dim> &fe,
                             const unsigned int i,
                             const unsigned int f_index)
    {
      const auto reference_cell = fe.reference_cell();
      if (reference_cell.is_hyper_cube())
        return fe.has_support_on_face(i, f_index);

#if DEAL_II_VERSION_GTE(9, 8, 0)
      const auto vertex = reference_cell.face_vertex_location(f_index, 0);
      const auto normal = reference_cell.face_normal_vector(f_index);
#else
      const auto vertex =
          reference_cell.template face_vertex_location<structdim>(f_index, 0);
      const auto normal =
          reference_cell.template unit_normal_vectors<structdim>(f_index);
#endif

      const auto offset = fe.get_unit_support_points()[i] - vertex;
      return std::abs(offset * normal) < 1.e-10;
    }


    /**
     * Add the couplings of a @p cell to an extended sparsity pattern for
     * the discontinuous Galerkin formulation: all degrees of freedom of
     * the cell couple with each other, and degrees of freedom on a face
     * couple with the degrees of freedom of the neighbor on that face.
     * The function only adds to the rows of the degrees of freedom of the
     * cell.
     *
     * @ingroup FiniteElement
     */
    template <typename CellIterator>
    void add_cell_couplings_dg(const CellIterator &cell,
                               DynamicSparsityPattern &sparsity)
    {
      thread_local std::vector<dealii::types::global_dof_index> dof_indices;
      thread_local std::vector<dealii::types::global_dof_index>
          neighbor_dof_indices;
      thread_local std::vector<dealii::types::global_dof_index>
          coupling_indices;
      thread_local std::vector<dealii::types::global_dof_index>
          neighbor_coupling_indices;

      const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      for (const auto i : dof_indices)
        sparsity.add_entries(i, dof_indices.begin(), dof_indices.end());

      for (const auto f_index : cell->face_indices()) {
        const auto &face = cell->face(f_index);

        /* Skip faces without neighbors... */
        const bool has_neighbor =
            !face->at_boundary() || cell->has_periodic_neighbor(f_index);
        if (!has_neighbor)
          continue;

        /* Avoid artificial cells: */
        const auto neighbor_cell = cell->neighbor_or_periodic_neighbor(f_index);
        if (neighbor_cell->is_artificial())
          continue;

        const unsigned int neighbor_dofs_per_cell =
            neighbor_cell->get_fe().n_dofs_per_cell();
        neighbor_dof_indices.resize(neighbor_dofs_per_cell);
        neighbor_cell->get_dof_indices(neighbor_dof_indices);

        const unsigned int f_index_neighbor =
            cell->has_periodic_neighbor(f_index)
                ? cell->periodic_neighbor_of_periodic_neighbor(f_index)
                : cell->neighbor_of_neighbor(f_index);

        /*
         * Construct all couplings between current and neighbor cell with
         * DoFs located at the boundary:
         */

        coupling_indices.resize(0);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (has_support_on_face(cell->get_fe(), i, f_index))
            coupling_indices.push_back(dof_indices[i]);

        neighbor_coupling_indices.resize(0);
        for (unsigned int j = 0; j < neighbor_dofs_per_cell; ++j)
          if (has_support_on_face(neighbor_cell->get_fe(), j, f_index_neighbor))
            neighbor_coupling_indices.push_back(neighbor_dof_indices[j]);

        for (const auto i : coupling_indices)
          sparsity.add_entries(i,
                               neighbor_coupling_indices.begin(),
                               neighbor_coupling_indices.end());
      }
    }


  } // namespace DoFTools

} // namespace ryujin
