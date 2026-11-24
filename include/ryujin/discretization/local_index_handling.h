//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/linear_algebra/sparsity_pattern.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/affine_constraints.h>

#include <algorithm>
#include <map>
#include <numeric>
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
     * The algorithm is the one of SparsityTools::reorder_Cuthill_McKee(),
     * applied independently and in parallel to blocks of consecutive
     * rows. The block size is fixed, so that the result does not depend
     * on the number of threads.
     *
     * The function returns the ordering as a vector @p order of global
     * indices: order[k] is the locally owned index that is placed at the
     * k-th position of the locally owned index range. The functions
     * strides_first() and internal_range() reorder such a vector further,
     * renumber_dofs() applies it.
     *
     * @ingroup FiniteElement
     */
    inline std::vector<dealii::types::global_dof_index>
    Cuthill_McKee(const DynamicSparsityPattern &sparsity,
                  const dealii::IndexSet &locally_owned)
    {
      /* The locally owned index range has to be contiguous */
      Assert(locally_owned.is_contiguous() == true,
             dealii::ExcMessage(
                 "Need a contiguous set of locally owned indices."));

      const unsigned int n = locally_owned.n_elements();
      if (n == 0)
        return {};
      const auto offset = *locally_owned.begin();

      constexpr unsigned int block_size = 1u << 18;

      std::vector<dealii::types::global_dof_index> order(n);
      std::vector<unsigned int> degree(n);
      std::vector<char> visited(n, 0);

      const auto number_block = [&](const unsigned int left,
                                    const unsigned int right) {
        /* The neighbors of a row are its columns within the block: */
        const auto for_each_neighbor = [&](const unsigned int i, auto f) {
          for (const auto column : sparsity.columns(offset + i))
            if (column >= offset + left && column < offset + right)
              f(column - offset);
        };

        for (unsigned int i = left; i < right; ++i)
          for_each_neighbor(i, [&](unsigned int) { ++degree[i]; });

        const auto by_degree = [&](const unsigned int a, const unsigned int b) {
          return std::pair(degree[a], a) < std::pair(degree[b], b);
        };

        /*
         * Number the graph front by front, every front by increasing
         * degree. A new component starts at the unnumbered row of smallest
         * degree, for which we sort all rows by degree (and index):
         */

        const auto max_degree =
            *std::max_element(degree.begin() + left, degree.begin() + right);
        std::vector<unsigned int> count(max_degree + 2);
        for (unsigned int i = left; i < right; ++i)
          ++count[degree[i] + 1];
        std::partial_sum(count.begin(), count.end(), count.begin());
        std::vector<unsigned int> candidates(right - left);
        for (unsigned int i = left; i < right; ++i)
          candidates[count[degree[i]]++] = i;
        auto candidate = candidates.begin();

        auto position = order.begin() + left;
        std::vector<unsigned int> front, next_front;
        while (position != order.begin() + right) {
          next_front.clear();
          for (const auto i : front)
            for_each_neighbor(i, [&](const unsigned int j) {
              if (!visited[j]) {
                visited[j] = 1;
                next_front.push_back(j);
              }
            });

          if (next_front.empty()) {
            while (visited[*candidate])
              ++candidate;
            visited[*candidate] = 1;
            next_front.push_back(*candidate);
          }

          std::sort(next_front.begin(), next_front.end(), by_degree);
          for (const auto i : next_front)
            *position++ = offset + i;
          front.swap(next_front);
        }
      };

      dealii::parallel::apply_to_subranges(
          0u,
          (n + block_size - 1) / block_size,
          [&](const unsigned int begin, const unsigned int end) {
            for (unsigned int block = begin; block < end; ++block)
              number_block(block * block_size,
                           std::min(n, (block + 1) * block_size));
          },
          1);

      return order;
    }


    /**
     * Move all strides of @p warp_size consecutive positions in [0, @p
     * n_strided) of an ordering @p order (see Cuthill_McKee()) for which
     * @p predicate(first, last) returns true to the front. The relative
     * order of strides is preserved. Returns the number of positions
     * occupied by these strides.
     *
     * @ingroup FiniteElement
     */
    template <typename Predicate>
    unsigned int
    strides_first(std::vector<dealii::types::global_dof_index> &order,
                  const unsigned int n_strided,
                  const std::size_t warp_size,
                  const Predicate &predicate)
    {
      Assert(n_strided <= order.size(), dealii::ExcInternalError());
      Assert(n_strided % warp_size == 0, dealii::ExcInternalError());

      std::vector<dealii::types::global_dof_index> new_order;
      new_order.reserve(order.size());

      const auto append_strides = [&](const bool selected) {
        for (unsigned int i = 0; i < n_strided; i += warp_size) {
          const auto stride = order.cbegin() + i;
          if (predicate(stride, stride + warp_size) == selected)
            new_order.insert(new_order.end(), stride, stride + warp_size);
        }
      };

      append_strides(true);
      const unsigned int n_selected = new_order.size();
      append_strides(false);
      new_order.insert(new_order.end(), order.begin() + n_strided, order.end());

      order = std::move(new_order);
      return n_selected;
    }


    /**
     * Apply an ordering @p order (see Cuthill_McKee()) of the locally
     * owned index range to the @p dof_handler.
     *
     * @ingroup FiniteElement
     */
    template <int structdim, int dim>
    void
    renumber_dofs(dealii::DoFHandler<structdim, dim> &dof_handler,
                  const std::vector<dealii::types::global_dof_index> &order)
    {
      const auto &locally_owned = dof_handler.locally_owned_dofs();
      const auto offset =
          locally_owned.n_elements() != 0 ? *locally_owned.begin() : 0;

      std::vector<dealii::types::global_dof_index> new_order(order.size());
      for (unsigned int k = 0; k < order.size(); ++k)
        new_order[order[k] - offset] = offset + k;
      dof_handler.renumber_dofs(new_order);
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
     * Add the couplings of a @p cell to a sparsity pattern for the
     * continuous Galerkin formulation, as
     * AffineConstraints::add_entries_local_to_global() with
     * keep_constrained_entries = false does: all degrees of freedom of the
     * cell couple with each other with constraints resolved, and every
     * constrained degree of freedom gets a diagonal entry. The function
     * adds to the rows of the degrees of freedom of the cell and of their
     * constraining degrees of freedom.
     *
     * @ingroup FiniteElement
     */
    template <typename CellIterator, typename Number>
    void add_cell_couplings(
        const CellIterator &cell,
        const dealii::AffineConstraints<Number> &affine_constraints,
        DynamicSparsityPattern &sparsity)
    {
      thread_local std::vector<dealii::types::global_dof_index> dof_indices;
      thread_local std::vector<dealii::types::global_dof_index> indices;

      dof_indices.resize(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_indices(dof_indices);

      indices.clear();
      for (const auto dof : dof_indices) {
        const auto *entries = affine_constraints.get_constraint_entries(dof);
        if (entries == nullptr) {
          indices.push_back(dof);
          continue;
        }
        for (const auto &[master, weight] : *entries)
          indices.push_back(master);
        sparsity.add_entries(dof, &dof, &dof + 1);
      }

      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
      for (const auto i : indices)
        sparsity.add_entries(i, indices.begin(), indices.end());
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
