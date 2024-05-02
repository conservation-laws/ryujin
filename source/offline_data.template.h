//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "discretization.h"
#include "local_index_handling.h"
#include "multicomponent_vector.h"
#include "offline_data.h"
#include "scratch_data.h"
#include "sparse_matrix_simd.template.h" /* instantiate read_in */

#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#ifdef DEAL_II_WITH_TRILINOS
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif

#ifdef FORCE_DEAL_II_SPARSE_MATRIX
#undef DEAL_II_WITH_TRILINOS
#endif

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  OfflineData<dim, Number>::OfflineData(
      const MPI_Comm &mpi_communicator,
      const Discretization<dim> &discretization,
      const std::string &subsection /*= "OfflineData"*/)
      : ParameterAcceptor(subsection)
      , discretization_(&discretization)
      , mpi_communicator_(mpi_communicator)
  {
  }


  template <int dim, typename Number>
  void OfflineData<dim, Number>::create_constraints_and_sparsity_pattern()
  {
    /*
     * First, we set up the locally_relevant index set, determine (globally
     * indexed) affine constraints and create a (globally indexed) sparsity
     * pattern:
     */

    auto &dof_handler = *dof_handler_;
    const IndexSet &locally_owned = dof_handler.locally_owned_dofs();

    IndexSet locally_relevant;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);

    affine_constraints_.reinit(locally_relevant);
    DoFTools::make_hanging_node_constraints(dof_handler, affine_constraints_);

#ifndef DEAL_II_WITH_TRILINOS
    AssertThrow(affine_constraints_.n_constraints() == 0,
                ExcMessage("ryujin was built without Trilinos support - no "
                           "hanging node support available"));
#endif

    /*
     * Enforce periodic boundary conditions. We assume that the mesh is in
     * "normal configuration".
     */

    const auto &periodic_faces =
        discretization_->triangulation().get_periodic_face_map();

    for (const auto &[left, value] : periodic_faces) {
      const auto &[right, orientation] = value;

      typename DoFHandler<dim>::cell_iterator dof_cell_left(
          &left.first->get_triangulation(),
          left.first->level(),
          left.first->index(),
          &dof_handler);

      typename DoFHandler<dim>::cell_iterator dof_cell_right(
          &right.first->get_triangulation(),
          right.first->level(),
          right.first->index(),
          &dof_handler);

      if constexpr (dim != 1 && std::is_same<Number, double>::value) {
        DoFTools::make_periodicity_constraints(
            dof_cell_left->face(left.second),
            dof_cell_right->face(right.second),
            affine_constraints_,
            ComponentMask(),
#if DEAL_II_VERSION_GTE(9, 6, 0)
            orientation);
#else
            /* orientation */ orientation[0],
            /* flip */ orientation[1],
            /* rotation */ orientation[2]);
#endif
      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
        __builtin_trap();
      }
    }

    affine_constraints_.close();

    sparsity_pattern_.reinit(
        dof_handler.n_dofs(), dof_handler.n_dofs(), locally_relevant);

    switch (discretization_->ansatz()) {
      /*
       * Create cG sparsity pattern:
       */
    case Ansatz::cg_q1:
      [[fallthrough]];
    case Ansatz::cg_q2:
      [[fallthrough]];
    case Ansatz::cg_q3:
#ifdef DEAL_II_WITH_TRILINOS
      DoFTools::make_sparsity_pattern(
          dof_handler, sparsity_pattern_, affine_constraints_, false);
#else
      /*
       * In case we use dealii::SparseMatrix<Number> for assembly we need a
       * sparsity pattern that also includes the full locally relevant -
       * locally relevant coupling block. This gets thrown out again later,
       * but nevertheless we have to add it.
       */
      DoFTools::make_extended_sparsity_pattern(
          dof_handler, sparsity_pattern_, affine_constraints_, false);
#endif
      break;
      /*
       * Create dG sparsity pattern:
       */
    case Ansatz::dg_q0:
      [[fallthrough]];
    case Ansatz::dg_q1:
      [[fallthrough]];
    case Ansatz::dg_q2:
      [[fallthrough]];
    case Ansatz::dg_q3:
      DoFTools::make_extended_sparsity_pattern_dg(
          dof_handler, sparsity_pattern_, affine_constraints_, false);
      break;
    }

    /*
     * We have to complete the local stencil to have consistent size over
     * all MPI ranks. Otherwise, MPI synchronization in our
     * SparseMatrixSIMD class will fail.
     */

    SparsityTools::distribute_sparsity_pattern(
        sparsity_pattern_, locally_owned, mpi_communicator_, locally_relevant);
  }


  template <int dim, typename Number>
  void OfflineData<dim, Number>::setup(const unsigned int problem_dimension)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "OfflineData<dim, Number>::setup()" << std::endl;
#endif

    /*
     * Initialize dof handler:
     */

    const auto &triangulation = discretization_->triangulation();
    if (!dof_handler_)
      dof_handler_ = std::make_unique<dealii::DoFHandler<dim>>(triangulation);
    auto &dof_handler = *dof_handler_;

    dof_handler.distribute_dofs(discretization_->finite_element());

    n_locally_owned_ = dof_handler.locally_owned_dofs().n_elements();

    /*
     * Renumbering:
     */

    DoFRenumbering::Cuthill_McKee(dof_handler);

    /*
     * Reorder all (individual) export indices at the beginning of the
     * locally_internal index range to achieve a better packing:
     *
     * Note: This function might miss export indices that come from
     * eliminating hanging node and periodicity constraints (which we do
     * not know at this point because they depend on the renumbering...).
     */
    DoFRenumbering::export_indices_first(
        dof_handler, mpi_communicator_, n_locally_owned_, 1);

    /*
     * Group degrees of freedom that have the same stencil size in groups
     * of multiples of the VectorizedArray<Number>::size().
     *
     * In order to determine the stencil size we have to create a first,
     * temporary sparsity pattern:
     */
    create_constraints_and_sparsity_pattern();
    n_locally_internal_ = DoFRenumbering::internal_range(
        dof_handler, sparsity_pattern_, VectorizedArray<Number>::size());

    /*
     * Reorder all (strides of) locally internal indices that contain
     * export indices to the start of the index range. This reordering
     * preserves the binning introduced by
     * DoFRenumbering::internal_range().
     *
     * Note: This function might miss export indices that come from
     * eliminating hanging node and periodicity constraints (which we do
     * not know at this point because they depend on the renumbering...).
     * We therefore have to update n_export_indices_ later again.
     */
    n_export_indices_ =
        DoFRenumbering::export_indices_first(dof_handler,
                                             mpi_communicator_,
                                             n_locally_internal_,
                                             VectorizedArray<Number>::size());

    /*
     * A small lambda to check for stride-level consistency of the internal
     * index range:
     */
    const auto consistent_stride_range [[maybe_unused]] = [&]() {
      constexpr auto group_size = VectorizedArray<Number>::size();
      const IndexSet &locally_owned = dof_handler.locally_owned_dofs();
      const auto offset = n_locally_owned_ != 0 ? *locally_owned.begin() : 0;

      unsigned int group_row_length = 0;
      unsigned int i = 0;
      for (; i < n_locally_internal_; ++i) {
        if (i % group_size == 0) {
          group_row_length = sparsity_pattern_.row_length(offset + i);
        } else {
          if (group_row_length != sparsity_pattern_.row_length(offset + i)) {
            break;
          }
        }
      }
      return i / group_size * group_size;
    };

    /*
     * A small lambda that performs a logical or over all MPI ranks:
     */
    const auto mpi_allreduce_logical_or = [&](const bool local_value) {
      std::function<bool(const bool &, const bool &)> comparator =
          [](const bool &left, const bool &right) -> bool {
        return left || right;
      };
      return Utilities::MPI::all_reduce(
          local_value, mpi_communicator_, comparator);
    };

    /*
     * Create final sparsity pattern:
     */

    create_constraints_and_sparsity_pattern();

    /*
     * We have to ensure that the locally internal numbering range is still
     * consistent, meaning that all strides have the same stencil size.
     * This property might not hold any more after the elimination
     * procedure of constrained degrees of freedom (periodicity, or hanging
     * node constraints). Therefore, the following little dance:
     */

#if DEAL_II_VERSION_GTE(9, 5, 0)
    if (mpi_allreduce_logical_or(affine_constraints_.n_constraints() > 0)) {
      if (mpi_allreduce_logical_or( //
              consistent_stride_range() != n_locally_internal_)) {
        /*
         * In this case we try to fix up the numbering by pushing affected
         * strides to the end and slightly lowering the n_locally_internal_
         * marker.
         */
        n_locally_internal_ = DoFRenumbering::inconsistent_strides_last(
            dof_handler,
            sparsity_pattern_,
            n_locally_internal_,
            VectorizedArray<Number>::size());
        create_constraints_and_sparsity_pattern();
        n_locally_internal_ = consistent_stride_range();
      }
    }
#endif

    /*
     * Check that after all the dof manipulation and setup we still end up
     * with indices in [0, locally_internal) that have uniform stencil size
     * within a stride.
     */
    Assert(consistent_stride_range() == n_locally_internal_,
           dealii::ExcInternalError());

    /*
     * Set up partitioner:
     */

    const IndexSet &locally_owned = dof_handler.locally_owned_dofs();
    Assert(n_locally_owned_ == locally_owned.n_elements(),
           dealii::ExcInternalError());

    IndexSet locally_relevant;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);
    /* Enlarge the locally relevant set to include all additional couplings: */
    {
      IndexSet additional_dofs(dof_handler.n_dofs());
      for (auto &entry : sparsity_pattern_)
        if (!locally_relevant.is_element(entry.column())) {
          Assert(locally_owned.is_element(entry.row()), ExcInternalError());
          additional_dofs.add_index(entry.column());
        }
      additional_dofs.compress();
      locally_relevant.add_indices(additional_dofs);
      locally_relevant.compress();
    }

    n_locally_relevant_ = locally_relevant.n_elements();

    scalar_partitioner_ = std::make_shared<dealii::Utilities::MPI::Partitioner>(
        locally_owned, locally_relevant, mpi_communicator_);

    vector_partitioner_ =
        create_vector_partitioner(scalar_partitioner_, problem_dimension);

    /*
     * After elminiating periodicity and hanging node constraints we need
     * to update n_export_indices_ again. This happens because we need to
     * call export_indices_first() with incomplete information (missing
     * eliminated degrees of freedom).
     */
    if (mpi_allreduce_logical_or(affine_constraints_.n_constraints() > 0)) {
      /*
       * Recalculate n_export_indices_:
       */
      n_export_indices_ = 0;
      for (const auto &it : scalar_partitioner_->import_indices())
        if (it.second <= n_locally_internal_)
          n_export_indices_ = std::max(n_export_indices_, it.second);

      constexpr auto simd_length = VectorizedArray<Number>::size();
      n_export_indices_ =
          (n_export_indices_ + simd_length - 1) / simd_length * simd_length;
    }

#ifdef DEBUG
    /* Check that n_export_indices_ is valid: */
    unsigned int control = 0;
    for (const auto &it : scalar_partitioner_->import_indices())
      if (it.second <= n_locally_internal_)
        control = std::max(control, it.second);

    Assert(control <= n_export_indices_, ExcInternalError());
    Assert(n_export_indices_ <= n_locally_internal_, ExcInternalError());
#endif

    /*
     * Set up SIMD sparsity pattern in local numbering. Nota bene: The
     * SparsityPatternSIMD::reinit() function will translates the pattern
     * from global deal.II (typical) dof indexing to local indices.
     */

    sparsity_pattern_simd_.reinit(
        n_locally_internal_, sparsity_pattern_, scalar_partitioner_);

    /*
     * Next we can (re)initialize all local matrices:
     */

    lumped_mass_matrix_.reinit(scalar_partitioner_);
    lumped_mass_matrix_inverse_.reinit(scalar_partitioner_);

    mass_matrix_.reinit(sparsity_pattern_simd_);
    betaij_matrix_.reinit(sparsity_pattern_simd_);
    cij_matrix_.reinit(sparsity_pattern_simd_);
  }


  template <int dim, typename Number>
  void OfflineData<dim, Number>::assemble()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "OfflineData<dim, Number>::assemble()" << std::endl;
#endif

    auto &dof_handler = *dof_handler_;

    measure_of_omega_ = 0.;

#ifdef DEAL_II_WITH_TRILINOS
    /* Variant using TrilinosWrappers::SparseMatrix with global numbering */

    AffineConstraints<double> affine_constraints_assembly;
    /* This small dance is necessary to translate from Number to double: */
    affine_constraints_assembly.reinit(affine_constraints_.get_local_lines());
    for (auto line : affine_constraints_.get_lines()) {
      affine_constraints_assembly.add_line(line.index);
      for (auto entry : line.entries)
        affine_constraints_assembly.add_entry(
            line.index, entry.first, entry.second);
      affine_constraints_assembly.set_inhomogeneity(line.index,
                                                    line.inhomogeneity);
    }
    affine_constraints_assembly.close();

    const IndexSet &locally_owned = dof_handler.locally_owned_dofs();
    TrilinosWrappers::SparsityPattern trilinos_sparsity_pattern;
    trilinos_sparsity_pattern.reinit(
        locally_owned, sparsity_pattern_, mpi_communicator_);

    TrilinosWrappers::SparseMatrix mass_matrix_tmp;
    TrilinosWrappers::SparseMatrix betaij_matrix_tmp;
    std::array<TrilinosWrappers::SparseMatrix, dim> cij_matrix_tmp;

    mass_matrix_tmp.reinit(trilinos_sparsity_pattern);
    betaij_matrix_tmp.reinit(trilinos_sparsity_pattern);
    for (auto &matrix : cij_matrix_tmp)
      matrix.reinit(trilinos_sparsity_pattern);

#else
    /* Variant using deal.II SparseMatrix with local numbering */

    AffineConstraints<Number> affine_constraints_assembly;
    affine_constraints_assembly.copy_from(affine_constraints_);
    transform_to_local_range(*scalar_partitioner_, affine_constraints_assembly);

    SparsityPattern sparsity_pattern_assembly;
    {
      DynamicSparsityPattern dsp(n_locally_relevant_, n_locally_relevant_);
      for (const auto &entry : sparsity_pattern_) {
        const auto i = scalar_partitioner_->global_to_local(entry.row());
        const auto j = scalar_partitioner_->global_to_local(entry.column());
        dsp.add(i, j);
      }
      sparsity_pattern_assembly.copy_from(dsp);
    }

    dealii::SparseMatrix<Number> mass_matrix_tmp;
    dealii::SparseMatrix<Number> betaij_matrix_tmp;
    std::array<dealii::SparseMatrix<Number>, dim> cij_matrix_tmp;

    mass_matrix_tmp.reinit(sparsity_pattern_assembly);
    betaij_matrix_tmp.reinit(sparsity_pattern_assembly);
    for (auto &matrix : cij_matrix_tmp)
      matrix.reinit(sparsity_pattern_assembly);
#endif

    const unsigned int dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = discretization_->quadrature().size();
    const unsigned int n_face_q_points =
        discretization_->face_quadrature().size();

    bool assemble_dg_interface_terms = false;
    switch (discretization_->ansatz()) {
    case Ansatz::cg_q1:
    case Ansatz::cg_q2: /* fallthrough */
    case Ansatz::cg_q3: /* fallthrough */
      assemble_dg_interface_terms = false;
      break;
    case Ansatz::dg_q0:
    case Ansatz::dg_q1: /* fallthrough */
    case Ansatz::dg_q2: /* fallthrough */
    case Ansatz::dg_q3: /* fallthrough */
      assemble_dg_interface_terms = true;
      break;
    }

    /*
     * Now, assemble all matrices:
     */

    /* The local, per-cell assembly routine: */

    const auto local_assemble_system =
        [&](const auto &cell, auto &scratch, auto &copy) {
          /* iterate over locally owned cells and the ghost layer */

          auto &is_locally_owned = copy.is_locally_owned_;
          auto &local_dof_indices = copy.local_dof_indices_;
          auto &neighbor_local_dof_indices = copy.neighbor_local_dof_indices_;

          auto &cell_mass_matrix = copy.cell_mass_matrix_;
          auto &cell_betaij_matrix = copy.cell_betaij_matrix_;
          auto &cell_cij_matrix = copy.cell_cij_matrix_;
          auto &interface_cij_matrix = copy.interface_cij_matrix_;
          auto &cell_measure = copy.cell_measure_;

          auto &fe_values = scratch.fe_values_;
          auto &fe_face_values = scratch.fe_face_values_;
          auto &fe_neighbor_face_values = scratch.fe_neighbor_face_values_;

#ifdef DEAL_II_WITH_TRILINOS
          is_locally_owned = cell->is_locally_owned();
#else
          /*
           * When using a local dealii::SparseMatrix<Number> we don not
           * have a compress(VectorOperation::add) available. In this case
           * we assemble contributions over all locally relevant (non
           * artificial) cells.
           */
          is_locally_owned = !cell->is_artificial();
#endif
          if (!is_locally_owned)
            return;

          cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_betaij_matrix.reinit(dofs_per_cell, dofs_per_cell);
          for (auto &matrix : cell_cij_matrix)
            matrix.reinit(dofs_per_cell, dofs_per_cell);
          if (assemble_dg_interface_terms)
            for (auto &it : interface_cij_matrix)
              for (auto &matrix : it)
                matrix.reinit(dofs_per_cell, dofs_per_cell);

          fe_values.reinit(cell);

          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          /* clear out copy data: */
          cell_mass_matrix = 0.;
          cell_betaij_matrix = 0.;
          for (auto &matrix : cell_cij_matrix)
            matrix = 0.;
          if (assemble_dg_interface_terms)
            for (auto &it : interface_cij_matrix)
              for (auto &matrix : it)
                matrix = 0.;
          cell_measure = 0.;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const auto JxW = fe_values.JxW(q_point);

            if (cell->is_locally_owned())
              cell_measure += Number(JxW);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              const auto value_JxW = fe_values.shape_value(j, q_point) * JxW;
              const auto grad_JxW = fe_values.shape_grad(j, q_point) * JxW;

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {

                const auto value = fe_values.shape_value(i, q_point);
                const auto grad = fe_values.shape_grad(i, q_point);

                cell_mass_matrix(i, j) += Number(value * value_JxW);
                cell_betaij_matrix(i, j) += Number(grad * grad_JxW);
                for (unsigned int d = 0; d < dim; ++d)
                  cell_cij_matrix[d](i, j) += Number((value * grad_JxW)[d]);

              } /* for i */
            }   /* for j */
          }     /* for q */

          /*
           * Assemble dG face terms:
           */

          if (!assemble_dg_interface_terms)
            return;

          for (const auto f_index : cell->face_indices()) {
            const auto &face = cell->face(f_index);

            fe_face_values.reinit(cell, f_index);

            /* Skip faces without neighbors... */
            const bool has_neighbor =
                !face->at_boundary() || cell->has_periodic_neighbor(f_index);
            if (!has_neighbor) {
              // set the vector of local dof indices to 0 to indicate that
              // there is nothing to do for this face:
              neighbor_local_dof_indices[f_index].resize(0);
              continue;
            }

            /* Avoid artificial cells: */
            const auto neighbor_cell = cell->neighbor(f_index);
            if (neighbor_cell->is_artificial()) {
              // set the vector of local dof indices to 0 to indicate that
              // there is nothing to do for this face:
              neighbor_local_dof_indices[f_index].resize(0);
              continue;
            }

            /* Face contribution: */

            for (unsigned int q = 0; q < n_face_q_points; ++q) {
              const auto JxW = fe_face_values.JxW(q);
              const auto &normal = fe_face_values.get_normal_vectors()[q];

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const auto value_JxW = fe_face_values.shape_value(j, q) * JxW;

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const auto value = fe_face_values.shape_value(i, q);

                  for (unsigned int d = 0; d < dim; ++d)
                    cell_cij_matrix[d](i, j) -=
                        Number(0.5 * normal[d] * value * value_JxW);
                } /* for i */
              }   /* for j */
            }     /* for q */

            /* Coupling part: */

            const unsigned int f_index_neighbor =
                cell->neighbor_of_neighbor(f_index);

            neighbor_local_dof_indices[f_index].resize(dofs_per_cell);
            neighbor_cell->get_dof_indices(neighbor_local_dof_indices[f_index]);

            fe_neighbor_face_values.reinit(neighbor_cell, f_index_neighbor);

            for (unsigned int q = 0; q < n_face_q_points; ++q) {
              const auto JxW = fe_face_values.JxW(q);
              const auto &normal = fe_face_values.get_normal_vectors()[q];

              /* index j for neighbor, index i for current cell: */
              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const auto value_JxW =
                    fe_neighbor_face_values.shape_value(j, q) * JxW;

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const auto value = fe_face_values.shape_value(i, q);

                  for (unsigned int d = 0; d < dim; ++d)
                    interface_cij_matrix[f_index][d](i, j) +=
                        Number(0.5 * normal[d] * value * value_JxW);
                } /* for i */
              }   /* for j */
            }     /* for q */
          }
        };

    const auto copy_local_to_global = [&](const auto &copy) {
      const auto &is_locally_owned = copy.is_locally_owned_;
#ifdef DEAL_II_WITH_TRILINOS
      const auto &local_dof_indices = copy.local_dof_indices_;
      const auto &neighbor_local_dof_indices = copy.neighbor_local_dof_indices_;
#else
      /*
       * We have to transform indices to the local index range
       * [0, n_locally_relevant_) when using the dealii::SparseMatrix.
       * Thus, copy all index vectors:
       */
      auto local_dof_indices = copy.local_dof_indices_;
      auto neighbor_local_dof_indices = copy.neighbor_local_dof_indices_;
#endif
      const auto &cell_mass_matrix = copy.cell_mass_matrix_;
      const auto &cell_cij_matrix = copy.cell_cij_matrix_;
      const auto &interface_cij_matrix = copy.interface_cij_matrix_;
      const auto &cell_betaij_matrix = copy.cell_betaij_matrix_;
      const auto &cell_measure = copy.cell_measure_;

      if (!is_locally_owned)
        return;

#ifndef DEAL_II_WITH_TRILINOS
      transform_to_local_range(*scalar_partitioner_, local_dof_indices);
      for (auto &indices : neighbor_local_dof_indices)
        transform_to_local_range(*scalar_partitioner_, indices);
#endif

      affine_constraints_assembly.distribute_local_to_global(
          cell_mass_matrix, local_dof_indices, mass_matrix_tmp);

      for (int k = 0; k < dim; ++k) {
        affine_constraints_assembly.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, cij_matrix_tmp[k]);

        for (unsigned int f_index = 0; f_index < copy.n_faces; ++f_index) {
          if (neighbor_local_dof_indices[f_index].size() != 0) {
            affine_constraints_assembly.distribute_local_to_global(
                interface_cij_matrix[f_index][k],
                local_dof_indices,
                neighbor_local_dof_indices[f_index],
                cij_matrix_tmp[k]);
          }
        }
      }

      affine_constraints_assembly.distribute_local_to_global(
          cell_betaij_matrix, local_dof_indices, betaij_matrix_tmp);

      measure_of_omega_ += cell_measure;
    };

    WorkStream::run(dof_handler.begin_active(),
                    dof_handler.end(),
                    local_assemble_system,
                    copy_local_to_global,
                    AssemblyScratchData<dim>(*discretization_),
#ifdef DEAL_II_WITH_TRILINOS
                    AssemblyCopyData<dim, double>());
#else
                    AssemblyCopyData<dim, Number>());
#endif

    measure_of_omega_ =
        Utilities::MPI::sum(measure_of_omega_, mpi_communicator_);

#ifdef DEAL_II_WITH_TRILINOS
    betaij_matrix_tmp.compress(VectorOperation::add);
    mass_matrix_tmp.compress(VectorOperation::add);
    for (auto &it : cij_matrix_tmp)
      it.compress(VectorOperation::add);
#endif

    /*
     * Create lumped mass matrix:
     */

    {
#ifdef DEAL_II_WITH_TRILINOS
      using scalar_type = dealii::LinearAlgebra::distributed::Vector<double>;
      scalar_type one(scalar_partitioner_);
      one = 1.;

      scalar_type local_lumped_mass_matrix(scalar_partitioner_);
      mass_matrix_tmp.vmult(local_lumped_mass_matrix, one);
      lumped_mass_matrix_.compress(VectorOperation::add);

      for (unsigned int i = 0; i < scalar_partitioner_->locally_owned_size();
           ++i) {
        lumped_mass_matrix_.local_element(i) =
            local_lumped_mass_matrix.local_element(i);
        lumped_mass_matrix_inverse_.local_element(i) =
            1. / lumped_mass_matrix_.local_element(i);
      }
      lumped_mass_matrix_.update_ghost_values();
      lumped_mass_matrix_inverse_.update_ghost_values();

#else

      Vector<Number> one(mass_matrix_tmp.m());
      one = 1.;

      Vector<Number> local_lumped_mass_matrix(mass_matrix_tmp.m());
      mass_matrix_tmp.vmult(local_lumped_mass_matrix, one);

      for (unsigned int i = 0; i < scalar_partitioner_->locally_owned_size();
           ++i) {
        lumped_mass_matrix_.local_element(i) = local_lumped_mass_matrix(i);
        lumped_mass_matrix_inverse_.local_element(i) =
            1. / lumped_mass_matrix_.local_element(i);
      }
      lumped_mass_matrix_.update_ghost_values();
      lumped_mass_matrix_inverse_.update_ghost_values();
#endif
    }

#ifdef DEAL_II_WITH_TRILINOS
    betaij_matrix_.read_in(betaij_matrix_tmp, /*locally_indexed*/ false);
    mass_matrix_.read_in(mass_matrix_tmp, /*locally_indexed*/ false);
    cij_matrix_.read_in(cij_matrix_tmp, /*locally_indexed*/ false);
#else
    betaij_matrix_.read_in(betaij_matrix_tmp, /*locally_indexed*/ true);
    mass_matrix_.read_in(mass_matrix_tmp, /*locally_indexed*/ true);
    cij_matrix_.read_in(cij_matrix_tmp, /*locally_indexed*/ true);
#endif
    betaij_matrix_.update_ghost_rows();
    mass_matrix_.update_ghost_rows();
    cij_matrix_.update_ghost_rows();

    /* Populate boundary map and collect coupling boundary pairs: */

    boundary_map_ = construct_boundary_map(
        dof_handler.begin_active(), dof_handler.end(), *scalar_partitioner_);

    coupling_boundary_pairs_ = collect_coupling_boundary_pairs(
        dof_handler.begin_active(), dof_handler.end(), *scalar_partitioner_);
  }


  template <int dim, typename Number>
  void OfflineData<dim, Number>::create_multigrid_data()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "OfflineData<dim, Number>::create_multigrid_data()"
              << std::endl;
#endif

    auto &dof_handler = *dof_handler_;

    dof_handler.distribute_mg_dofs();

    const auto n_levels = dof_handler.get_triangulation().n_global_levels();

    AffineConstraints<float> level_constraints;
    // TODO not yet thread-parallel and without periodicity

    level_boundary_map_.resize(n_levels);
    level_lumped_mass_matrix_.resize(n_levels);

    for (unsigned int level = 0; level < n_levels; ++level) {
      /* Assemble lumped mass matrix vector: */

      IndexSet relevant_dofs;
      dealii::DoFTools::extract_locally_relevant_level_dofs(
          dof_handler, level, relevant_dofs);
      const auto partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          dof_handler.locally_owned_mg_dofs(level),
          relevant_dofs,
          lumped_mass_matrix_.get_mpi_communicator());
      level_lumped_mass_matrix_[level].reinit(partitioner);
      std::vector<types::global_dof_index> dof_indices(
          dof_handler.get_fe().dofs_per_cell);
      Vector<Number> mass_values(dof_handler.get_fe().dofs_per_cell);
      FEValues<dim> fe_values(discretization_->mapping(),
                              discretization_->finite_element(),
                              discretization_->quadrature(),
                              update_values | update_JxW_values);
      for (const auto &cell : dof_handler.cell_iterators_on_level(level))
        // TODO for assembly with dealii::SparseMatrix and local
        // numbering this probably has to read !cell->is_artificial()
        if (cell->is_locally_owned_on_level()) {
          fe_values.reinit(cell);
          for (unsigned int i = 0; i < mass_values.size(); ++i) {
            double sum = 0;
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              sum += fe_values.shape_value(i, q) * fe_values.JxW(q);
            mass_values(i) = sum;
          }
          cell->get_mg_dof_indices(dof_indices);
          level_constraints.distribute_local_to_global(
              mass_values, dof_indices, level_lumped_mass_matrix_[level]);
        }
      level_lumped_mass_matrix_[level].compress(VectorOperation::add);

      /* Populate boundary map: */

      level_boundary_map_[level] = construct_boundary_map(
          dof_handler.begin_mg(level), dof_handler.end_mg(level), *partitioner);
    }
  }


  template <int dim, typename Number>
  template <typename ITERATOR1, typename ITERATOR2>
  typename OfflineData<dim, Number>::boundary_map_type
  OfflineData<dim, Number>::construct_boundary_map(
      const ITERATOR1 &begin,
      const ITERATOR2 &end,
      const Utilities::MPI::Partitioner &partitioner) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "OfflineData<dim, Number>::construct_boundary_map()"
              << std::endl;
#endif

    decltype(boundary_map_) preliminary_map;

    std::vector<dealii::types::global_dof_index> local_dof_indices;

    const dealii::QGauss<dim - 1> face_quadrature(3);
    dealii::FEFaceValues<dim> fe_face_values(discretization_->mapping(),
                                             discretization_->finite_element(),
                                             face_quadrature,
                                             dealii::update_normal_vectors |
                                                 dealii::update_values |
                                                 dealii::update_JxW_values);

    const unsigned int dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    const auto support_points =
        discretization_->finite_element().get_unit_support_points();

    for (auto cell = begin; cell != end; ++cell) {

      if (!cell->is_locally_owned_on_level())
        continue;

      local_dof_indices.resize(dofs_per_cell);
      cell->get_active_or_mg_dof_indices(local_dof_indices);

      for (auto f : GeometryInfo<dim>::face_indices()) {
        const auto face = cell->face(f);
        const auto id = face->boundary_id();

        if (!face->at_boundary())
          continue;

        /*
         * Skip periodic boundary faces. For our algorithm these are
         * interior degrees of freedom (if not simultaneously located at
         * another boundary as well).
         */
        if (id == Boundary::periodic)
          continue;

        fe_face_values.reinit(cell, f);
        const unsigned int n_face_q_points = face_quadrature.size();

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          if (!discretization_->finite_element().has_support_on_face(j, f))
            continue;

          Number boundary_mass = 0.;
          dealii::Tensor<1, dim, Number> normal;

          for (unsigned int q = 0; q < n_face_q_points; ++q) {
            const auto JxW = fe_face_values.JxW(q);
            const auto phi_i = fe_face_values.shape_value(j, q);

            boundary_mass += phi_i * JxW;
            normal += phi_i * fe_face_values.normal_vector(q) * JxW;
          }

          const auto global_index = local_dof_indices[j];
          const auto index = partitioner.global_to_local(global_index);

          /* Skip nonlocal degrees of freedom: */
          if (index >= n_locally_owned_)
            continue;

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length =
              sparsity_pattern_simd_.row_length(index);
          if (row_length == 1)
            continue;

          Point<dim> position =
              discretization_->mapping().transform_unit_to_real_cell(
                  cell, support_points[j]);

          /*
           * Temporarily insert a (wrong) boundary mass value for the
           * normal mass. We'll fix this later.
           */
          preliminary_map.insert(
              {index, {normal, boundary_mass, boundary_mass, id, position}});
        } /* j */
      }   /* f */
    }     /* cell */

    /*
     * Filter boundary map:
     *
     * At this point we have collected multiple cell contributions for each
     * boundary degree of freedom. We now merge all entries that have the
     * same boundary id and whose normals describe an acute angle of about
     * 85 degrees or less.
     *
     * FIXME: is this robust in 3D?
     */

    decltype(boundary_map_) filtered_map;
    std::set<dealii::types::global_dof_index> boundary_dofs;
    for (auto entry : preliminary_map) {
      bool inserted = false;
      const auto range = filtered_map.equal_range(entry.first);
      for (auto it = range.first; it != range.second; ++it) {
        const auto &[new_normal,
                     new_normal_mass,
                     new_boundary_mass,
                     new_id,
                     new_point] = entry.second;
        auto &[normal, normal_mass, boundary_mass, id, point] = it->second;

        if (id != new_id)
          continue;

        Assert(point.distance(new_point) < 1.0e-14, dealii::ExcInternalError());

        if (normal * new_normal / normal.norm() / new_normal.norm() > 0.08) {
          /* Both normals describe an acute angle of 85 degrees or less. */
          normal += new_normal;
          boundary_mass += new_boundary_mass;
          inserted = true;
        }
      }
      if (!inserted)
        filtered_map.insert(entry);
    }

    /* Normalize all normal vectors: */

    for (auto &it : filtered_map) {
      auto &[normal, normal_mass, boundary_mass, id, point] = it.second;
      const auto new_normal_mass =
          normal.norm() + std::numeric_limits<Number>::epsilon();
      /* Replace boundary mass with new definition: */
      normal_mass = new_normal_mass;
      normal /= new_normal_mass;
    }

    return filtered_map;
  }


  template <int dim, typename Number>
  template <typename ITERATOR1, typename ITERATOR2>
  typename OfflineData<dim, Number>::coupling_boundary_pairs_type
  OfflineData<dim, Number>::collect_coupling_boundary_pairs(
      const ITERATOR1 &begin,
      const ITERATOR2 &end,
      const Utilities::MPI::Partitioner &partitioner) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "OfflineData<dim, Number>::collect_coupling_boundary_pairs()"
              << std::endl;
#endif

    /*
     * First, collect *all* locally relevant degrees of freedom that are
     * located on a (non periodic) boundary. We also collect constrained
     * degrees of freedom for the time being (and filter afterwards).
     */

    std::set<unsigned int> locally_relevant_boundary_indices;

    std::vector<dealii::types::global_dof_index> local_dof_indices;

    const unsigned int dofs_per_cell =
        discretization_->finite_element().dofs_per_cell;

    for (auto cell = begin; cell != end; ++cell) {

      /* Make sure to iterate over the entire locally relevant set: */
      if (cell->is_artificial())
        continue;

      local_dof_indices.resize(dofs_per_cell);
      cell->get_active_or_mg_dof_indices(local_dof_indices);

      for (auto f : GeometryInfo<dim>::face_indices()) {
        const auto face = cell->face(f);
        const auto id = face->boundary_id();

        if (!face->at_boundary())
          continue;

        /* Skip periodic boundary faces; see above. */
        if (id == Boundary::periodic)
          continue;

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          if (!discretization_->finite_element().has_support_on_face(j, f))
            continue;

          const auto global_index = local_dof_indices[j];
          const auto index = partitioner.global_to_local(global_index);

          /* Skip irrelevant degrees of freedom: */
          if (index >= n_locally_relevant_)
            continue;

          locally_relevant_boundary_indices.insert(index);
        } /* j */
      }   /* f */
    }     /* cell */

    /*
     * Now, collect all coupling boundary pairs:
     */

    coupling_boundary_pairs_type result;

    for (const auto i : locally_relevant_boundary_indices) {

      /* Only record pairs with a left index that is locally owned: */
      if (i >= n_locally_owned_)
        continue;

      const unsigned int row_length = sparsity_pattern_simd_.row_length(i);

      /* Skip all constrained degrees of freedom: */
      if (row_length == 1)
        continue;

      const unsigned int *js = sparsity_pattern_simd_.columns(i);
      constexpr auto simd_length = VectorizedArray<Number>::size();
      /* skip diagonal: */
      for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
        const auto j = *(i < n_locally_internal_ ? js + col_idx * simd_length
                                                 : js + col_idx);

        if (locally_relevant_boundary_indices.count(j) != 0) {
          result.push_back({i, col_idx, j});
        }
      }
    }

    return result;
  }

} /* namespace ryujin */
