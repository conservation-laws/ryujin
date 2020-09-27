//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef OFFLINE_DATA_TEMPLATE_H
#define OFFLINE_DATA_TEMPLATE_H

#include "local_index_handling.h"
#include "multicomponent_vector.h"
#include "offline_data.h"
#include "problem_description.h"
#include "scratch_data.h"
#include "sparse_matrix_simd.template.h" /* instantiate read_in */

#include <deal.II/base/graph_coloring.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#ifdef DEAL_II_WITH_TRILINOS
#include <deal.II/lac/trilinos_sparse_matrix.h>
#endif

#include <boost/range/irange.hpp>
#include <boost/range/iterator_range.hpp>

// #undef  DEAL_II_WITH_TRILINOS FIXME

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  OfflineData<dim, Number>::OfflineData(
      const MPI_Comm &mpi_communicator,
      const ryujin::Discretization<dim> &discretization,
      const std::string &subsection /*= "OfflineData"*/)
      : ParameterAcceptor(subsection)
      , discretization_(&discretization)
      , mpi_communicator_(mpi_communicator)
  {
  }


  template <int dim, typename Number>
  void OfflineData<dim, Number>::setup()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "OfflineData<dim, Number>::setup()" << std::endl;
#endif

    /* Initialize dof_handler and gather all locally owned indices: */

    dof_handler_.initialize(discretization_->triangulation(),
                            discretization_->finite_element());

    /*
     * Renumbering:
     */

    /* Cuthill McKee actually helps with cache locality. */
    DoFRenumbering::Cuthill_McKee(dof_handler_);

#ifdef USE_COMMUNICATION_HIDING
#ifdef DEBUG
    const unsigned int n_export_indices_preliminary =
#endif
        DoFRenumbering::export_indices_first(dof_handler_, mpi_communicator_);
#endif

#ifdef USE_SIMD
    n_locally_internal_ =
        DoFRenumbering::internal_range(dof_handler_, mpi_communicator_);

    /* Round down to the nearest multiple of the VectorizedArray width: */
    n_locally_internal_ = n_locally_internal_ -
                          n_locally_internal_ % VectorizedArray<Number>::size();
#else
    /*
     * If USE_SIMD is not set, we disable all SIMD instructions by
     * setting the [0, n_locally_internal) range to [0,0).
     */
    n_locally_internal_ = 0;
#endif

    /*
     * First, we set up the locally_relevant index set, determine (globally
     * indexed) affine constraints and create a (globally indexed) sparsity
     * pattern:
     */

    const IndexSet &locally_owned = dof_handler_.locally_owned_dofs();

    IndexSet locally_relevant;
    DoFTools::extract_locally_relevant_dofs(dof_handler_, locally_relevant);

    affine_constraints_.reinit(locally_relevant);
    DoFTools::make_hanging_node_constraints(dof_handler_, affine_constraints_);

#ifndef DEAL_II_WITH_TRILINOS
    AssertThrow(affine_constraints_.n_constraints() == 0,
                ExcMessage("ryujin was built without Trilinos support - no "
                           "hanging node support available"));
#endif

    /*
     * Enforce periodic boundary conditions. We assume that the mesh is in
     * "normal configuration".
     */
    const auto n_periodic_faces =
        discretization_->triangulation().get_periodic_face_map().size();
    if (n_periodic_faces != 0) {
      if constexpr (dim != 1 && std::is_same<Number, double>::value) {
        for (int i = 0; i < dim; ++i)
          DoFTools::make_periodicity_constraints(dof_handler_,
                                                 /*b_id */ Boundary::periodic,
                                                 /*direction*/ i,
                                                 affine_constraints_);
      } else {
        AssertThrow(false, dealii::ExcNotImplemented());
      }
    }

    affine_constraints_.close();

    sparsity_pattern_.reinit(
        dof_handler_.n_dofs(), dof_handler_.n_dofs(), locally_relevant);
#ifdef DEAL_II_WITH_TRILINOS
    DoFTools::make_sparsity_pattern(
        dof_handler_, sparsity_pattern_, affine_constraints_, false);
#else
    /*
     * In case we use dealii::SparseMatrix<Number> for assembly we need a
     * sparsity pattern that also includes the full locally relevant -
     * locally relevant coupling block. This gets thrown out again later,
     * but nevertheless we have to add it.
     */
    DoFTools::make_extended_sparsity_pattern(
        dof_handler_, sparsity_pattern_, affine_constraints_, false);
#endif

    /*
     * We have to complete the local stencil to have consistent size over
     * all MPI ranks. Otherwise, MPI synchronization in our
     * SparseMatrixSIMD class will fail.
     */

    SparsityTools::distribute_sparsity_pattern(
        sparsity_pattern_, locally_owned, mpi_communicator_, locally_relevant);

    /*
     * Next, we enlarge the locally relevant set to include all additional
     * couplings:
     */

    {
      IndexSet additional_dofs(dof_handler_.n_dofs());

      for (auto &entry : sparsity_pattern_)
        if (!locally_relevant.is_element(entry.column())) {
          Assert(locally_owned.is_element(entry.row()), ExcInternalError());
          additional_dofs.add_index(entry.column());
        }

      additional_dofs.compress();
      locally_relevant.add_indices(additional_dofs);
      locally_relevant.compress();
    }

    /* Set up partitioner: */

    n_locally_owned_ = locally_owned.n_elements();
    n_locally_relevant_ = locally_relevant.n_elements();

    scalar_partitioner_ = std::make_shared<dealii::Utilities::MPI::Partitioner>(
        locally_owned, locally_relevant, mpi_communicator_);

    constexpr auto problem_dimension =
        ProblemDescription::problem_dimension<dim>;
    vector_partitioner_ =
        create_vector_partitioner<problem_dimension>(scalar_partitioner_);

    /*
     * Determine the subset [0, n_export_indices) of [0,
     * n_locally_internal) that has to be computed before MPI exchange
     * communication can be started.
     */

#ifdef USE_COMMUNICATION_HIDING
    n_export_indices_ = 0;
    for (const auto &it : scalar_partitioner_->import_indices())
      if (it.second <= n_locally_internal_)
        n_export_indices_ = std::max(n_export_indices_, it.second);

    Assert(n_export_indices_ <= n_export_indices_preliminary,
           dealii::ExcInternalError());
#else
    n_export_indices_ = n_locally_internal_;
#endif

    /*
     * Set up SIMD sparsity pattern in local numbering. Nota bene: The
     * SparsityPatternSIMD::reinit() function will translates the pattern
     * from global deal.II (typical) dof indexing to local indices.
     */

    sparsity_pattern_simd_.reinit(
        n_locally_internal_, sparsity_pattern_, scalar_partitioner_);

    /* Next we can (re)initialize all local matrices: */

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

    measure_of_omega_ = 0.;
    boundary_map_.clear();

#ifdef DEAL_II_WITH_TRILINOS
    /* Variant using TrilinosWrappers::SparseMatrix with global numbering */

    const auto &affine_constraints_assembly = affine_constraints_;

    const IndexSet &locally_owned = dof_handler_.locally_owned_dofs();
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

    /*
     * Now, assemble all matrices:
     */

    /* The local, per-cell assembly routine: */

    const auto local_assemble_system = [&](const auto &cell,
                                           auto &scratch,
                                           auto &copy) {
      /* iterate over locally owned cells and the ghost layer */

      auto &is_locally_owned = copy.is_locally_owned_;
      auto &local_dof_indices = copy.local_dof_indices_;

      auto &local_boundary_map = copy.local_boundary_map_;
      auto &cell_mass_matrix = copy.cell_mass_matrix_;
      auto &cell_betaij_matrix = copy.cell_betaij_matrix_;
      auto &cell_cij_matrix = copy.cell_cij_matrix_;
      auto &cell_measure = copy.cell_measure_;

      auto &fe_values = scratch.fe_values_;
      auto &fe_face_values = scratch.fe_face_values_;

#ifdef DEAL_II_WITH_TRILINOS
      is_locally_owned = cell->is_locally_owned();
#else
      /*
       * When using a local dealii::SparseMatrix<Number> we don not have a
       * compress(VectorOperation::add) available. In this case just
       * assemble contributions over the locally
       */
      is_locally_owned = !cell->is_artificial();
#endif
      if (!is_locally_owned)
        return;

      cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_betaij_matrix.reinit(dofs_per_cell, dofs_per_cell);
      for (auto &matrix : cell_cij_matrix)
        matrix.reinit(dofs_per_cell, dofs_per_cell);

      fe_values.reinit(cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      /* clear out copy data: */
      local_boundary_map.clear();
      cell_mass_matrix = 0.;
      cell_betaij_matrix = 0.;
      for (auto &matrix : cell_cij_matrix)
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

      for (auto f : GeometryInfo<dim>::face_indices()) {
        const auto face = cell->face(f);
        const auto id = face->boundary_id();

        if (!face->at_boundary())
          continue;

        fe_face_values.reinit(cell, f);
        const unsigned int n_face_q_points = scratch.face_quadrature_.size();

        for (unsigned int j = 0; j < dofs_per_cell; ++j) {

          if (!discretization_->finite_element().has_support_on_face(j, f))
            continue;

          dealii::Tensor<1, dim, Number> normal;
          for (unsigned int q = 0; q < n_face_q_points; ++q)
            normal += fe_face_values.normal_vector(q) *
                      fe_face_values.shape_value(j, q);

          const auto global_index = local_dof_indices[j];
          const auto index = scalar_partitioner_->global_to_local(global_index);

          /*
           * This is a bloody hack: Use "vertex_dof_index" to retrieve the
           * vertex associated to the current degree of freedom.
           */
          Point<dim> position;
          for (auto v : GeometryInfo<dim>::vertex_indices())
            if (cell->vertex_dof_index(v, 0) == global_index) {
              position = cell->vertex(v);
              break;
            }

          local_boundary_map.insert({index, {normal, id, position}});
        } /* j */
      }   /* f */
    };

    const auto copy_local_to_global = [&](const auto &copy) {
      const auto &is_locally_owned = copy.is_locally_owned_;
      auto local_dof_indices = copy.local_dof_indices_; /* make a copy */
      const auto &local_boundary_map = copy.local_boundary_map_;
      const auto &cell_mass_matrix = copy.cell_mass_matrix_;
      const auto &cell_cij_matrix = copy.cell_cij_matrix_;
      const auto &cell_betaij_matrix = copy.cell_betaij_matrix_;
      const auto &cell_measure = copy.cell_measure_;

      if (!is_locally_owned)
        return;

      boundary_map_.insert(local_boundary_map.begin(),
                           local_boundary_map.end());

#ifndef DEAL_II_WITH_TRILINOS
      transform_to_local_range(*scalar_partitioner_, local_dof_indices);
#endif

      affine_constraints_assembly.distribute_local_to_global(
          cell_mass_matrix, local_dof_indices, mass_matrix_tmp);

      for (int k = 0; k < dim; ++k) {
        affine_constraints_assembly.distribute_local_to_global(
            cell_cij_matrix[k], local_dof_indices, cij_matrix_tmp[k]);
      }

      affine_constraints_assembly.distribute_local_to_global(
          cell_betaij_matrix, local_dof_indices, betaij_matrix_tmp);

      measure_of_omega_ += cell_measure;
    };

    WorkStream::run(dof_handler_.begin_active(),
                    dof_handler_.end(),
                    local_assemble_system,
                    copy_local_to_global,
                    AssemblyScratchData<dim>(*discretization_),
                    AssemblyCopyData<dim, Number>());

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
      scalar_type one(lumped_mass_matrix_);
      one = 1.;

      mass_matrix_tmp.vmult(lumped_mass_matrix_, one);
      lumped_mass_matrix_.compress(VectorOperation::add);

#else

      Vector<Number> one(mass_matrix_tmp.m());
      one = 1.;

      Vector<Number> local_lumped_mass_matrix(mass_matrix_tmp.m());
      mass_matrix_tmp.vmult(local_lumped_mass_matrix, one);

      for (unsigned int i = 0; i < scalar_partitioner_->local_size(); ++i) {
        lumped_mass_matrix_.local_element(i) = local_lumped_mass_matrix(i);
      lumped_mass_matrix_.update_ghost_values();
#endif

      for (unsigned int i = 0; i < scalar_partitioner_->local_size(); ++i) {
        lumped_mass_matrix_inverse_.local_element(i) =
            1. / lumped_mass_matrix_.local_element(i);
      }
      lumped_mass_matrix_inverse_.update_ghost_values();
    }

    /*
     * Update boundary map:
     *
     * At this point we have collected multiple cell contributions for each
     * boundary degree of freedom. We now merge all entries that have the
     * same boundary id and whose normals describe an acute angle of about
     * 85 degrees or less.
     */

    const auto temporary_boundary_map = std::move(boundary_map_);
    boundary_map_.clear();

    std::set<dealii::types::global_dof_index> boundary_dofs;
    for (auto entry : temporary_boundary_map) {
      bool inserted = false;
      const auto range = boundary_map_.equal_range(entry.first);
      for (auto it = range.first; it != range.second; ++it) {
        const auto &[new_normal, new_b_id, new_point] = entry.second;
        auto &[normal, b_id, point] = it->second;

        if (b_id != new_b_id)
          continue;

        Assert(point.distance(new_point) < 1.0e-16, dealii::ExcInternalError());

        if (normal * new_normal / normal.norm() / new_normal.norm() > 0.08) {
          /* Both normals describe an acute angle of 85 degrees or less. */
          normal += new_normal;
          inserted = true;
        }
      }
      if (!inserted)
        boundary_map_.insert(entry);
    }

    /*
     * Normalize all normal vectors:
     */
    for (auto &it : boundary_map_) {
      auto &[normal, id, _] = it.second;
      normal /= (normal.norm() + std::numeric_limits<Number>::epsilon());
    }

#ifdef DEBUG
    /*
     * Sanity check: The mass matrix must have a stencil with nonzero
     * entries.
     */
// FIXME
//     for(const auto &entry : mass_matrix_tmp)
//       Assert(std::abs(entry.value()) > 1.0e-16,
//              ExcMessage("Stencil with zero mass matrix entries encountered."));
#endif

#ifdef DEAL_II_WITH_TRILINOS
    betaij_matrix_.read_in(betaij_matrix_tmp, /*locally_indexed*/ false);
    mass_matrix_.read_in(mass_matrix_tmp, /*locally_indexed*/ false);
    cij_matrix_.read_in(cij_matrix_tmp, /*locally_indexed*/ false);
#else
    betaij_matrix_.read_in(betaij_matrix_tmp, /*locally_indexed*/ true);
    mass_matrix_.read_in(mass_matrix_tmp, /*locally_indexed*/ true);
    cij_matrix_.read_in(cij_matrix_tmp, /*locally_indexed*/ true);
#endif
  }

} /* namespace ryujin */

#endif /* OFFLINE_DATA_TEMPLATE_H */
