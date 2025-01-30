//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2024 - 2024 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "discretization.h"
#include "solution_transfer.h"
#if DEAL_II_VERSION_GTE(9, 6, 0)
#include "tensor_product_point_kernels.h"
#endif

#include <deal.II/base/config.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#if DEAL_II_VERSION_GTE(9, 6, 0)
#include <deal.II/grid/cell_status.h>
#endif
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>

#define EXPENSIVE_BOUNDS_CHECK

namespace ryujin
{
  template <typename Description, int dim, typename Number>
  SolutionTransfer<Description, dim, Number>::SolutionTransfer(
      const MPIEnsemble &mpi_ensemble,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const ParabolicSystem &parabolic_system,
      const std::string &subsection /* = "/SolutionTransfer" */)
      : ParameterAcceptor(subsection)
      , limiter_parameters_(subsection + "/mass transfer limiter")
      , mpi_ensemble_(mpi_ensemble)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , handle_(dealii::numbers::invalid_unsigned_int)
  {
  }


  namespace
  {
    /**
     * Pack a vector of local state values into a char vector.
     */
    template <typename state_type>
    std::vector<char>
    pack_state_values(const std::vector<state_type> &state_values)
    {
      std::vector<char> buffer(sizeof(state_type) * state_values.size());
      std::memcpy(buffer.data(), state_values.data(), buffer.size());
      return buffer;
    }


    /**
     * Unpack a char vector into a vector of local state values.
     */
    template <typename state_type>
    std::vector<state_type> unpack_state_values(
        const boost::iterator_range<std::vector<char>::const_iterator>
            &data_range)
    {
      const std::size_t n_bytes = data_range.size();
      Assert(n_bytes % sizeof(state_type) == 0, dealii::ExcInternalError());
      std::vector<state_type> state_values(n_bytes / sizeof(state_type));
      std::memcpy(state_values.data(),
                  &data_range[0],
                  state_values.size() * sizeof(state_type));
      return state_values;
    }
  } // namespace


  template <typename Description, int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE auto
  SolutionTransfer<Description, dim, Number>::get_tensor(
      const HyperbolicVector &U, const dealii::types::global_dof_index global_i)
      -> state_type
  {
    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto local_i = scalar_partitioner->global_to_local(global_i);
    if (affine_constraints.is_constrained(global_i)) {
      state_type result;
      const auto &line = *affine_constraints.get_constraint_entries(global_i);
      for (const auto &[global_k, c_k] : line) {
        const auto local_k = scalar_partitioner->global_to_local(global_k);
        result += c_k * U.get_tensor(local_k);
      }
      return result;
    } else {
      return U.get_tensor(local_i);
    }
  }


  template <typename Description, int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE void
  SolutionTransfer<Description, dim, Number>::add_tensor(
      HyperbolicVector &U,
      const state_type &new_U_i,
      const dealii::types::global_dof_index global_i)
  {
    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    const auto local_i = scalar_partitioner->global_to_local(global_i);
    U.add_tensor(new_U_i, local_i);
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::prepare_projection(
      const StateVector &old_state_vector [[maybe_unused]])
  {
#ifdef DEBUG_OUTPUT
    std::cout
        << "SolutionTransfer<Description, dim, Number>::prepare_projection()"
        << std::endl;
#endif

#if !DEAL_II_VERSION_GTE(9, 6, 0)
    AssertThrow(
        false,
        dealii::ExcMessage(
            "The SolutionTransfer class needs deal.II version 9.6.0 or newer"));

#else
    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    const auto &discretization = offline_data_->discretization();
    auto &triangulation = *discretization.triangulation_; /* writable */

    Assert(handle_ == dealii::numbers::invalid_unsigned_int,
           dealii::ExcMessage(
               "You can only add one solution per SolutionTransfer object."));

    /*
     * Add a register_data_attach callback that
     */

    handle_ = triangulation.register_data_attach(
        [this, &old_state_vector](const auto cell,
                                  const dealii::CellStatus status) {
          const auto &dof_handler = offline_data_->dof_handler();
          const auto dof_cell = typename dealii::DoFHandler<dim>::cell_iterator(
              &cell->get_triangulation(),
              cell->level(),
              cell->index(),
              &dof_handler);

          const auto &scalar_partitioner = offline_data_->scalar_partitioner();

          const auto view = hyperbolic_system_->template view<dim, Number>();

          const auto &U = std::get<0>(old_state_vector);
          const auto &precomputed = std::get<1>(old_state_vector);

          using Limiter = typename Description::template Limiter<dim, Number>;
          const Limiter limiter(
              *hyperbolic_system_, limiter_parameters_, precomputed);

          /*
           * Collect state values and merged bounds for packing:
           */

          const auto n_dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
          std::vector<state_type> state_values(n_dofs_per_cell);
          Bounds bounds;

          switch (status) {
          case dealii::CellStatus::cell_will_persist:
            [[fallthrough]];
          case dealii::CellStatus::cell_will_be_refined: {
            /*
             * For both cases we need state values from the currently
             * active cell:
             */

            Assert(dof_cell->is_active(), dealii::ExcInternalError());
            std::vector<dealii::types::global_dof_index> dof_indices(
                n_dofs_per_cell);
            dof_cell->get_dof_indices(dof_indices);

            /* We want a "left fold first" for the bounds: */
            if (std::begin(dof_indices) != std::end(dof_indices)) {
              const auto global_i = dof_indices[0];
              const auto U_i = get_tensor(U, global_i);
              const auto local_i =
                  scalar_partitioner->global_to_local(global_i);
              bounds = limiter.projection_bounds_from_state(local_i, U_i);
            }

            std::transform( //
                std::begin(dof_indices),
                std::end(dof_indices),
                std::begin(state_values),
                [&](const auto global_i) {
                  const auto U_i = get_tensor(U, global_i);
                  const auto local_i =
                      scalar_partitioner->global_to_local(global_i);
                  const auto bounds_i =
                      limiter.projection_bounds_from_state(local_i, U_i);
                  bounds = limiter.combine_bounds(bounds, bounds_i);
                  return U_i;
                });
          } break;

          case dealii::CellStatus::children_will_be_coarsened: {
            /*
             * We need to project values from the active child cells up to
             * the present parent cell that will become active after
             * coarsening.
             */

            Assert(dof_cell->has_children(), dealii::ExcInternalError());

            const auto &discretization = offline_data_->discretization();
            const auto &finite_element = discretization.finite_element();
            const auto &mapping = discretization.mapping();
            const auto &quadrature = discretization.quadrature();

            dealii::FEValues<dim> fe_values(
                mapping,
                finite_element,
                quadrature,
                dealii::update_values | dealii::update_JxW_values |
                    dealii::update_quadrature_points);

            const auto polynomial_space =
                dealii::internal::FEPointEvaluation::get_polynomial_space(
                    finite_element);

            std::vector<dealii::Point<dim, Number>> unit_points(
                quadrature.size());
            /*
             * for Number == float we need a temporary vector for the
             * transform_points_real_to_unit_cell() function:
             */
            std::vector<dealii::Point<dim>> unit_points_temp(
                std::is_same_v<Number, float> ? quadrature.size() : 0);

            /*
             * Step 1: build up right hand side by iterating over children:
             */

            std::vector<state_type> state_values_quad(quadrature.size());
            std::vector<state_type> local_rhs(n_dofs_per_cell);

            std::vector<dealii::types::global_dof_index> dof_indices(
                n_dofs_per_cell);

            for (unsigned int child = 0; child < dof_cell->n_children();
                 ++child) {
              const auto child_cell = dof_cell->child(child);
              Assert(child_cell->is_active(), dealii::ExcInternalError());

              fe_values.reinit(child_cell);

              if constexpr (std::is_same_v<Number, float>) {
                mapping.transform_points_real_to_unit_cell(
                    dof_cell,
                    fe_values.get_quadrature_points(),
                    unit_points_temp);
                std::transform(std::begin(unit_points_temp),
                               std::end(unit_points_temp),
                               std::begin(unit_points),
                               [](const auto &x) { return x; });
              } else {
                mapping.transform_points_real_to_unit_cell(
                    dof_cell, fe_values.get_quadrature_points(), unit_points);
              }

              child_cell->get_dof_indices(dof_indices);

              /* We want a "left fold first" for the bounds: */
              if (child == 0 &&
                  std::begin(dof_indices) != std::end(dof_indices)) {
                const auto global_i = dof_indices[0];
                const auto U_i = get_tensor(U, global_i);
                const auto local_i =
                    scalar_partitioner->global_to_local(global_i);
                bounds = limiter.projection_bounds_from_state(local_i, U_i);
              }

              for (auto &it : state_values_quad)
                it = state_type{};

              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                const auto global_i = dof_indices[i];
                const auto U_i = get_tensor(U, global_i);
                const auto local_i =
                    scalar_partitioner->global_to_local(global_i);
                const auto bounds_i =
                    limiter.projection_bounds_from_state(local_i, U_i);
                bounds = limiter.combine_bounds(bounds, bounds_i);

                for (unsigned int q = 0; q < quadrature.size(); ++q) {
                  state_values_quad[q] += U_i * fe_values.shape_value(i, q);
                }
              }

              for (unsigned int q = 0; q < quadrature.size(); ++q)
                state_values_quad[q] *= fe_values.JxW(q);

              for (unsigned int q = 0; q < quadrature.size(); ++q) {
                const unsigned int n_shapes = polynomial_space.size();
                AssertIndexRange(n_shapes, 10);
                dealii::ndarray<Number, 10, 2, dim> shapes;
                // Evaluate 1d polynomials and their derivatives
                std::array<Number, dim> point;
                for (unsigned int d = 0; d < dim; ++d)
                  point[d] = unit_points[q][d];
                for (unsigned int i = 0; i < n_shapes; ++i)
                  polynomial_space[i].values_of_array(point, 1, &shapes[i][0]);

                Assert(finite_element.degree == 1, dealii::ExcNotImplemented());

                ryujin::internal::integrate_tensor_product_value<
                    /*is linear*/ true,
                    dim,
                    Number,
                    state_type>(shapes.data(),
                                n_shapes,
                                state_values_quad[q],
                                local_rhs.data(),
                                unit_points[q],
                                true);
              }
            }

            /* Step 2: construct inverse mass matrix on coarse cell: */

            fe_values.reinit(dof_cell);

            dealii::FullMatrix<double> mij(n_dofs_per_cell, n_dofs_per_cell);
            dealii::Vector<double> mi(n_dofs_per_cell);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                double sum = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  sum += fe_values.shape_value(i, q) *
                         fe_values.shape_value(j, q) * fe_values.JxW(q);
                mij(i, j) = sum;
                mi(i) += sum;
              }
            }

            mij.gauss_jordan();

            /* FIXME */

            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              // high order:
              state_type U_i;
              for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                U_i += mij(i, j) * local_rhs[j];
              }

              if (view.is_admissible(U_i))
                state_values[i] = U_i;
              else {
                std::cout << "DEBUG: inadmissible state encountered:\n»» "
                          << U_i << " ««" << std::endl;
                // low order:
                state_values[i] = 1. / mi(i) * local_rhs[i];
              }

#ifdef EXPENSIVE_BOUNDS_CHECK
              AssertThrow(
                  view.is_admissible(state_values[i]),
                  dealii::ExcMessage(
                      "Error: inadmissible state encountered in "
                      "register_data_attach / children_will_be_coarsened"));
#endif
            }
          } break;

          case dealii::CellStatus::cell_invalid:
            Assert(false, dealii::ExcInternalError());
            __builtin_trap();
            break;
          }

          return pack_state_values(state_values);
        },
        /* returns_variable_size_data =*/false);
#endif
  }


  template <typename Description, int dim, typename Number>
  void SolutionTransfer<Description, dim, Number>::project(
      StateVector &new_state_vector [[maybe_unused]])
  {
#ifdef DEBUG_OUTPUT
    std::cout << "SolutionTransfer<Description, dim, Number>::project()"
              << std::endl;
#endif

#if !DEAL_II_VERSION_GTE(9, 6, 0)
    AssertThrow(
        false,
        dealii::ExcMessage(
            "The SolutionTransfer class needs deal.II version 9.6.0 or newer"));

#else

    AssertThrow(have_distributed_triangulation<dim>,
                dealii::ExcMessage(
                    "The SolutionTransfer class is not implemented for a "
                    "distributed::shared::Triangulation which we use in 1D"));

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    const auto &affine_constraints = offline_data_->affine_constraints();

    const auto &discretization = offline_data_->discretization();
    auto &triangulation = *discretization.triangulation_; /* writable */

    Assert(
        handle_ != dealii::numbers::invalid_unsigned_int,
        dealii::ExcMessage(
            "Cannot project() a state vector without valid handle. "
            "prepare_projection() or set_handle() have to be called first."));

    /*
     * Reconstruct and project state vector:
     */

    ScalarVector projected_mass;
    projected_mass.reinit(offline_data_->scalar_partitioner());
    HyperbolicVector projected_state;
    projected_state.reinit(offline_data_->hyperbolic_vector_partitioner());

    triangulation.notify_ready_to_unpack( //
        handle_,
        [this, &projected_mass, &projected_state](
            const auto &cell,
            const dealii::CellStatus status,
            const auto &data_range) {
          const auto &dof_handler = offline_data_->dof_handler();
          const auto dof_cell = typename dealii::DoFHandler<dim>::cell_iterator(
              &cell->get_triangulation(),
              cell->level(),
              cell->index(),
              &dof_handler);

          const auto view = hyperbolic_system_->template view<dim, Number>();

          /*
           * Retrieve packed values and project onto cell:
           */

          const auto n_dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
          std::vector<dealii::types::global_dof_index> dof_indices(
              n_dofs_per_cell);

          const auto state_values = unpack_state_values<state_type>(data_range);

          switch (status) {
          case dealii::CellStatus::cell_will_persist:
            [[fallthrough]];
          case dealii::CellStatus::children_will_be_coarsened: {
            /*
             * For both cases we distribute stored state_values to the
             * projected_state and projected_mass vectors.
             */

            Assert(dof_cell->is_active(), dealii::ExcInternalError());
            dof_cell->get_dof_indices(dof_indices);

            const auto &discretization = offline_data_->discretization();
            const auto &finite_element = discretization.finite_element();
            const auto &mapping = discretization.mapping();
            const auto &quadrature = discretization.quadrature();

            dealii::FEValues<dim> fe_values(mapping,
                                            finite_element,
                                            quadrature,
                                            dealii::update_values |
                                                dealii::update_JxW_values);

            fe_values.reinit(dof_cell);

            dealii::Vector<double> mi(n_dofs_per_cell);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              double sum = 0;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                sum += fe_values.shape_value(i, q) * fe_values.JxW(q);
              mi(i) += sum;
            }

            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              const auto global_i = dof_indices[i];
              add_tensor(projected_state, mi(i) * state_values[i], global_i);
              projected_mass(global_i) += mi(i);
            }

          } break;

          case dealii::CellStatus::cell_will_be_refined: {
            /*
             * We are on a (non active) cell that has been refined. Project
             * onto the children and do a local mass projection there:
             */

            Assert(dof_cell->has_children(), dealii::ExcInternalError());

            const auto &discretization = offline_data_->discretization();
            const auto &finite_element = discretization.finite_element();
            const auto &mapping = discretization.mapping();
            const auto &quadrature = discretization.quadrature();

            dealii::FEValues<dim> fe_values(
                mapping,
                finite_element,
                quadrature,
                dealii::update_values | dealii::update_JxW_values |
                    dealii::update_quadrature_points);

            const auto polynomial_space =
                dealii::internal::FEPointEvaluation::get_polynomial_space(
                    finite_element);
            std::vector<dealii::Point<dim, Number>> unit_points(
                quadrature.size());
            /*
             * for Number == float we need a temporary vector for the
             * transform_points_real_to_unit_cell() function:
             */
            std::vector<dealii::Point<dim>> unit_points_temp(
                std::is_same_v<Number, float> ? quadrature.size() : 0);

            dealii::FullMatrix<double> mij(n_dofs_per_cell, n_dofs_per_cell);
            dealii::Vector<double> mi(n_dofs_per_cell);
            std::vector<state_type> local_rhs(n_dofs_per_cell);

            for (unsigned int child = 0; child < dof_cell->n_children();
                 ++child) {
              const auto child_cell = dof_cell->child(child);

              Assert(child_cell->is_active(), dealii::ExcInternalError());
              child_cell->get_dof_indices(dof_indices);

              /* Step 1: build up right hand side on child cell: */

              fe_values.reinit(child_cell);

              if constexpr (std::is_same_v<Number, float>) {
                mapping.transform_points_real_to_unit_cell(
                    dof_cell,
                    fe_values.get_quadrature_points(),
                    unit_points_temp);
                std::transform(std::begin(unit_points_temp),
                               std::end(unit_points_temp),
                               std::begin(unit_points),
                               [](const auto &x) { return x; });
              } else {
                mapping.transform_points_real_to_unit_cell(
                    dof_cell, fe_values.get_quadrature_points(), unit_points);
              }

              for (auto &it : local_rhs)
                it = state_type{};

              for (unsigned int q = 0; q < quadrature.size(); ++q) {
                Assert(finite_element.degree == 1, dealii::ExcNotImplemented());
                auto coefficient =
                    dealii::internal::evaluate_tensor_product_value(
                        polynomial_space,
                        make_const_array_view(state_values),
                        unit_points[q],
                        /*is linear*/ true);
                coefficient *= fe_values.JxW(q);

                for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                  local_rhs[i] += coefficient * fe_values.shape_value(i, q);
              }

              /* Step 2: solve with inverse mass matrix on child cell: */

              /* FIXME */

              mi = 0.;
              mij = 0.;
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                  double sum = 0;
                  for (unsigned int q = 0; q < quadrature.size(); ++q)
                    sum += fe_values.shape_value(i, q) *
                           fe_values.shape_value(j, q) * fe_values.JxW(q);
                  mij(i, j) = sum;
                  mi(i) += sum;
                }
              }

              mij.gauss_jordan();

              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                state_type U_i;
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                  U_i += mij(i, j) * local_rhs[j];
                }

#ifdef EXPENSIVE_BOUNDS_CHECK
                AssertThrow(view.is_admissible(U_i),
                            dealii::ExcMessage(
                                "Error: inadmissible state encountered in "
                                "ready_to_unpack / cell_will_be_refined"));
#endif
                const auto global_i = dof_indices[i];
                add_tensor(projected_state, mi(i) * U_i, global_i);
                projected_mass(global_i) += mi(i);
              }
            } /*child*/

          } break;

          case dealii::CellStatus::cell_invalid:
            Assert(false, dealii::ExcInternalError());
            __builtin_trap();
            break;
          }
        });

    /*
     * Distribute values, take the weighted average of unconstrained
     * degrees of freedom, and store the result in new_U:
     */

    auto &new_U = std::get<0>(new_state_vector);
    const auto n_locally_owned = offline_data_->n_locally_owned();
    const auto view = hyperbolic_system_->template view<dim, Number>();

    // We have to perform the following operation twice, so let's create a
    // small lambda for it.
    const auto update_new_state_vector = [&]() {
      projected_mass.compress(dealii::VectorOperation::add);
      projected_state.compress(dealii::VectorOperation::add);

      for (unsigned int local_i = 0; local_i < n_locally_owned; ++local_i) {
        const auto global_i = scalar_partitioner->local_to_global(local_i);
        if (affine_constraints.is_constrained(global_i))
          continue;

        const auto U_i = projected_state.get_tensor(local_i);
        const auto m_i = projected_mass.local_element(local_i);

#ifdef EXPENSIVE_BOUNDS_CHECK
        AssertThrow(
            view.is_admissible(U_i),
            dealii::ExcMessage("Error: inadmissible state encountered in "
                               "update_new_state_vector()"));
#endif

        new_U.write_tensor(U_i / m_i, local_i);
      }
      new_U.update_ghost_values();
    };

    update_new_state_vector();

    /*
     * Now redistribute the mass defect introduced by constrained degrees
     * of freedom. This mostly affects hanging nodes neighboring a
     * coarsened cell. Here, cell-wise mass projection might lead to a
     * value that differs from the algebraic relationship expressed by our
     * affine constraints. Thus, we first compute the defect and then we
     * redistribute it to all degrees of freedom on the constraint line.
     */

    for (const auto &line : affine_constraints.get_lines()) {
      const auto global_i = line.index;
      const auto local_i = scalar_partitioner->global_to_local(global_i);

      /* Only operate on a locally owned, constrained degree of freedom: */
      if (local_i >= n_locally_owned)
        continue;

      /* The result of the mass projection: */
      const auto m_i_star = projected_mass.local_element(local_i);
      const auto U_i_star = projected_state.get_tensor(local_i) / m_i_star;

      /* The value obtained from the affine constraints object: */
      state_type U_i_interp;
      for (const auto &[global_k, c_k] : line.entries) {
        const auto local_k = scalar_partitioner->global_to_local(global_k);
        U_i_interp += c_k * new_U.get_tensor(local_k);
      }

      /* And distribute the defect: */
      const auto defect = U_i_star - U_i_interp;
      for (const auto &[global_k, c_k] : line.entries) {
        const auto local_k = scalar_partitioner->global_to_local(global_k);
        const auto U_j = new_U.get_tensor(local_k);

        projected_state.add_tensor(c_k * m_i_star * (U_j + defect), local_k);
        projected_mass.local_element(local_k) += c_k * m_i_star;
      }
    }

    update_new_state_vector();

#ifdef DEBUG
    /*
     * Sanity check: Final masses must agree:
     */
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    for (unsigned int local_i = 0; local_i < n_locally_owned; ++local_i) {
      const auto global_i = scalar_partitioner->local_to_global(local_i);
      if (affine_constraints.is_constrained(global_i))
        continue;

      const auto m_i = projected_mass.local_element(local_i);
      const auto m_i_reference = lumped_mass_matrix.local_element(local_i);
      Assert(std::abs(m_i - m_i_reference) < 1.e-10,
             dealii::ExcMessage(
                 "SolutionTransfer::projection(): something went wrong. Final "
                 "masses do not agree with those computed in OfflineData."));
    }
#endif
#endif
  }
} // namespace ryujin
