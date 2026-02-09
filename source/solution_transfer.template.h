//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception or LGPL-2.1-or-later
// Copyright (C) 2024 - 2025 by the ryujin authors
//

#pragma once


#include "loop.h"
#include "solution_transfer.h"
#if DEAL_II_VERSION_GTE(9, 6, 0)
#include "tensor_product_point_kernels.h"
#endif

#include <deal.II/base/exceptions.h>
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
    const auto &discretization = offline_data_->discretization();
    auto &triangulation = *discretization.triangulation_; /* writable */

    Assert(handle_ == dealii::numbers::invalid_unsigned_int,
           dealii::ExcMessage(
               "You can only add one solution per SolutionTransfer object."));

    /*
     * -----------------------------------------------------------------------
     * Cell-level projection to parent cells and packing data:
     * -----------------------------------------------------------------------
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

          const auto &U = std::get<0>(old_state_vector);
          /* precomputed needs to be valid for bounds computation */
          const auto &precomputed = std::get<1>(old_state_vector);

          using Limiter = typename Description::template Limiter<dim, Number>;
          const Limiter limiter(
              *hyperbolic_system_, limiter_parameters_, precomputed);

          /*
           * Collect state values for packing:
           */

          const auto n_dofs_per_cell = dof_cell->get_fe().n_dofs_per_cell();
          std::vector<state_type> state_values(n_dofs_per_cell);

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

            std::transform(
                std::begin(dof_indices),
                std::end(dof_indices),
                std::begin(state_values),
                [&](const auto global_i) { return get_tensor(U, global_i); });
          } break;

          case dealii::CellStatus::children_will_be_coarsened: {
            /*
             * We need to project values from the active child cells up to
             * the present parent cell that will become active after
             * coarsening.
             */

            Assert(dof_cell->has_children(), dealii::ExcInternalError());

            const auto &discretization = offline_data_->discretization();
            const auto index = dof_cell->active_fe_index();
            const auto &finite_element = discretization.finite_element()[index];
            const auto &mapping = discretization.mapping()[index];
            const auto &quadrature = discretization.quadrature()[index];

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

            /* Step 1: build up right hand side by iterating over children: */

            std::vector<state_type> state_values_quad(quadrature.size());
            std::vector<state_type> local_rhs(n_dofs_per_cell);

            std::vector<dealii::types::global_dof_index> dof_indices(
                n_dofs_per_cell);

            Bounds bounds;

            for (unsigned int child = 0; child < dof_cell->n_children();
                 ++child) {
              const auto child_cell = dof_cell->child(child);

              Assert(child_cell->is_active(), dealii::ExcInternalError());
              Assert(dof_cell->active_fe_index() ==
                         child_cell->active_fe_index(),
                     dealii::ExcMessage("SolutionTransfer: projection between "
                                        "different FE space is not set up."));

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

            /* Step 2: construct inverse mass matrices: */

            fe_values.reinit(dof_cell);

            dealii::FullMatrix<double> mass_inverse(n_dofs_per_cell,
                                                    n_dofs_per_cell);
            dealii::Vector<double> lumped_mass(n_dofs_per_cell);
            dealii::Vector<double> lumped_mass_inverse(n_dofs_per_cell);

            auto total_mass = Number(0.);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                double sum = 0;
                for (unsigned int q = 0; q < quadrature.size(); ++q)
                  sum += fe_values.shape_value(i, q) *
                         fe_values.shape_value(j, q) * fe_values.JxW(q);
                mass_inverse(i, j) = sum;
                lumped_mass(i) += sum;
              }
              lumped_mass_inverse(i) = Number(1.) / lumped_mass(i);
              total_mass += lumped_mass(i);
            }
            mass_inverse.gauss_jordan();

            /* Step 3: compute low-order update and P_ij matrix: */

            bounds = limiter.fully_relax_bounds(bounds, total_mass);

            std::vector<state_type> pij_matrix(n_dofs_per_cell *
                                               n_dofs_per_cell);
            dealii::FullMatrix<Number> lij_matrix(n_dofs_per_cell,
                                                  n_dofs_per_cell);

            const auto kappa_inverse = Number(n_dofs_per_cell);
            const auto kappa = Number(1.) / kappa_inverse;

            for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
              const state_type U_i = lumped_mass_inverse(i) * local_rhs[i];
              state_values[i] = U_i;

              for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                const auto kronecker_ij = Number(i == j ? 1. : 0.);
                const auto b_ij =
                    lumped_mass(i) * mass_inverse(i, j) - kronecker_ij;
                const auto b_ji =
                    lumped_mass(j) * mass_inverse(i, j) - kronecker_ij;
                const auto P_ij = kappa_inverse * lumped_mass_inverse(i) *
                                  (b_ij * local_rhs[j] - b_ji * local_rhs[i]);
                pij_matrix[n_dofs_per_cell * i + j] = P_ij;
              }
            }

            /* Step 4: compute l_ij matrix and apply limited update: */

            const auto n_iterations = limiter_parameters_.iterations();
            for (unsigned int pass = 0; pass < n_iterations; ++pass) {

              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                const auto &U_i = state_values[i];

                for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                  const auto &P_ij = pij_matrix[n_dofs_per_cell * i + j];
                  const auto &[l_ij, check] = limiter.limit(bounds, U_i, P_ij);
                  lij_matrix(i, j) = l_ij;
                }
              }

              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                auto &U_i = state_values[i];

                for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                  const auto l_ij =
                      std::min(lij_matrix(i, j), lij_matrix(j, i));
                  auto &P_ij = pij_matrix[n_dofs_per_cell * i + j];
                  U_i += kappa * l_ij * P_ij;
                  P_ij -= l_ij * P_ij;
                }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
                const auto view =
                    hyperbolic_system_->template view<dim, Number>();
                AssertThrow(
                    view.is_admissible(U_i),
                    dealii::ExcMessage(
                        "Error: inadmissible state encountered in "
                        "register_data_attach / children_will_be_coarsened"));
#endif
              }
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

    const auto &discretization = offline_data_->discretization();
    auto &triangulation = *discretization.triangulation_; /* writable */

    Assert(
        handle_ != dealii::numbers::invalid_unsigned_int,
        dealii::ExcMessage(
            "Cannot project() a state vector without valid handle. "
            "prepare_projection() or set_handle() have to be called first."));

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    const auto &affine_constraints = offline_data_->affine_constraints();
    const auto n_locally_owned = offline_data_->n_locally_owned();


    ScalarVector projected_mass;
    projected_mass.reinit(offline_data_->scalar_partitioner());
    HyperbolicVector projected_state;
    projected_state.reinit(offline_data_->hyperbolic_vector_partitioner());

    /*
     * We only need to construct entries in a pik_matrix for a small subset
     * of affected degrees of freedom for which we have to construct the
     * entire pik_matrix first for the limiting process (in contrast to the
     * entirely cell-local limiting done before). Let's simply use a map.
     */
    std::map<std::tuple<unsigned int /*i*/, unsigned int /*k*/>, state_type>
        pik_matrix;
    std::map<unsigned int /*i*/, Bounds> bounds_map;

    ScalarVector kappa;
    kappa.reinit(offline_data_->scalar_partitioner());

    /*
     * -----------------------------------------------------------------------
     * Unpacking data and cell-level interpolation/projection to child cells:
     * -----------------------------------------------------------------------
     */

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

          /*
           * Retrieve packed values and project onto cell:
           */

          const auto n_dofs_per_cell = dof_cell->get_fe().n_dofs_per_cell();
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
            const auto index = dof_cell->active_fe_index();
            const auto &finite_element = discretization.finite_element()[index];
            const auto &mapping = discretization.mapping()[index];
            const auto &quadrature = discretization.quadrature()[index];

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
            const auto index = dof_cell->active_fe_index();
            const auto &finite_element = discretization.finite_element()[index];
            const auto &mapping = discretization.mapping()[index];
            const auto &quadrature = discretization.quadrature()[index];

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

            dealii::FullMatrix<double> mass_inverse(n_dofs_per_cell,
                                                    n_dofs_per_cell);
            dealii::Vector<double> lumped_mass(n_dofs_per_cell);
            std::vector<state_type> local_rhs(n_dofs_per_cell);

            for (unsigned int child = 0; child < dof_cell->n_children();
                 ++child) {
              const auto child_cell = dof_cell->child(child);

              Assert(child_cell->is_active(), dealii::ExcInternalError());
              Assert(dof_cell->active_fe_index() ==
                         child_cell->active_fe_index(),
                     dealii::ExcMessage("SolutionTransfer: projection between "
                                        "different FE space is not set up."));

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

              mass_inverse = Number(0.);
              lumped_mass = Number(0.);
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                  double sum = 0;
                  for (unsigned int q = 0; q < quadrature.size(); ++q)
                    sum += fe_values.shape_value(i, q) *
                           fe_values.shape_value(j, q) * fe_values.JxW(q);
                  mass_inverse(i, j) = sum;
                  lumped_mass(i) += sum;
                }
              }
              mass_inverse.gauss_jordan();

              /* Step 3: compute high order update and write back: */

              for (unsigned int i = 0; i < n_dofs_per_cell; ++i) {
                state_type U_i;
                for (unsigned int j = 0; j < n_dofs_per_cell; ++j) {
                  U_i += mass_inverse(i, j) * local_rhs[j];
                }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
                const auto view =
                    hyperbolic_system_->template view<dim, Number>();
                AssertThrow(view.is_admissible(U_i),
                            dealii::ExcMessage(
                                "Error: inadmissible state encountered in "
                                "ready_to_unpack / cell_will_be_refined"));
#endif
                const auto global_i = dof_indices[i];
                add_tensor(projected_state, lumped_mass(i) * U_i, global_i);
                projected_mass(global_i) += lumped_mass(i);
              }
            } /*child*/

          } break;

          case dealii::CellStatus::cell_invalid:
            Assert(false, dealii::ExcInternalError());
            __builtin_trap();
            break;
          }
        });

    projected_mass.compress(dealii::VectorOperation::add);
    projected_state.compress(dealii::VectorOperation::add);

    /*
     * -----------------------------------------------------------------------
     * Redistribute masses to satisfy hanging-node constraints:
     *
     * Now redistribute the mass defect introduced by constrained degrees
     * of freedom. This mostly affects hanging nodes neighboring a
     * coarsened cell. Here, cell-wise mass projection might lead to a
     * value for a constrained degree of freedom that differs from the
     * algebraic relationship expressed by our affine constraints. Thus, we
     * first compute the defect and then we redistribute it to all degrees
     * of freedom on the constraint line.
     * -----------------------------------------------------------------------
     */

    auto &new_U = std::get<0>(new_state_vector);

    /*
     * A small lambda that takes the weighted average of all degrees of
     * freedom, and stores the result in new_U:
     */
    const auto update_new_state_vector = [&]() {
      for (unsigned int local_i = 0; local_i < n_locally_owned; ++local_i) {

        const auto mU_i = projected_state.get_tensor(local_i);
        const auto m_i = projected_mass.local_element(local_i);

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        const auto view = hyperbolic_system_->template view<dim, Number>();
        AssertThrow(
            view.is_admissible(mU_i / m_i),
            dealii::ExcMessage("Error: inadmissible state encountered in "
                               "update_new_state_vector()"));
#endif

        new_U.write_tensor(mU_i / m_i, local_i);
      }
      new_U.update_ghost_values();
    };

    update_new_state_vector();

    const auto &precomputed = std::get<1>(new_state_vector);

    /* The limiter requires valid precomputed values. Therefore, update: */
    const auto update_precomputed_values = [&]() {
      new_U.update_ghost_values();
      hyperbolic_system_->fill_precomputed_values(
          *offline_data_, new_state_vector, /*skip_constrainted_dofs*/ false);
      precomputed.update_ghost_values();
    };

    update_precomputed_values();

    using Limiter = typename Description::template Limiter<dim, Number>;
    const Limiter limiter(
        *hyperbolic_system_, limiter_parameters_, precomputed);

    /*
     * Step 1: compute low-order update P_ij matrix, and bounds:
     *
     * We compute limiter bounds as a single value over the constraint
     * line. This makes sense as we need to limit the update for each
     * affected (unconstrained) degree of freedom of a constraint line with
     * a single limiter value anyway to ensure mass conservation.
     * (Incidentally, this avoids having to update a global, distributed
     * bounds vector over all MPI ranks.)
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

      auto &bounds = bounds_map[local_i]; /* by reference */
      bounds = limiter.projection_bounds_from_state(local_i, U_i_star);

      /* The value obtained from the affine constraints object: */
      state_type U_i_interp;
      for (const auto &[global_k, c_k] : line.entries) {
        const auto local_k = scalar_partitioner->global_to_local(global_k);
        U_i_interp += c_k * new_U.get_tensor(local_k);
      }

      /* And redistribute low order update: */
      for (const auto &[global_k, c_k] : line.entries) {
        const auto local_k = scalar_partitioner->global_to_local(global_k);
        const auto U_k = new_U.get_tensor(local_k);

        const auto bounds_k =
            limiter.projection_bounds_from_state(local_k, U_k);
        bounds = limiter.combine_bounds(bounds, bounds_k);

        projected_state.add_tensor(c_k * m_i_star * U_i_star, local_k);
        projected_mass.local_element(local_k) += c_k * m_i_star;

        kappa.local_element(local_k) += Number(1.);
        pik_matrix[{local_i, local_k}] = c_k * m_i_star * (U_k - U_i_interp);
      }
    }

    /* Compress vectors, recalculate unconstrained states: */
    projected_mass.compress(dealii::VectorOperation::add);
    projected_state.compress(dealii::VectorOperation::add);
    kappa.compress(dealii::VectorOperation::add);
    update_new_state_vector();

    /* Redistribute ghost layer for masses and kappa: */
    projected_mass.update_ghost_values();
    kappa.update_ghost_values();

    /* Step 2: Apply limiter: */

    const auto n_iterations = limiter_parameters_.iterations();
    for (unsigned int pass = 0; pass < n_iterations; ++pass) {

      /* Update precomputed values for bounds correction: */
      update_precomputed_values();

      for (const auto &line : affine_constraints.get_lines()) {
        const auto global_i = line.index;
        const auto local_i = scalar_partitioner->global_to_local(global_i);

        /* Only operate on a locally owned, constrained degree of freedom: */
        if (local_i >= n_locally_owned)
          continue;

        /*
         * We are computing bounds only over a local constraint line
         * without recombining such bounds per (unconstrained) degree of
         * freedom globally. We avoid doing the latter because it would
         * require a custom "VectorOperation" invoking
         * Limiter::combine_bounds(), which we currently do not have at our
         * disposal.
         *
         * As a simple workaround we simply recompute bounds for the
         * constraint line after the low-order update and each limiter pass
         * and recombine those into the stored value.
         */

        auto &bounds = bounds_map[local_i]; /* by reference */
        auto total_mass = Number(0.);
        for (const auto &[global_k, c_k] : line.entries) {
          const auto local_k = scalar_partitioner->global_to_local(global_k);
          const auto U_k = new_U.get_tensor(local_k);
          const auto bounds_k =
              limiter.projection_bounds_from_state(local_k, U_k);
          bounds = limiter.combine_bounds(bounds, bounds_k);

          const auto m_k = projected_mass.local_element(local_k);
          total_mass += m_k;
        }

        auto l = Number(1.);

        /* Apply relaxation: */
        const auto relaxed_bounds =
            limiter.fully_relax_bounds(bounds, total_mass);

        /* Compute limiter values: */

        for (const auto &[global_k, c_k] : line.entries) {
          const auto local_k = scalar_partitioner->global_to_local(global_k);
          const auto kappa_k = kappa.local_element(local_k);
          const auto m_k = projected_mass.local_element(local_k);
          const auto U_k = new_U.get_tensor(local_k);
          const auto P_ik = pik_matrix[{local_i, local_k}] * kappa_k / m_k;

          const auto &[l_k, check] = limiter.limit(relaxed_bounds, U_k, P_ik);
          l = std::min(l, l_k);
        }

        /* Apply limiter values: */

        for (const auto &[global_k, c_k] : line.entries) {
          const auto local_k = scalar_partitioner->global_to_local(global_k);
          auto &mP_ik = pik_matrix[{local_i, local_k}];
          projected_state.add_tensor(l * mP_ik, local_k);
          mP_ik -= l * mP_ik;
        }
      }

      /* Compress state vector, recalculate unconstrained states: */
      projected_state.compress(dealii::VectorOperation::add);
      update_new_state_vector();
    }

    /* Zero out constrained degrees of freedom: */
    for (unsigned int local_i = 0; local_i < n_locally_owned; ++local_i) {
      const auto global_i = scalar_partitioner->local_to_global(local_i);
      if (affine_constraints.is_constrained(global_i))
        new_U.write_tensor(state_type{}, local_i);
    }
    new_U.update_ghost_values();

#ifdef DEBUG_SYMMETRY_CHECK
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
