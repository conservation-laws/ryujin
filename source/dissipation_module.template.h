//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "dissipation_module.h"
#include "introspection.h"
#include "openmp.h"
#include "scope.h"
#include "simd.h"

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <atomic>

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  DissipationModule<dim, Number>::DissipationModule(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const ParabolicSystem &parabolic_system,
      const OfflineData<dim, Number> &offline_data,
      const InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "DissipationModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , parabolic_system_(&parabolic_system)
      , offline_data_(&offline_data)
      , initial_values_(&initial_values)
      , n_warnings_(0)
      , theta_(0.5)
  {
    tolerance_ = Number(1.0e-12);
    add_parameter("tolerance", tolerance_, "Tolerance for linear solvers");

    tolerance_linfty_norm_ = false;
    add_parameter("tolerance linfty norm",
                  tolerance_linfty_norm_,
                  "Use the l_infty norm instead of the l_2 norm for the "
                  "stopping criterion");

    std::fill_n(use_gmg_.begin(), n_implicit_systems, false);
    add_parameter("multigrid",
                  use_gmg_,
                  "Use a geometric multigrid for the specified linear systems");

    enter_subsection("Multigrid");

    std::fill_n(gmg_max_iter_.begin(), n_implicit_systems, 12); // 8
    add_parameter("max iter",
                  gmg_max_iter_,
                  "Maximal number of CG iterations with GMG smoother before "
                  "falling back to diagonal preconditiong");

    std::fill_n(gmg_smoother_range_.begin(), n_implicit_systems, 8.); // 15.
    add_parameter("chebyshev range",
                  gmg_smoother_range_,
                  "Chebyshev smoother: eigenvalue range parameter");

    std::fill_n(gmg_smoother_max_eig_.begin(), n_implicit_systems, 2.0);
    add_parameter("chebyshev max eig",
                  gmg_smoother_max_eig_,
                  "Chebyshev smoother: maximal eigenvalue");

    std::fill_n(gmg_smoother_degree_.begin(), n_implicit_systems, 3);
    add_parameter("chebyshev degree",
                  gmg_smoother_degree_,
                  "Chebyshev smoother: degree");

    std::fill_n(gmg_smoother_n_cg_iter_.begin(), n_implicit_systems, 10);
    add_parameter("chebyshev cg iter",
                  gmg_smoother_n_cg_iter_,
                  "Chebyshev smoother: number of CG iterations to approximate "
                  "eigenvalue");

    std::fill_n(gmg_min_level_.begin(), n_implicit_systems, 0);
    add_parameter("min level",
                  gmg_min_level_,
                  "Minimal mesh level to be visited in the geometric multigrid "
                  "cycle where the coarse grid solver (Chebyshev) is called");

    leave_subsection();
  }


  template <int dim, typename Number>
  void DissipationModule<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::prepare()" << std::endl;
#endif

    /* Nothing to do: */
    if constexpr (n_implicit_systems == 0)
      return;

    /* Initialize matrix free context, diagonal preconditioner, and vectors: */

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;

    matrix_free_.reinit(offline_data_->discretization().mapping(),
                        offline_data_->dof_handler(),
                        offline_data_->affine_constraints(),
                        offline_data_->discretization().quadrature_1d(),
                        additional_data);

    const auto &scalar_partitioner =
        matrix_free_.get_dof_info(0).vector_partitioner;

    diagonal_preconditioner_.reinit(scalar_partitioner);

    static_assert(n_implicit_systems <= 2, "not implemented");

    if constexpr (n_implicit_systems >= 1) {
      dissipation_operator_[0].reinit(
          matrix_free_,
          scalar_partitioner,
          [this](block_vector_type &dst, const block_vector_type &src) mutable
          -> void { this->apply_boundary_action(dst, src); });
    }

    if constexpr (n_implicit_systems >= 2) {
      dissipation_operator_[1].reinit(
          matrix_free_,
          scalar_partitioner,
          [this](block_vector_type &dst, const block_vector_type &src) mutable
          -> void { this->apply_boundary_action(dst, src); });
    }

    solution_.reinit(parabolic_problem_dimension);
    right_hand_side_.reinit(parabolic_problem_dimension);
    for (unsigned int i = 0; i < parabolic_problem_dimension; ++i) {
      solution_.block(i).reinit(scalar_partitioner);
      right_hand_side_.block(i).reinit(scalar_partitioner);
    }

    /* Initialize multigrid: */

    if (!std::any_of(
            use_gmg_.begin(), use_gmg_.end(), [](auto value) { return value; }))
      return;

    __builtin_trap(); // FIXME: GMG
  }


  template <int dim, typename Number>
  Number DissipationModule<dim, Number>::step(vector_type &U,
                                              Number t,
                                              Number tau,
                                              unsigned int /*cycle*/) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "DissipationModule<dim, Number>::step()" << std::endl;
#endif

    /* Nothing to do: */
    if constexpr (n_implicit_systems == 0)
      return tau;

#ifdef DEBUG_OUTPUT
    std::cout << "        perform time-step with tau = " << tau << std::endl;
    std::cout << "        (theta: " << theta_ << ")" << std::endl;
#endif

    CALLGRIND_START_INSTRUMENTATION

    using VA = VectorizedArray<Number>;

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &affine_constraints = offline_data_->affine_constraints();

    /* Index ranges for the iteration over the sparsity pattern : */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    /*
     * Step 0:
     *
     * Build right hand sides and initialize solution vectors.
     */
    {
      Scope scope(computing_timer_,
                  "time step [N] 0 - initialize scaling, vectors and rhs");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_n_0");

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          const auto m_i = load_value<T>(lumped_mass_matrix, i);
          const auto U_i = U.template get_tensor<T>(i);

          store_value<T>(diagonal_preconditioner_.scaling_vector(),
                         parabolic_system_->compute_diagonal_scaling(U_i, m_i),
                         i);

          for (auto &it : dissipation_operator_)
            store_value<T>(it.diagonal_action(),
                           parabolic_system_->compute_diagonal_action(U_i, m_i),
                           i);

          const auto &[V_i, rhs_i] =
              parabolic_system_->compute_parabolic_state_and_rhs(U_i, m_i);
          for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
            store_value<T>(solution_.block(d), V_i[d], i);
            store_value<T>(right_hand_side_.block(d), rhs_i[d], i);
          }
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP("time_step_n_0");
      RYUJIN_PARALLEL_REGION_END

      /*
       * Set up "strongly enforced" boundary conditions that are not stored
       * in the AffineConstraints map. In this case we enforce boundary
       * values by imposing them strongly in the iteration by setting the
       * initial vector and the right hand side to the right value:
       */
      set_boundary_values(t + theta_ * tau);

      /*
       * Zero out constrained degrees of freedom due to periodic boundary
       * conditions. These boundary conditions are enforced by modifying
       * the stencil - consequently we have to remove constrained dofs from
       * the linear system.
       */
      affine_constraints.set_zero(diagonal_preconditioner_.scaling_vector());
      for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
        affine_constraints.set_zero(solution_.block(d));
        affine_constraints.set_zero(right_hand_side_.block(d));
      }
    }

    for (unsigned int pass = 0; pass < n_implicit_systems; ++pass) {
      bool last_round = (pass + 1 == n_implicit_systems);

      /*
       * Step 1, 3: Solve linear system.
       */
      {
        std::string step_no = std::to_string(1 + 2 * pass);
        std::string description = //
            "time step [N] " + step_no + " - solve parabolic (sub)system [" +
            parabolic_system_->implicit_system_names[pass] + "]";
        Scope scope(computing_timer_, description);

        LIKWID_MARKER_START(("time_step_n_" + step_no).c_str());

        const auto tolerance =
            (tolerance_linfty_norm_ ? right_hand_side_.linfty_norm()
                                    : right_hand_side_.l2_norm()) *
            tolerance_;

        try {
          throw SolverControl::NoConvergence(0, 0.);

          // FIXME: GMG

        } catch (SolverControl::NoConvergence &) {

          SolverControl solver_control(1000, tolerance);
          SolverCG<block_vector_type> solver(solver_control);
          solver.solve(dissipation_operator_[pass],
                       solution_,
                       right_hand_side_,
                       diagonal_preconditioner_);

          /* update exponential moving average, counting also GMG iterations */
          n_iterations_[pass] *= 0.9;
          n_iterations_[pass] +=
              0.1 * (use_gmg_[pass] ? gmg_max_iter_[pass] : 0) +
              0.1 * solver_control.last_step();
        }

        LIKWID_MARKER_STOP(("time_step_n_" + step_no).c_str());
      }

      /*
       * Step 2: Update right hand side.
       */
      if (!last_round) {
        std::string step_no = std::to_string(2 + 2 * pass);
        std::string description = //
            "time step [N] " + step_no + " - update right hand side [" +
            parabolic_system_->implicit_system_names[pass] + "]";
        Scope scope(computing_timer_, description);

        LIKWID_MARKER_START(("time_step_n_" + step_no).c_str());

        // FIXME

        LIKWID_MARKER_STOP(("time_step_n_" + step_no).c_str());
      }
    }

    /*
     * Step 2 * n_implicit_systems: Write back vectors
     */
    {
      std::string step_no = std::to_string(2 * n_implicit_systems);
      Scope scope(computing_timer_,
                  "time step [N] " + step_no + " - write back vectors");

      const auto alpha = Number(1.) / theta_;

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START(("time_step_n_" + step_no).c_str());

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          auto U_i = U.template get_tensor<T>(i);

          auto V_i = parabolic_system_->to_parabolic_state(U_i);
          V_i *= (Number(1.) - alpha);
          for (unsigned int d = 0; d < parabolic_problem_dimension; ++d)
            V_i[d] += alpha * load_value<T>(solution_.block(d), i);

          U_i = parabolic_system_->from_parabolic_state(U_i, V_i);
          U.template write_tensor<T>(U_i, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP(("time_step_n_" + step_no).c_str());
      RYUJIN_PARALLEL_REGION_END
    }

    return tau;
  }


  template <int dim, typename Number>
  void DissipationModule<dim, Number>::set_boundary_values(Number t) const
  {
    const auto &boundary_map = offline_data_->boundary_map();

    for (auto entry : boundary_map) {
      const auto i = entry.first;
      const auto &[normal, normal_mass, boundary_mass, id, position] =
          entry.second;

      if (id == Boundary::do_nothing)
        continue;

      parabolic_state_type V_i, rhs_i;
      for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
        V_i[d] = load_value<Number>(solution_.block(d), i);
        rhs_i[d] = load_value<Number>(right_hand_side_.block(d), i);
      }

      /* Use a lambda to avoid computing unnecessary state values */
      auto get_dirichlet_data = [position = position, t = t, this]() {
        return initial_values_->initial_state(position, t);
      };

      V_i = parabolic_system_->apply_boundary_conditions(
          id, V_i, normal, get_dirichlet_data);
      rhs_i = parabolic_system_->apply_boundary_conditions(
          id, rhs_i, normal, get_dirichlet_data);

      for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
        store_value<Number>(solution_.block(d), V_i[d], i);
        store_value<Number>(right_hand_side_.block(d), rhs_i[d], i);
      }
    }
  }


  template <int dim, typename Number>
  void DissipationModule<dim, Number>::apply_boundary_action(
      block_vector_type &dst, const block_vector_type &src) const
  {
    const auto &boundary_map = offline_data_->boundary_map();

    for (auto entry : boundary_map) {
      const auto i = entry.first;
      const auto &[normal, normal_mass, boundary_mass, id, position] =
          entry.second;

      if (id == Boundary::do_nothing)
        continue;

      parabolic_state_type dst_i, src_i;
      for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
        src_i[d] = load_value<Number>(src.block(d), i);
        dst_i[d] = load_value<Number>(dst.block(d), i);
      }

      dst_i =
          parabolic_system_->apply_boundary_action(id, dst_i, normal, src_i);

      for (unsigned int d = 0; d < parabolic_problem_dimension; ++d) {
        store_value<Number>(dst.block(d), dst_i[d], i);
      }
    }
  }

} // namespace ryujin
