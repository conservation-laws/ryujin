//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/computing_timer.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/simd.h>
#include <ryujin/scalar_diffusion/parabolic_module.h>

#include <deal.II/lac/solver_cg.h>

#include <algorithm>
#include <iterator>

namespace ryujin
{
  namespace ScalarDiffusion
  {
    using namespace dealii;

    template <int dim, typename Number>
    ParabolicModule<dim, Number>::ParabolicModule(
        const MPIEnsemble &mpi_ensemble,
        const OfflineData<dim, dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const ParabolicSystem &parabolic_system,
        const InitialValues &initial_values,
        const std::string &subsection /*= "ParabolicModule"*/)
        : Interface::ParabolicModule<dim, Number>(subsection)
        , mpi_ensemble_(mpi_ensemble)
        , hyperbolic_system_(&hyperbolic_system)
        , parabolic_system_(&parabolic_system)
        , offline_data_(&offline_data)
        , initial_values_(&initial_values)
        , n_iterations_(0.)
    {
      tolerance_ = Number(1.0e-12);
      this->add_parameter("tolerance",
                          tolerance_,
                          "Relative tolerance (with respect to the l_2 norm "
                          "of the right hand side) of the linear solver");
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare()
    {
      RYUJIN_DEBUG_ANNOUNCE();

      const auto &discretization = offline_data_->discretization();
      AssertThrow(discretization.ansatz() == Ansatz::cg_q1,
                  dealii::ExcMessage("The scalar diffusion module currently "
                                     "only supports cG Q1 finite elements."));

      AssertThrow(!discretization.dof_handler().has_hp_capabilities(),
                  dealii::ExcMessage(
                      "The scalar diffusion module currently does not "
                      "support DoFHandlers set up with hp capabilities."));

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim, Number>::AdditionalData::none;

      matrix_free_.reinit(discretization.mapping(),
                          discretization.dof_handler(),
                          offline_data_->affine_constraints(),
                          discretization.quadrature_1d(),
                          additional_data);

      const auto &scalar_partitioner =
          matrix_free_.get_dof_info(0).vector_partitioner;

      diffusivity_.reinit(scalar_partitioner);
      right_hand_side_.reinit(scalar_partitioner);
      solution_.reinit(scalar_partitioner);
      temp_.reinit(scalar_partitioner);
      temp_stiffness_.reinit(scalar_partitioner);
    }


    template <int dim, typename Number>
    Number ParabolicModule<dim, Number>::backward_euler_step(
        const StateVector<Number> &old_state_vector,
        StageVectors stage_state_vectors,
        StageWeights stage_weights_explicit,
        StageWeights stage_weights_implicit,
        StateVector<Number> &new_state_vector,
        Number /*t*/,
        Number tau,
        Number /*tau_max*/) const
    {
      RYUJIN_DEBUG_ANNOUNCE();

      AssertThrow(tau != Number(0.),
                  dealii::ExcMessage("A parabolic module cannot choose the "
                                     "time step size: tau must be nonzero."));

      const auto stages = stage_state_vectors.size();
      AssertDimension(stage_weights_explicit.size(), stages);
      AssertDimension(stage_weights_implicit.size(), stages + 1);

      using HostSpace = dealii::MemorySpace::Host;
      const auto &lumped_mass_matrix =
          offline_data_->lumped_mass_matrix().deal_ii_vector();
      const unsigned int n_owned = offline_data_->n_locally_owned();

      /*
       * Step 1: Assemble the right hand side,
       *
       *   M W - tau sum_s { w^e_s K(k(U^s)) U^s
       *                     + (w^i_s - w^e_s) K(k(U^n)) U^s },
       *
       * and initialize the operator with the diffusivity k(U^n).
       */
      {
        ComputingTimer::Scope scope("time step [P] 1 - assemble rhs");

        /*
         * The sum of all contributions that enter with the diffusivity
         * k(U^n). For a linear diffusivity this is the case for all
         * contributions and the explicit weights drop out.
         */
        const bool is_linear = parabolic_system_->is_linear();

        right_hand_side_ = Number(0.);
        temp_ = Number(0.);

        auto weight_explicit = stage_weights_explicit.begin();
        auto weight_implicit = stage_weights_implicit.begin();
        for (const auto &stage_state_vector : stage_state_vectors) {
          const Number w_e = is_linear ? Number(0.) : *weight_explicit++;
          const Number w_i = *weight_implicit++;

          extract(stage_state_vector.get(), solution_);
          temp_.add(w_i - w_e, solution_);

          if (w_e != Number(0.)) {
            reinit_operator(solution_);
            diffusion_operator_.apply_stiffness(temp_stiffness_, solution_);
            right_hand_side_.add(-tau * w_e, temp_stiffness_);
          }
        }

        /* The state U^n at the beginning of the time step: */
        const auto &U_n =
            stages > 0 ? stage_state_vectors.begin()->get() : old_state_vector;
        extract(U_n, solution_);
        reinit_operator(solution_);

        if (stages > 0) {
          diffusion_operator_.apply_stiffness(temp_stiffness_, temp_);
          right_hand_side_.add(-tau, temp_stiffness_);
        }

        /* Add M W and use W as initial guess: */

        extract(old_state_vector, solution_);

        const auto body = [&](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);
          const auto m_i = read_entry<T>(lumped_mass_matrix, i);
          const auto w_i = read_entry<T>(solution_, i);
          const auto rhs_i = read_entry<T>(right_hand_side_, i);
          write_entry<T>(right_hand_side_, rhs_i + m_i * w_i, i);
        };

        cpu_simd_loop<Number>(
            "time_step_parabolic_1", body, 0, n_owned, n_owned);
      }

      /*
       * Step 2: Solve the linear system and update the state.
       */
      {
        ComputingTimer::Scope scope("time step [P] 2 - solve");

        /* The last implicit weight is the weight of the new state: */
        const Number weight = *std::prev(stage_weights_implicit.end());

        solve(tau * weight);
      }

      copy_to_memory_space<HostSpace>(old_state_vector, StateVectorPart::state);
      move_to_memory_space<HostSpace>(new_state_vector, StateVectorPart::state);

      const auto new_U_view =
          new_state_vector.state.template view<problem_dimension>();
      const auto sparsity_simd_view =
          offline_data_->sparsity_pattern_simd().view();
      const unsigned int n_internal = offline_data_->n_locally_internal();

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);

        /* Skip constrained degrees of freedom: */
        if (sparsity_simd_view.row_length(i) == 1)
          return;

        dealii::Tensor<1, problem_dimension, T> U_i;
        U_i[0] = read_entry<T>(solution_, i);
        new_U_view.template write_tensor<T>(U_i, i);
      };

      cpu_simd_loop<Number>(
          "time_step_parabolic_2", body, 0, n_internal, n_owned);

      new_U_view.update_ghost_values();

      return tau;
    }


    template <int dim, typename Number>
    Number ParabolicModule<dim, Number>::crank_nicolson_step(
        const StateVector<Number> &old_state_vector,
        StateVector<Number> &new_state_vector,
        Number /*t*/,
        Number tau,
        Number /*tau_max*/) const
    {
      RYUJIN_DEBUG_ANNOUNCE();

      AssertThrow(tau != Number(0.),
                  dealii::ExcMessage("A parabolic module cannot choose the "
                                     "time step size: tau must be nonzero."));

      using HostSpace = dealii::MemorySpace::Host;
      const auto &lumped_mass_matrix =
          offline_data_->lumped_mass_matrix().deal_ii_vector();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const unsigned int n_internal = offline_data_->n_locally_internal();

      /*
       * Crank-Nicolson (the midpoint rule): perform a backward Euler step
       * to the half time step and extrapolate.
       */
      {
        ComputingTimer::Scope scope("time step [P] 1 - assemble rhs");

        extract(old_state_vector, temp_);
        reinit_operator(temp_);

        const auto body = [&](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);
          const auto m_i = read_entry<T>(lumped_mass_matrix, i);
          const auto u_i = read_entry<T>(temp_, i);
          write_entry<T>(right_hand_side_, m_i * u_i, i);
          write_entry<T>(solution_, u_i, i);
        };

        cpu_simd_loop<Number>(
            "time_step_parabolic_1", body, 0, n_owned, n_owned);
      }

      {
        ComputingTimer::Scope scope("time step [P] 2 - solve");

        solve(Number(0.5) * tau);

        if (!parabolic_system_->is_linear()) {
          /* Repeat with the diffusivity of the predicted midpoint value: */
          reinit_operator(solution_);
          solve(Number(0.5) * tau);
        }
      }

      copy_to_memory_space<HostSpace>(old_state_vector, StateVectorPart::state);
      move_to_memory_space<HostSpace>(new_state_vector, StateVectorPart::state);

      const auto new_U_view =
          new_state_vector.state.template view<problem_dimension>();
      const auto sparsity_simd_view =
          offline_data_->sparsity_pattern_simd().view();

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);

        /* Skip constrained degrees of freedom: */
        if (sparsity_simd_view.row_length(i) == 1)
          return;

        dealii::Tensor<1, problem_dimension, T> U_i;
        U_i[0] = T(2.) * read_entry<T>(solution_, i) - read_entry<T>(temp_, i);
        new_U_view.template write_tensor<T>(U_i, i);
      };

      cpu_simd_loop<Number>(
          "time_step_parabolic_2", body, 0, n_internal, n_owned);

      new_U_view.update_ghost_values();

      return tau;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::print_solver_statistics(
        std::ostream &output) const
    {
      output << "        [ " << std::setprecision(2) << std::fixed
             << n_iterations_ << " CG diffusion ]" << std::endl;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::extract(
        const StateVector<Number> &state_vector,
        ScalarHostVector<Number> &dst) const
    {
      using HostSpace = dealii::MemorySpace::Host;
      copy_to_memory_space<HostSpace>(state_vector, StateVectorPart::state);

      const auto U_view = state_vector.state.template view<problem_dimension>();
      const unsigned int n_owned = offline_data_->n_locally_owned();

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        const auto view = hyperbolic_system_->template view<dim, T>();
        const auto U_i = U_view.template read_tensor<T>(i);
        write_entry<T>(dst, view.state(U_i), i);
      };

      cpu_simd_loop<Number>("", body, 0, n_owned, n_owned);

      /*
       * Zero out constrained degrees of freedom due to hanging nodes and
       * periodic boundary conditions. These are enforced by modifying the
       * stencil - consequently we have to remove constrained dofs from
       * the linear system.
       */
      offline_data_->affine_constraints().set_zero(dst);
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::reinit_operator(
        const ScalarHostVector<Number> &u) const
    {
      const auto &lumped_mass_matrix =
          offline_data_->lumped_mass_matrix().deal_ii_vector();
      const unsigned int n_owned = offline_data_->n_locally_owned();

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        const auto u_i = read_entry<T>(u, i);
        write_entry<T>(diffusivity_, parabolic_system_->diffusivity(u_i), i);
      };

      cpu_simd_loop<Number>("", body, 0, n_owned, n_owned);

      /*
       * The finite element interpolant of the diffusivity has to satisfy
       * the hanging node and periodicity constraints:
       */
      offline_data_->affine_constraints().distribute(diffusivity_);

      diffusion_operator_.reinit(
          matrix_free_, lumped_mass_matrix, diffusivity_);
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::solve(const Number sigma) const
    {
      const auto &affine_constraints = offline_data_->affine_constraints();
      affine_constraints.set_zero(right_hand_side_);
      affine_constraints.set_zero(solution_);

      diffusion_operator_.set_factor(sigma);

      DiagonalMatrix<ScalarHostVector<Number>> jacobi;
      diffusion_operator_.compute_diagonal(jacobi);

      if (sigma == Number(0.)) {
        jacobi.vmult(solution_, right_hand_side_);
        return;
      }

      SolverControl solver_control(10000,
                                   tolerance_ * right_hand_side_.l2_norm());
      SolverCG<ScalarHostVector<Number>> solver(solver_control);
      solver.solve(diffusion_operator_, solution_, right_hand_side_, jacobi);

      /* update exponential moving average */
      n_iterations_ = 0.9 * n_iterations_ + 0.1 * solver_control.last_step();
    }
  } // namespace ScalarDiffusion
} // namespace ryujin
