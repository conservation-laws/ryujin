//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/computing_timer.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/simd.h>
#include <ryujin/navier_stokes/parabolic_module.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_transfer.templates.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <atomic>

namespace ryujin
{
  namespace NavierStokes
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
        , cycle_(0)
        , n_iterations_velocity_(0.)
        , n_iterations_internal_energy_(0.)
    {
      tolerance_ = Number(1.0e-12);
      this->add_parameter(
          "tolerance", tolerance_, "Tolerance for linear solvers");

      tolerance_linfty_norm_ = false;
      this->add_parameter(
          "tolerance linfty norm",
          tolerance_linfty_norm_,
          "Use the l_infty norm instead of the l_2 norm for the "
          "stopping criterion");

      use_gmg_velocity_ = false;
      this->add_parameter("multigrid velocity",
                          use_gmg_velocity_,
                          "Use geometric multigrid for velocity component");

      gmg_smoother_max_eig_vel_ = 2.0;
      this->add_parameter("multigrid velocity - chebyshev max eig",
                          gmg_smoother_max_eig_vel_,
                          "Chebyshev smoother: maximal eigenvalue");

      gmg_smoother_range_vel_ = 8.;
      this->add_parameter("multigrid velocity - chebyshev range",
                          gmg_smoother_range_vel_,
                          "Chebyshev smoother: eigenvalue range parameter");

      gmg_max_iter_vel_ = 12;
      this->add_parameter("multigrid velocity - max iter",
                          gmg_max_iter_vel_,
                          "Maximal number of CG iterations with GMG smoother");

      use_gmg_internal_energy_ = false;
      this->add_parameter(
          "multigrid energy",
          use_gmg_internal_energy_,
          "Use geometric multigrid for internal energy component");

      gmg_smoother_max_eig_en_ = 2.0;
      this->add_parameter("multigrid energy - chebyshev max eig",
                          gmg_smoother_max_eig_en_,
                          "Chebyshev smoother: maximal eigenvalue");

      gmg_smoother_range_en_ = 15.;
      this->add_parameter("multigrid energy - chebyshev range",
                          gmg_smoother_range_en_,
                          "Chebyshev smoother: eigenvalue range parameter");

      gmg_max_iter_en_ = 15;
      this->add_parameter("multigrid energy - max iter",
                          gmg_max_iter_en_,
                          "Maximal number of CG iterations with GMG smoother");

      gmg_smoother_n_cg_iter_ = 10;
      this->add_parameter(
          "multigrid - chebyshev cg iter",
          gmg_smoother_n_cg_iter_,
          "Chebyshev smoother: number of CG iterations to approximate "
          "eigenvalue");

      gmg_smoother_degree_ = 3;
      this->add_parameter("multigrid - chebyshev degree",
                          gmg_smoother_degree_,
                          "Chebyshev smoother: degree");

      gmg_min_level_ = 0;
      this->add_parameter(
          "multigrid - min level",
          gmg_min_level_,
          "Minimal mesh level to be visited in the geometric multigrid "
          "cycle where the coarse grid solver (Chebyshev) is called");

      gmg_reinitialization_interval_ = 1;
      this->add_parameter(
          "multigrid - reinitialization interval",
          gmg_reinitialization_interval_,
          "Reinitialize the level matrices and smoothers of the geometric "
          "multigrid every n-th time step");
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare()
    {
      RYUJIN_DEBUG_ANNOUNCE();
      /*
       * The cycle_ variabe is only used for gmg reinitialization, simply
       * reset it to zero on prepare().
       */
      cycle_ = 0;

      AssertThrow(gmg_reinitialization_interval_ > 0,
                  dealii::ExcMessage("The multigrid reinitialization "
                                     "interval must be positive."));

      const auto &discretization = offline_data_->discretization();
      AssertThrow(discretization.ansatz() == Ansatz::cg_q1,
                  dealii::ExcMessage("The Navier-Stokes module currently only "
                                     "supports cG Q1 finite elements."));

      AssertThrow(!discretization.dof_handler().has_hp_capabilities(),
                  dealii::ExcMessage(
                      "The Navier-Stokes module currently does not support "
                      "DoFHandlers set up with hp capabilities."));

      /* Initialize vectors: */

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

      for (auto *it : {&velocity_,
                       &velocity_rhs_,
                       &stage_velocity_,
                       &stage_stiffness_,
                       &stage_force_}) {
        it->reinit(dim);
        for (unsigned int i = 0; i < dim; ++i)
          it->block(i).reinit(scalar_partitioner);
        it->collect_sizes();
      }

      for (auto *it : {&internal_energy_,
                       &internal_energy_rhs_,
                       &density_,
                       &stage_work_,
                       &stage_internal_energy_,
                       &stage_diffusion_,
                       &stage_energy_rhs_})
        it->reinit(scalar_partitioner);

      /* Initialize multigrid: */

      if (!use_gmg_velocity_ && !use_gmg_internal_energy_)
        return;

      const unsigned int n_levels =
          discretization.triangulation().n_global_levels();
      const unsigned int min_level = std::min(gmg_min_level_, n_levels - 1);
      MGLevelObject<IndexSet> relevant_sets(0, n_levels - 1);
      for (unsigned int level = 0; level < n_levels; ++level)
        relevant_sets[level] =
            dealii::DoFTools::extract_locally_relevant_level_dofs(
                discretization.dof_handler(), level);
      mg_constrained_dofs_.initialize(discretization.dof_handler(),
                                      relevant_sets);
      std::set<types::boundary_id> boundary_ids;
      boundary_ids.insert(Boundary::dirichlet);
      boundary_ids.insert(Boundary::no_slip);
      mg_constrained_dofs_.make_zero_boundary_constraints(
          discretization.dof_handler(), boundary_ids);

      typename MatrixFree<dim, float>::AdditionalData additional_data_level;
      additional_data_level.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;

      level_matrix_free_.resize(min_level, n_levels - 1);
      level_density_.resize(min_level, n_levels - 1);
      for (unsigned int level = min_level; level < n_levels; ++level) {
        additional_data_level.mg_level = level;
        AffineConstraints<double> constraints(relevant_sets[level],
                                              relevant_sets[level]);
        // constraints.add_lines(mg_constrained_dofs_.get_boundary_indices(level));
        // constraints.merge(mg_constrained_dofs_.get_level_constraints(level));
        constraints.close();
        level_matrix_free_[level].reinit(discretization.mapping(),
                                         discretization.dof_handler(),
                                         constraints,
                                         discretization.quadrature_1d(),
                                         additional_data_level);
        level_matrix_free_[level].initialize_dof_vector(level_density_[level]);
      }

      mg_transfer_velocity_.build(discretization.dof_handler(),
                                  mg_constrained_dofs_,
                                  level_matrix_free_);
      mg_transfer_energy_.build(discretization.dof_handler(),
                                level_matrix_free_);
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare_state_vector(
        StateVector<Number> & /*state_vector*/, Number /*t*/) const
    {
      RYUJIN_DEBUG_ANNOUNCE();

      /*
       * There is no parabolic part of the state vector for Navier-Stokes,
       * so we do nothing.
       */
    }


    template <int dim, typename Number>
    Number ParabolicModule<dim, Number>::backward_euler_step(
        const StateVector<Number> &old_state_vector,
        StageVectors stage_state_vectors,
        [[maybe_unused]] StageWeights stage_weights_explicit,
        StageWeights stage_weights_implicit,
        StateVector<Number> &new_state_vector,
        Number t,
        Number tau,
        Number /*tau_max*/) const
    {
      const auto stages = stage_state_vectors.size();
      AssertDimension(stage_weights_explicit.size(), stages);
      AssertDimension(stage_weights_implicit.size(), stages + 1);
      Assert(stage_weights_implicit.back() != Number(0.),
             dealii::ExcInternalError());

      step(old_state_vector,
           stage_state_vectors,
           stage_weights_implicit.first(stages),
           stage_weights_implicit.back(),
           new_state_vector,
           t,
           tau);

      return tau;
    }


    template <int dim, typename Number>
    Number ParabolicModule<dim, Number>::stage_combination_step(
        const StateVector<Number> &old_state_vector,
        StageVectors stage_state_vectors,
        [[maybe_unused]] StageWeights stage_weights_explicit,
        StageWeights stage_weights_implicit,
        StateVector<Number> &new_state_vector,
        Number t,
        Number tau) const
    {
      RYUJIN_DEBUG_ANNOUNCE();

      [[maybe_unused]] const auto stages = stage_state_vectors.size();
      AssertDimension(stage_weights_explicit.size(), stages);
      AssertDimension(stage_weights_implicit.size(), stages);

      step(old_state_vector,
           stage_state_vectors,
           stage_weights_implicit,
           Number(0.),
           new_state_vector,
           t,
           tau);

      ComputingTimer::Scope scope("time step [P] 4 - stage combination");

      constexpr ScalarNumber eps = std::numeric_limits<ScalarNumber>::epsilon();

      const auto old_U_view =
          old_state_vector.state.template view<problem_dimension>();
      const auto new_U_view =
          new_state_vector.state.template view<problem_dimension>();

      const unsigned int n_internal = offline_data_->n_locally_internal();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const auto sparsity_simd_view =
          offline_data_->sparsity_pattern_simd().view();

      /*
       * The new state is a high order update with weights that are in
       * general not convex. We require the lower bound eps * rho_i * e_min
       * on the internal energy, where e_min is the global minimum of the
       * specific internal energy of the old state:
       */

      const auto view = hyperbolic_system_->template view<dim, Number>();

      Number e_min = std::numeric_limits<Number>::max();
      for (unsigned int i = 0; i < n_owned; ++i) {
        if (sparsity_simd_view.row_length(i) == 1)
          continue;
        const auto U_i = old_U_view.template read_tensor<Number>(i);
        e_min = std::min(e_min, view.internal_energy(U_i) / view.density(U_i));
      }
      e_min = Utilities::MPI::min(e_min, mpi_ensemble_.ensemble_communicator());

      /*
       * Compute the largest limiter value l in [0, 1] such that W + l
       * (U^new - W) satisfies the bound, where W is the old state. The
       * internal energy rho e(l) = rho e^W + beta l - alpha l^2 is
       * concave in l and admissible for l = 0. We choose a single limiter
       * value for the whole domain in order to maintain conservation of
       * total energy.
       */

      Number l_min = Number(1.);
      for (unsigned int i = 0; i < n_owned; ++i) {
        if (sparsity_simd_view.row_length(i) == 1)
          continue;

        const auto U_s = old_U_view.template read_tensor<Number>(i);
        const auto delta = new_U_view.template read_tensor<Number>(i) - U_s;

        const auto rho = view.density(U_s);
        const auto m_s = view.momentum(U_s);
        const auto delta_m = view.momentum(delta);
        const auto delta_E = view.total_energy(delta);

        const auto f_0 = view.internal_energy(U_s) - eps * rho * e_min;
        const auto beta = delta_E - m_s * delta_m / rho;
        const auto alpha = Number(0.5) * delta_m.norm_square() / rho;

        if (f_0 + beta - alpha >= Number(0.))
          continue;

        Number l = Number(0.);
        if (f_0 > Number(0.)) {
          if (alpha > Number(0.))
            l = (beta + std::sqrt(beta * beta + 4. * alpha * f_0)) /
                (2. * alpha);
          else
            l = -f_0 / beta;
        }
        l_min = std::min(l_min, std::clamp(l, Number(0.), Number(1.)));
      }

      l_min = Utilities::MPI::min(l_min, mpi_ensemble_.ensemble_communicator());

      /*
       * Synchronize whether limiting is needed. Even though the bound only
       * affects the local ensemble we need to synchronize the boolean in
       * case we perform synchronized global time steps.
       */
      const bool limiting_needed = Utilities::MPI::logical_or(
          l_min < Number(1.), mpi_ensemble_.synchronization_communicator());

      if (!limiting_needed)
        return tau;

      /* If we can restart the time step, we do so: */
      if (this->id_violation_strategy_ ==
          IDViolationStrategy::raise_exception) {
        this->n_restarts_++;
        /* Half step size is a good heuristic: */
        throw Restart{Number(0.5) * tau};
      }

      this->n_corrections_++;

#ifdef DEBUG_OUTPUT
      std::cout << "        limiting stage combination with l = " << l_min
                << std::endl;
#endif

      const auto body_limit = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);

        /* Skip constrained degrees of freedom: */
        if (sparsity_simd_view.row_length(i) == 1)
          return;

        const auto U_s = old_U_view.template read_tensor<T>(i);
        const auto U_i = new_U_view.template read_tensor<T>(i);
        new_U_view.template write_tensor<T>(U_s + T(l_min) * (U_i - U_s), i);
      };

      cpu_simd_loop<Number>(
          "time_step_parabolic_4", body_limit, 0, n_internal, n_owned);

      new_U_view.update_ghost_values();

      return tau;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::print_solver_statistics(
        std::ostream &output) const
    {
      output << "        [ " << std::setprecision(2) << std::fixed
             << n_iterations_velocity_
             << (use_gmg_velocity_ ? " GMG vel -- " : " CG vel -- ")
             << n_iterations_internal_energy_
             << (use_gmg_internal_energy_ ? " GMG int ]" : " CG int ]")
             << std::endl;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::assemble_viscous_dissipation(
        ScalarHostVector<Number> &dst, const BlockHostVector &velocity) const
    {
      matrix_free_.template cell_loop<ScalarHostVector<Number>,
                                      BlockHostVector>(
          [this](const auto &data,
                 auto &dst,
                 const auto &src,
                 const auto cell_range) {
            FEEvaluation<dim, order_fe, order_quad, dim, Number> velocity(data);
            FEEvaluation<dim, order_fe, order_quad, 1, Number> energy(data);

            const auto mu = parabolic_system_->mu();
            const auto lambda = parabolic_system_->lambda();

            for (unsigned int cell = cell_range.first; cell < cell_range.second;
                 ++cell) {
              velocity.reinit(cell);
              energy.reinit(cell);
              velocity.gather_evaluate(src, EvaluationFlags::gradients);

              for (unsigned int q = 0; q < velocity.n_q_points; ++q) {
                if constexpr (dim == 1) {
                  /* Workaround: no symmetric gradient for dim == 1: */
                  const auto gradient = velocity.get_gradient(q);
                  auto S = (4. / 3. * mu + lambda) * gradient;
                  energy.submit_value(gradient * S, q);

                } else {

                  const auto symmetric_gradient =
                      velocity.get_symmetric_gradient(q);
                  const auto divergence = trace(symmetric_gradient);
                  auto S = 2. * mu * symmetric_gradient;
                  for (unsigned int d = 0; d < dim; ++d)
                    S[d][d] += (lambda - 2. / 3. * mu) * divergence;
                  energy.submit_value(symmetric_gradient * S, q);
                }
              }
              energy.integrate_scatter(EvaluationFlags::values, dst);
            }
          },
          dst,
          velocity,
          /* zero destination */ true);
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::step(
        const StateVector<Number> &old_state_vector,
        StageVectors stage_state_vectors,
        StageWeights stage_weights,
        const Number weight,
        StateVector<Number> &new_state_vector,
        Number t,
        Number tau) const
    {
      RYUJIN_DEBUG_ANNOUNCE();

      using HostSpace = dealii::MemorySpace::Host;

      copy_to_memory_space<HostSpace>(old_state_vector, StateVectorPart::state);
      move_to_memory_space<HostSpace>(new_state_vector, StateVectorPart::state);

      const auto old_U_view =
          old_state_vector.state.template view<problem_dimension>();
      const auto new_U_view =
          new_state_vector.state.template view<problem_dimension>();

      const auto lumped_mass_matrix_view =
          offline_data_->lumped_mass_matrix().template view<1>();
      const auto &affine_constraints = offline_data_->affine_constraints();

      /* Index ranges for the iteration over the sparsity pattern : */

      const unsigned int n_internal = offline_data_->n_locally_internal();
      const unsigned int n_owned = offline_data_->n_locally_owned();

      const auto sparsity_simd_view =
          offline_data_->sparsity_pattern_simd().view();

      DiagonalMatrix<dim, Number> diagonal_matrix;

#ifdef DEBUG_OUTPUT
      std::cout << "        perform time-step with tau = " << tau << std::endl;
#endif

      /*
       * The effective step size of the implicit operator. For a vanishing
       * weight of the new state the operators reduce to the (diagonal)
       * lumped mass matrix weighted with the density: we invert it
       * directly and skip the minimum principle check, see
       * stage_combination_step().
       */
      const Number tau_implicit = tau * weight;
      const bool implicit = weight != Number(0.);
      const bool use_gmg_velocity = use_gmg_velocity_ && implicit;
      const bool use_gmg_internal_energy = use_gmg_internal_energy_ && implicit;

      /* Whether the stage state vectors contribute to the right hand side: */
      const bool have_stages =
          std::any_of(stage_weights.begin(),
                      stage_weights.end(),
                      [](const Number w) { return w != Number(0.); });

      /*
       * Update MG matrices every n-th implicit step; this is a balance
       * because more refreshes will render the approximation better, at
       * some additional cost.
       */
      const bool reinitialize_gmg =
          implicit && (cycle_++ % gmg_reinitialization_interval_ == 0);

      /*
       * A boolean indicating that a restart is required.
       *
       * In our current implementation we set this boolean to true if the
       * backward Euler step produces an internal energy update that
       * violates the minimum principle, i.e., the minimum of the new
       * internal energy is smaller than the minimum of the old internal
       * energy.
       *
       * Depending on the chosen "id_violation_strategy" we either signal a
       * restart by throwing a "Restart" object, or we simply increase the
       * number of warnings.
       */
      std::atomic<bool> restart_needed = false;

      /*
       * Step 1:
       *
       * Build right hand side for the velocity update.
       * Also initialize solution vectors for internal energy and velocity
       * update.
       */
      {
        ComputingTimer::Scope scope("time step [P] 1 - update velocities");

        const auto body = [&](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);

          const auto view = hyperbolic_system_->template view<dim, T>();

          const auto U_i = old_U_view.template read_tensor<T>(i);
          const auto rho_i = view.density(U_i);
          const auto M_i = view.momentum(U_i);
          const auto rho_e_i = view.internal_energy(U_i);
          const auto m_i = lumped_mass_matrix_view.template read_entry<T>(i);

          write_entry<T>(density_, rho_i, i);
          /* (5.4a) */
          for (unsigned int d = 0; d < dim; ++d) {
            write_entry<T>(velocity_.block(d), M_i[d] / rho_i, i);
            write_entry<T>(velocity_rhs_.block(d), m_i * (M_i[d]), i);
          }
          write_entry<T>(internal_energy_, rho_e_i / rho_i, i);
        };

        cpu_simd_loop<Number>(
            "time_step_parabolic_1", body, 0, n_owned, n_owned);

        /*
         * Add the contributions of the stages to the right hand sides. For
         * a stage with velocity V^k, internal energy e^k, and implicit
         * weight w_k we add the nodal force
         *
         *   f_k = - tau w_k sum_j B_ij V^k_j
         *
         * to the velocity right hand side, and the viscous dissipation and
         * heat flux
         *
         *   tau w_k (m_i K_i(V^k) - sum_j beta_ij e^k_j)
         *
         * to the internal energy right hand side. We also record the sum
         * of the forces F = sum_k f_k and the work S = sum_k f_k V^k that
         * enter the energy conservation correction, see Step 2.
         */
        if (have_stages) {
          stage_force_ = Number(0.);
          stage_work_ = Number(0.);
          stage_energy_rhs_ = Number(0.);

          VelocityMatrix<dim, Number, Number> velocity_stiffness;
          velocity_stiffness.initialize(*parabolic_system_,
                                        *offline_data_,
                                        matrix_free_,
                                        density_,
                                        Number(1.));

          EnergyMatrix<dim, Number, Number> energy_stiffness;
          energy_stiffness.initialize(*offline_data_,
                                      matrix_free_,
                                      density_,
                                      parabolic_system_->cv_inverse_kappa());

          for (std::size_t k = 0; k < stage_state_vectors.size(); ++k) {
            const Number weight = stage_weights[k];
            if (weight == Number(0.))
              continue;

            const auto &stage_state_vector = stage_state_vectors[k].get();
            copy_to_memory_space<HostSpace>(stage_state_vector,
                                            StateVectorPart::state);
            const auto stage_U_view =
                stage_state_vector.state.template view<problem_dimension>();

            const auto body_extract = [&](auto sentinel, unsigned int i) {
              using T = decltype(sentinel);
              const auto view = hyperbolic_system_->template view<dim, T>();
              const auto U_i = stage_U_view.template read_tensor<T>(i);
              const auto rho_i = view.density(U_i);
              const auto M_i = view.momentum(U_i);
              for (unsigned int d = 0; d < dim; ++d)
                write_entry<T>(stage_velocity_.block(d), M_i[d] / rho_i, i);
              write_entry<T>(
                  stage_internal_energy_, view.internal_energy(U_i) / rho_i, i);
            };

            cpu_simd_loop<Number>(
                "time_step_parabolic_1", body_extract, 0, n_owned, n_owned);

            velocity_stiffness.apply_stiffness(stage_stiffness_,
                                               stage_velocity_);
            assemble_viscous_dissipation(internal_energy_rhs_, stage_velocity_);
            energy_stiffness.apply_stiffness(stage_diffusion_,
                                             stage_internal_energy_);

            const auto body_add = [&](auto sentinel, unsigned int i) {
              using T = decltype(sentinel);
              const T factor = tau * weight;

              T work = read_entry<T>(stage_work_, i);
              for (unsigned int d = 0; d < dim; ++d) {
                const auto f =
                    -factor * read_entry<T>(stage_stiffness_.block(d), i);
                const auto v = read_entry<T>(stage_velocity_.block(d), i);
                const auto rhs = read_entry<T>(velocity_rhs_.block(d), i);
                const auto force = read_entry<T>(stage_force_.block(d), i);
                write_entry<T>(velocity_rhs_.block(d), rhs + f, i);
                write_entry<T>(stage_force_.block(d), force + f, i);
                work += f * v;
              }
              write_entry<T>(stage_work_, work, i);

              const auto dissipation = read_entry<T>(internal_energy_rhs_, i);
              const auto diffusion = read_entry<T>(stage_diffusion_, i);
              const auto energy_rhs = read_entry<T>(stage_energy_rhs_, i);
              write_entry<T>(stage_energy_rhs_,
                             energy_rhs + factor * (dissipation - diffusion),
                             i);
            };

            cpu_simd_loop<Number>(
                "time_step_parabolic_1", body_add, 0, n_owned, n_owned);
          }
        }

        /*
         * Set up "strongly enforced" boundary conditions that are not stored
         * in the AffineConstraints map. In this case we enforce boundary
         * values by imposing them strongly in the iteration by setting the
         * initial vector and the right hand side to the right value:
         */

        const auto &boundary_map = offline_data_->boundary_map();

        for (auto entry : boundary_map) {
          // [i, normal, normal_mass, boundary_mass, id, position] = entry
          const auto i = std::get<0>(entry);
          if (i >= n_owned)
            continue;

          const auto normal = std::get<1>(entry);
          const auto id = std::get<4>(entry);
          const auto position = std::get<5>(entry);

          if (id == Boundary::slip) {
            /* Remove normal component of velocity: */
            Tensor<1, dim, Number> V_i;
            Tensor<1, dim, Number> RHS_i;
            for (unsigned int d = 0; d < dim; ++d) {
              V_i[d] = velocity_.block(d).local_element(i);
              RHS_i[d] = velocity_rhs_.block(d).local_element(i);
            }
            V_i -= 1. * (V_i * normal) * normal;
            RHS_i -= 1. * (RHS_i * normal) * normal;
            for (unsigned int d = 0; d < dim; ++d) {
              velocity_.block(d).local_element(i) = V_i[d];
              velocity_rhs_.block(d).local_element(i) = RHS_i[d];
            }

          } else if (id == Boundary::no_slip) {

            /* Set velocity to zero: */
            for (unsigned int d = 0; d < dim; ++d) {
              velocity_.block(d).local_element(i) = Number(0.);
              velocity_rhs_.block(d).local_element(i) = Number(0.);
            }

          } else if (id == Boundary::dirichlet) {

            /* Prescribe velocity: */
            const auto U_i = initial_values_->initial_state(position, t + tau);
            const auto view = hyperbolic_system_->template view<dim, Number>();
            const auto rho_i = view.density(U_i);
            const auto V_i = view.momentum(U_i) / rho_i;
            const auto e_i = view.internal_energy(U_i) / rho_i;

            for (unsigned int d = 0; d < dim; ++d) {
              velocity_.block(d).local_element(i) = V_i[d];
              velocity_rhs_.block(d).local_element(i) = V_i[d];
            }

            internal_energy_.local_element(i) = e_i;
          }
        }

        /*
         * Zero out constrained degrees of freedom due to hanging nodes and
         * periodic boundary conditions. These boundary conditions are
         * enforced by modifying the stencil - consequently we have to
         * remove constrained dofs from the linear system.
         */

        affine_constraints.set_zero(density_);
        affine_constraints.set_zero(internal_energy_);
        for (unsigned int d = 0; d < dim; ++d) {
          affine_constraints.set_zero(velocity_.block(d));
          affine_constraints.set_zero(velocity_rhs_.block(d));
        }

        /* Prepare preconditioner: */

        diagonal_matrix.reinit(
            lumped_mass_matrix_view, density_, affine_constraints);

        if ((use_gmg_velocity || use_gmg_internal_energy) && reinitialize_gmg)
          mg_transfer_velocity_.interpolate_to_mg(
              offline_data_->discretization().dof_handler(),
              level_density_,
              density_);

        if (use_gmg_velocity && reinitialize_gmg) {
          MGLevelObject<typename PreconditionChebyshev<
              VelocityMatrix<dim, float, Number>,
              LinearAlgebra::distributed::BlockVector<float>,
              DiagonalMatrix<dim, float>>::AdditionalData>
              smoother_data(level_matrix_free_.min_level(),
                            level_matrix_free_.max_level());

          level_velocity_matrices_.resize(level_matrix_free_.min_level(),
                                          level_matrix_free_.max_level());

          for (unsigned int level = level_matrix_free_.min_level();
               level <= level_matrix_free_.max_level();
               ++level) {
            level_velocity_matrices_[level].initialize(
                *parabolic_system_,
                *offline_data_,
                level_matrix_free_[level],
                level_density_[level],
                tau_implicit,
                level);
            level_velocity_matrices_[level].compute_diagonal(
                smoother_data[level].preconditioner);
            if (level == level_matrix_free_.min_level()) {
              smoother_data[level].degree = numbers::invalid_unsigned_int;
              smoother_data[level].eig_cg_n_iterations = 500;
              smoother_data[level].smoothing_range = 1e-3;
            } else {
              smoother_data[level].degree = gmg_smoother_degree_;
              smoother_data[level].eig_cg_n_iterations =
                  gmg_smoother_n_cg_iter_;
              smoother_data[level].smoothing_range = gmg_smoother_range_vel_;
              if (gmg_smoother_n_cg_iter_ == 0)
                smoother_data[level].max_eigenvalue = gmg_smoother_max_eig_vel_;
            }
          }
          mg_smoother_velocity_.initialize(level_velocity_matrices_,
                                           smoother_data);
        }
      }

      Number e_min_old;

      {
        ComputingTimer::Scope scope(
            "time step [X] _ - synchronization barriers");

        /*
         * Compute the global minimum of the internal energy over all
         * unconstrained degrees of freedom:
         */

        e_min_old = std::numeric_limits<Number>::max();
        for (unsigned int i = 0; i < n_owned; ++i)
          if (sparsity_simd_view.row_length(i) != 1)
            e_min_old = std::min(e_min_old, internal_energy_.local_element(i));

        e_min_old = Utilities::MPI::min(e_min_old,
                                        mpi_ensemble_.ensemble_communicator());

        // FIXME: create a meaningful relaxation based on global mesh size min.
        constexpr Number eps = std::numeric_limits<Number>::epsilon();
        e_min_old *= (1. - 1000. * eps);
      }

      /*
       * Step 1: Solve velocity update:
       */
      {
        ComputingTimer::Scope scope("time step [P] 1 - update velocities");

        VelocityMatrix<dim, Number, Number> velocity_operator;
        velocity_operator.initialize(*parabolic_system_,
                                     *offline_data_,
                                     matrix_free_,
                                     density_,
                                     tau_implicit);

        const auto tolerance_velocity =
            (tolerance_linfty_norm_ ? velocity_rhs_.linfty_norm()
                                    : velocity_rhs_.l2_norm()) *
            tolerance_;

        /*
         * Without implicit contribution the operator is diagonal. The
         * rows of degrees of freedom with prescribed velocity are the
         * identity.
         */
        if (!implicit) {
          diagonal_matrix.vmult(velocity_, velocity_rhs_);

          for (auto entry : offline_data_->boundary_map()) {
            const auto i = std::get<0>(entry);
            const auto id = std::get<4>(entry);
            if (i >= n_owned)
              continue;
            if (id == Boundary::no_slip || id == Boundary::dirichlet)
              for (unsigned int d = 0; d < dim; ++d)
                velocity_.block(d).local_element(i) =
                    velocity_rhs_.block(d).local_element(i);
          }

        } else {
          /*
           * Multigrid might lack robustness for some cases, so in case it takes
           * too many iterations we better switch to the more robust plain
           * conjugate gradient method.
           */
          try {
            if (!use_gmg_velocity)
              throw SolverControl::NoConvergence(0, 0.);

            using bvt_float = LinearAlgebra::distributed::BlockVector<float>;

            MGCoarseGridApplySmoother<bvt_float> mg_coarse;
            mg_coarse.initialize(mg_smoother_velocity_);

            mg::Matrix<bvt_float> mg_matrix(level_velocity_matrices_);

            Multigrid<bvt_float> mg(mg_matrix,
                                    mg_coarse,
                                    mg_transfer_velocity_,
                                    mg_smoother_velocity_,
                                    mg_smoother_velocity_,
                                    level_velocity_matrices_.min_level(),
                                    level_velocity_matrices_.max_level());

            const auto &dof_handler =
                offline_data_->discretization().dof_handler();
            PreconditionMG<dim, bvt_float, MGTransferVelocity<dim, float>>
                preconditioner(dof_handler, mg, mg_transfer_velocity_);

            SolverControl solver_control(gmg_max_iter_vel_, tolerance_velocity);
            SolverCG<BlockHostVector> solver(solver_control);
            solver.solve(
                velocity_operator, velocity_, velocity_rhs_, preconditioner);

            /* update exponential moving average */
            n_iterations_velocity_ =
                0.9 * n_iterations_velocity_ + 0.1 * solver_control.last_step();

          } catch (SolverControl::NoConvergence &) {

            SolverControl solver_control(1000, tolerance_velocity);
            SolverCG<BlockHostVector> solver(solver_control);
            solver.solve(
                velocity_operator, velocity_, velocity_rhs_, diagonal_matrix);

            /* update exponential moving average, counting also GMG iterations
             */
            n_iterations_velocity_ *= 0.9;
            n_iterations_velocity_ +=
                0.1 * (use_gmg_velocity ? gmg_max_iter_vel_ : 0) +
                0.1 * solver_control.last_step();
          }
        }
      }

      /*
       * Step 2: Build internal energy right hand side:
       */
      {
        ComputingTimer::Scope scope("time step [P] 2 - update internal energy");

        /* Compute m_i K_i^{n+1/2}:  (5.5) */
        if (implicit)
          assemble_viscous_dissipation(internal_energy_rhs_, velocity_);

        const auto lumped_mass_matrix_view =
            offline_data_->lumped_mass_matrix().template view<1>();

        const auto body = [&](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);

          const auto view = hyperbolic_system_->template view<dim, T>();

          const auto rhs_i = read_entry<T>(internal_energy_rhs_, i);
          const auto m_i = lumped_mass_matrix_view.template read_entry<T>(i);
          const auto rho_i = read_entry<T>(density_, i);
          const auto e_i = read_entry<T>(internal_energy_, i);

          const auto U_i = old_U_view.template read_tensor<T>(i);
          const auto V_i = view.momentum(U_i) / rho_i;

          dealii::Tensor<1, dim, T> V_i_new;
          for (unsigned int d = 0; d < dim; ++d) {
            V_i_new[d] = read_entry<T>(velocity_.block(d), i);
          }

          /*
           * We have to add an algebraic correction to ensure conservation
           * of total energy. With the nodal forces f_k of the stages, their
           * sum F, and their work S = sum_k f_k V^k (see Step 1) the
           * correction reads
           *
           *   m_i rho_i 1/2 |V_i - V_i^new|^2 + S_i - F_i V_i^new,
           *
           * which reduces to the classical correction for a plain backward
           * Euler step.
           */
          auto result =
              m_i * rho_i * (e_i + Number(0.5) * (V_i - V_i_new).norm_square());

          /* rhs_i contains already m_i K_i^{n+1/2} */
          if (implicit)
            result += tau_implicit * rhs_i;

          if (have_stages) {
            result += read_entry<T>(stage_energy_rhs_, i);
            result += read_entry<T>(stage_work_, i);
            for (unsigned int d = 0; d < dim; ++d)
              result -= read_entry<T>(stage_force_.block(d), i) * V_i_new[d];
          }

          write_entry<T>(internal_energy_rhs_, result, i);
        };

        cpu_simd_loop<Number>(
            "time_step_parabolic_2", body, 0, n_owned, n_owned);

        /*
         * Set up "strongly enforced" boundary conditions that are not stored
         * in the AffineConstraints map: We enforce Neumann conditions (i.e.,
         * insulating boundary conditions) everywhere except for Dirichlet
         * boundaries where we have to enforce prescribed conditions:
         */

        const auto &boundary_map = offline_data_->boundary_map();

        for (auto entry : boundary_map) {
          // [i, normal, normal_mass, boundary_mass, id, position] = entry
          const auto i = std::get<0>(entry);
          if (i >= n_owned)
            continue;

          const auto id = std::get<4>(entry);
          const auto position = std::get<5>(entry);

          if (id == Boundary::dirichlet) {
            /* Prescribe internal energy: */
            const auto U_i = initial_values_->initial_state(position, t + tau);
            const auto view = hyperbolic_system_->template view<dim, Number>();
            const auto rho_i = view.density(U_i);
            const auto e_i = view.internal_energy(U_i) / rho_i;
            internal_energy_rhs_.local_element(i) = e_i;
          }
        }

        /*
         * Zero out constrained degrees of freedom due to hanging nodes and
         * periodic boundary conditions. These boundary conditions are
         * enforced by modifying the stencil - consequently we have to
         * remove constrained dofs from the linear system.
         */
        affine_constraints.set_zero(internal_energy_);
        affine_constraints.set_zero(internal_energy_rhs_);

        if (use_gmg_internal_energy && reinitialize_gmg) {
          MGLevelObject<typename PreconditionChebyshev<
              EnergyMatrix<dim, float, Number>,
              LinearAlgebra::distributed::Vector<float>>::AdditionalData>
              smoother_data(level_matrix_free_.min_level(),
                            level_matrix_free_.max_level());

          level_energy_matrices_.resize(level_matrix_free_.min_level(),
                                        level_matrix_free_.max_level());

          for (unsigned int level = level_matrix_free_.min_level();
               level <= level_matrix_free_.max_level();
               ++level) {
            level_energy_matrices_[level].initialize(
                *offline_data_,
                level_matrix_free_[level],
                level_density_[level],
                tau_implicit * parabolic_system_->cv_inverse_kappa(),
                level);
            level_energy_matrices_[level].compute_diagonal(
                smoother_data[level].preconditioner);
            if (level == level_matrix_free_.min_level()) {
              smoother_data[level].degree = numbers::invalid_unsigned_int;
              smoother_data[level].eig_cg_n_iterations = 500;
              smoother_data[level].smoothing_range = 1e-3;
            } else {
              smoother_data[level].degree = gmg_smoother_degree_;
              smoother_data[level].eig_cg_n_iterations =
                  gmg_smoother_n_cg_iter_;
              smoother_data[level].smoothing_range = gmg_smoother_range_en_;
              if (gmg_smoother_n_cg_iter_ == 0)
                smoother_data[level].max_eigenvalue = gmg_smoother_max_eig_en_;
            }
          }
          mg_smoother_energy_.initialize(level_energy_matrices_, smoother_data);
        }
      }

      /*
       * Step 2: Solve internal energy update:
       */
      {
        ComputingTimer::Scope scope("time step [P] 2 - update internal energy");

        EnergyMatrix<dim, Number, Number> energy_operator;
        const auto &kappa = parabolic_system_->cv_inverse_kappa();
        energy_operator.initialize(
            *offline_data_, matrix_free_, density_, tau_implicit * kappa);

        const auto tolerance_internal_energy =
            (tolerance_linfty_norm_ ? internal_energy_rhs_.linfty_norm()
                                    : internal_energy_rhs_.l2_norm()) *
            tolerance_;

        /*
         * Without implicit contribution the operator is diagonal. The
         * rows of degrees of freedom with prescribed internal energy are
         * the identity.
         */
        if (!implicit) {
          diagonal_matrix.vmult(internal_energy_, internal_energy_rhs_);

          for (auto entry : offline_data_->boundary_map()) {
            const auto i = std::get<0>(entry);
            const auto id = std::get<4>(entry);
            if (i < n_owned && id == Boundary::dirichlet)
              internal_energy_.local_element(i) =
                  internal_energy_rhs_.local_element(i);
          }

        } else {
          try {
            if (!use_gmg_internal_energy)
              throw SolverControl::NoConvergence(0, 0.);

            using vt_float = LinearAlgebra::distributed::Vector<float>;
            MGCoarseGridApplySmoother<vt_float> mg_coarse;
            mg_coarse.initialize(mg_smoother_energy_);
            mg::Matrix<vt_float> mg_matrix(level_energy_matrices_);

            Multigrid<vt_float> mg(mg_matrix,
                                   mg_coarse,
                                   mg_transfer_energy_,
                                   mg_smoother_energy_,
                                   mg_smoother_energy_,
                                   level_energy_matrices_.min_level(),
                                   level_energy_matrices_.max_level());

            const auto &dof_handler =
                offline_data_->discretization().dof_handler();
            PreconditionMG<dim, vt_float, MGTransferEnergy<dim, float>>
                preconditioner(dof_handler, mg, mg_transfer_energy_);

            SolverControl solver_control(gmg_max_iter_en_,
                                         tolerance_internal_energy);
            SolverCG<ScalarHostVector<Number>> solver(solver_control);
            solver.solve(energy_operator,
                         internal_energy_,
                         internal_energy_rhs_,
                         preconditioner);

            /* update exponential moving average */
            n_iterations_internal_energy_ =
                0.9 * n_iterations_internal_energy_ +
                0.1 * solver_control.last_step();

          } catch (SolverControl::NoConvergence &) {

            SolverControl solver_control(1000, tolerance_internal_energy);
            SolverCG<ScalarHostVector<Number>> solver(solver_control);
            solver.solve(energy_operator,
                         internal_energy_,
                         internal_energy_rhs_,
                         diagonal_matrix);

            /* update exponential moving average, counting also GMG iterations
             */
            n_iterations_internal_energy_ *= 0.9;
            n_iterations_internal_energy_ +=
                0.1 * (use_gmg_internal_energy ? gmg_max_iter_en_ : 0) +
                0.1 * solver_control.last_step();
          }
        }
      }

      /*
       * Step 3: Copy vectors and check for local minimum principle on
       * internal energy:
       *
       * FIXME: Memory access is suboptimal...
       */
      {
        ComputingTimer::Scope scope("time step [P] 3 - write back vectors");

        const auto body = [&](auto sentinel, unsigned int i) {
          using T = decltype(sentinel);

          const auto view = hyperbolic_system_->template view<dim, T>();

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd_view.row_length(i);
          if (row_length == 1)
            return;

          auto U_i = old_U_view.template read_tensor<T>(i);
          const auto rho_i = view.density(U_i);

          Tensor<1, dim, T> m_i_new;
          for (unsigned int d = 0; d < dim; ++d) {
            m_i_new[d] = rho_i * read_entry<T>(velocity_.block(d), i);
          }

          const auto rho_e_i_new = rho_i * read_entry<T>(internal_energy_, i);

          /*
           * Check that the backward Euler step itself (which is our "low
           * order" update) satisfies bounds. If not, signal a restart.
           */

          if (implicit &&
              !(T(0.) == std::max(T(0.), rho_i * e_min_old - rho_e_i_new))) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            const auto e_i_new = rho_e_i_new / rho_i;
            std::cout << "Bounds violation: internal energy (critical)!\n"
                      << "\t\te_min_old:         " << e_min_old << "\n"
                      << "\t\te_min_old (delta): "
                      << negative_part(e_i_new - e_min_old) << "\n"
                      << "\t\te_min_new:         " << e_i_new << "\n"
                      << std::endl;
#endif
            restart_needed = true;
          }

          const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

          for (unsigned int d = 0; d < dim; ++d)
            U_i[1 + d] = m_i_new[d];
          U_i[1 + dim] = E_i_new;

          new_U_view.template write_tensor<T>(U_i, i);
        };

        cpu_simd_loop<Number>(
            "time_step_parabolic_3", body, 0, n_internal, n_owned);

        new_U_view.update_ghost_values();
      }

      {
        ComputingTimer::Scope scope(
            "time step [X] _ - synchronization barriers");

        /*
         * Synchronize whether we have to restart the time step. Even
         * though the restart condition itself only affects the local
         * ensemble we nevertheless need to synchronize the boolean in case
         * we perform synchronized global time steps. (Otherwise different
         * ensembles might end up with a different time step.)
         */

        restart_needed.store(Utilities::MPI::logical_or(
            restart_needed.load(),
            mpi_ensemble_.synchronization_communicator()));
      }

      if (restart_needed) {
        switch (this->id_violation_strategy_) {
        case IDViolationStrategy::warn:
          this->n_warnings_++;
          break;
        case IDViolationStrategy::raise_exception:
          this->n_restarts_++;
          /* Half step size is a good heuristic: */
          throw Restart{Number(0.5) * tau};
        }
      }
    }

  } // namespace NavierStokes
} // namespace ryujin
