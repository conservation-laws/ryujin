//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 by the ryujin authors
//

#pragma once

#include "parabolic_solver.h"

#include <introspection.h>
#include <openmp.h>
#include <scope.h>
#include <simd.h>

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

    template <typename Description, int dim, typename Number>
    ParabolicSolver<Description, dim, Number>::ParabolicSolver(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const HyperbolicSystem &hyperbolic_system,
        const ParabolicSystem &parabolic_system,
        const OfflineData<dim, Number> &offline_data,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection /*= "ParabolicSolver"*/)
        : ParameterAcceptor(subsection)
        , mpi_communicator_(mpi_communicator)
        , computing_timer_(computing_timer)
        , hyperbolic_system_(&hyperbolic_system)
        , parabolic_system_(&parabolic_system)
        , offline_data_(&offline_data)
        , initial_values_(&initial_values)
        , n_restarts_(0)
        , n_warnings_(0)
        , n_iterations_velocity_(0.)
        , n_iterations_internal_energy_(0.)
    {
      use_gmg_velocity_ = false;
      add_parameter("multigrid velocity",
                    use_gmg_velocity_,
                    "Use geometric multigrid for velocity component");

      gmg_max_iter_vel_ = 12;
      add_parameter("multigrid velocity - max iter",
                    gmg_max_iter_vel_,
                    "Maximal number of CG iterations with GMG smoother");

      gmg_smoother_range_vel_ = 8.;
      add_parameter("multigrid velocity - chebyshev range",
                    gmg_smoother_range_vel_,
                    "Chebyshev smoother: eigenvalue range parameter");

      gmg_smoother_max_eig_vel_ = 2.0;
      add_parameter("multigrid velocity - chebyshev max eig",
                    gmg_smoother_max_eig_vel_,
                    "Chebyshev smoother: maximal eigenvalue");

      use_gmg_internal_energy_ = false;
      add_parameter("multigrid energy",
                    use_gmg_internal_energy_,
                    "Use geometric multigrid for internal energy component");

      gmg_max_iter_en_ = 15;
      add_parameter("multigrid energy - max iter",
                    gmg_max_iter_en_,
                    "Maximal number of CG iterations with GMG smoother");

      gmg_smoother_range_en_ = 15.;
      add_parameter("multigrid energy - chebyshev range",
                    gmg_smoother_range_en_,
                    "Chebyshev smoother: eigenvalue range parameter");

      gmg_smoother_max_eig_en_ = 2.0;
      add_parameter("multigrid energy - chebyshev max eig",
                    gmg_smoother_max_eig_en_,
                    "Chebyshev smoother: maximal eigenvalue");

      gmg_smoother_degree_ = 3;
      add_parameter("multigrid - chebyshev degree",
                    gmg_smoother_degree_,
                    "Chebyshev smoother: degree");

      gmg_smoother_n_cg_iter_ = 10;
      add_parameter(
          "multigrid - chebyshev cg iter",
          gmg_smoother_n_cg_iter_,
          "Chebyshev smoother: number of CG iterations to approximate "
          "eigenvalue");

      gmg_min_level_ = 0;
      add_parameter(
          "multigrid - min level",
          gmg_min_level_,
          "Minimal mesh level to be visited in the geometric multigrid "
          "cycle where the coarse grid solver (Chebyshev) is called");

      tolerance_ = Number(1.0e-12);
      add_parameter("tolerance", tolerance_, "Tolerance for linear solvers");

      tolerance_linfty_norm_ = false;
      add_parameter("tolerance linfty norm",
                    tolerance_linfty_norm_,
                    "Use the l_infty norm instead of the l_2 norm for the "
                    "stopping criterion");
    }


    template <typename Description, int dim, typename Number>
    void ParabolicSolver<Description, dim, Number>::prepare()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicSolver<dim, Number>::prepare()" << std::endl;
#endif

      const auto &discretization = offline_data_->discretization();
      AssertThrow(discretization.ansatz() == Ansatz::cg_q1,
                  dealii::ExcMessage("The NavierStokes module currently only "
                                     "supports cG Q1 finite elements."));

      /* Initialize vectors: */

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim, Number>::AdditionalData::none;

      matrix_free_.reinit(discretization.mapping(),
                          offline_data_->dof_handler(),
                          offline_data_->affine_constraints(),
                          discretization.quadrature_1d(),
                          additional_data);

      const auto &scalar_partitioner =
          matrix_free_.get_dof_info(0).vector_partitioner;

      velocity_.reinit(dim);
      velocity_rhs_.reinit(dim);
      for (unsigned int i = 0; i < dim; ++i) {
        velocity_.block(i).reinit(scalar_partitioner);
        velocity_rhs_.block(i).reinit(scalar_partitioner);
      }

      internal_energy_.reinit(scalar_partitioner);
      internal_energy_rhs_.reinit(scalar_partitioner);

      density_.reinit(scalar_partitioner);

      /* Initialize multigrid: */

      if (!use_gmg_velocity_ && !use_gmg_internal_energy_)
        return;

      const unsigned int n_levels =
          offline_data_->dof_handler().get_triangulation().n_global_levels();
      const unsigned int min_level = std::min(gmg_min_level_, n_levels - 1);
      MGLevelObject<IndexSet> relevant_sets(0, n_levels - 1);
      for (unsigned int level = 0; level < n_levels; ++level)
        dealii::DoFTools::extract_locally_relevant_level_dofs(
            offline_data_->dof_handler(), level, relevant_sets[level]);
      mg_constrained_dofs_.initialize(offline_data_->dof_handler(),
                                      relevant_sets);
      std::set<types::boundary_id> boundary_ids;
      boundary_ids.insert(Boundary::dirichlet);
      boundary_ids.insert(Boundary::no_slip);
      mg_constrained_dofs_.make_zero_boundary_constraints(
          offline_data_->dof_handler(), boundary_ids);

      typename MatrixFree<dim, float>::AdditionalData additional_data_level;
      additional_data_level.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;

      level_matrix_free_.resize(min_level, n_levels - 1);
      level_density_.resize(min_level, n_levels - 1);
      for (unsigned int level = min_level; level < n_levels; ++level) {
        additional_data_level.mg_level = level;
        AffineConstraints<double> constraints(relevant_sets[level]);
        // constraints.add_lines(mg_constrained_dofs_.get_boundary_indices(level));
        // constraints.merge(mg_constrained_dofs_.get_level_constraints(level));
        constraints.close();
        level_matrix_free_[level].reinit(discretization.mapping(),
                                         offline_data_->dof_handler(),
                                         constraints,
                                         discretization.quadrature_1d(),
                                         additional_data_level);
        level_matrix_free_[level].initialize_dof_vector(level_density_[level]);
      }

      mg_transfer_velocity_.build(offline_data_->dof_handler(),
                                  mg_constrained_dofs_,
                                  level_matrix_free_);
      mg_transfer_energy_.build(offline_data_->dof_handler(),
                                level_matrix_free_);
    }


    template <typename Description, int dim, typename Number>
    void ParabolicSolver<Description, dim, Number>::backward_euler_step(
        const StateVector &old_state_vector,
        const Number t,
        StateVector &new_state_vector,
        Number tau,
        const IDViolationStrategy id_violation_strategy,
        const bool reinitialize_gmg) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicSolver<dim, Number>::step()" << std::endl;
#endif

      const auto &old_U = std::get<0>(old_state_vector);
      auto &new_U = std::get<0>(new_state_vector);

      CALLGRIND_START_INSTRUMENTATION;

      using VA = VectorizedArray<Number>;

      const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
      const auto &affine_constraints = offline_data_->affine_constraints();

      /* Index ranges for the iteration over the sparsity pattern : */

      constexpr auto simd_length = VA::size();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const unsigned int n_regular = n_owned / simd_length * simd_length;

      DiagonalMatrix<dim, Number> diagonal_matrix;

#ifdef DEBUG_OUTPUT
      std::cout << "        perform time-step with tau = " << tau << std::endl;
#endif

      /* A boolean signalling that a restart is necessary: */
      std::atomic<bool> restart_needed = false;

      /*
       * Step 1:
       *
       * Build right hand side for the velocity update.
       * Also initialize solution vectors for internal energy and velocity
       * update.
       */
      {
        Scope scope(computing_timer_, "time step [P] 1 - update velocities");
        RYUJIN_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START("time_step_parabolic_1");

        auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
          using T = decltype(sentinel);
          unsigned int stride_size = get_stride_size<T>;

          const auto view = hyperbolic_system_->template view<dim, T>();

          RYUJIN_OMP_FOR
          for (unsigned int i = left; i < right; i += stride_size) {
            const auto U_i = old_U.template get_tensor<T>(i);
            const auto rho_i = view.density(U_i);
            const auto M_i = view.momentum(U_i);
            const auto rho_e_i = view.internal_energy(U_i);
            const auto m_i = get_entry<T>(lumped_mass_matrix, i);

            write_entry<T>(density_, rho_i, i);
            /* (5.4a) */
            for (unsigned int d = 0; d < dim; ++d) {
              write_entry<T>(velocity_.block(d), M_i[d] / rho_i, i);
              write_entry<T>(velocity_rhs_.block(d), m_i * (M_i[d]), i);
            }
            write_entry<T>(internal_energy_, rho_e_i / rho_i, i);
          }
        };

        /* Parallel non-vectorized loop: */
        loop(Number(), n_regular, n_owned);
        /* Parallel vectorized SIMD loop: */
        loop(VA(), 0, n_regular);

        RYUJIN_PARALLEL_REGION_END

        /*
         * Set up "strongly enforced" boundary conditions that are not stored
         * in the AffineConstraints map. In this case we enforce boundary
         * values by imposing them strongly in the iteration by setting the
         * initial vector and the right hand side to the right value:
         */

        const auto &boundary_map = offline_data_->boundary_map();

        for (auto entry : boundary_map) {
          const auto i = entry.first;
          if (i >= n_owned)
            continue;

          const auto normal = std::get<0>(entry.second);
          const auto id = std::get<3>(entry.second);
          const auto position = std::get<4>(entry.second);

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
         * Zero out constrained degrees of freedom due to periodic boundary
         * conditions. These boundary conditions are enforced by modifying
         * the stencil - consequently we have to remove constrained dofs from
         * the linear system.
         */

        affine_constraints.set_zero(density_);
        affine_constraints.set_zero(internal_energy_);
        for (unsigned int d = 0; d < dim; ++d) {
          affine_constraints.set_zero(velocity_.block(d));
          affine_constraints.set_zero(velocity_rhs_.block(d));
        }

        /* Prepare preconditioner: */

        diagonal_matrix.reinit(
            lumped_mass_matrix, density_, affine_constraints);

        /*
         * Update MG matrices all 4 time steps; this is a balance because more
         * refreshes will render the approximation better, at some additional
         * cost.
         */
        if (use_gmg_velocity_ && reinitialize_gmg) {
          MGLevelObject<typename PreconditionChebyshev<
              VelocityMatrix<dim, float, Number>,
              LinearAlgebra::distributed::BlockVector<float>,
              DiagonalMatrix<dim, float>>::AdditionalData>
              smoother_data(level_matrix_free_.min_level(),
                            level_matrix_free_.max_level());

          level_velocity_matrices_.resize(level_matrix_free_.min_level(),
                                          level_matrix_free_.max_level());
          mg_transfer_velocity_.interpolate_to_mg(
              offline_data_->dof_handler(), level_density_, density_);

          for (unsigned int level = level_matrix_free_.min_level();
               level <= level_matrix_free_.max_level();
               ++level) {
            level_velocity_matrices_[level].initialize(
                *parabolic_system_,
                *offline_data_,
                level_matrix_free_[level],
                level_density_[level],
                tau,
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

        LIKWID_MARKER_STOP("time_step_parabolic_1");
      }

      /* Compute the global minimum of the internal energy: */

      // .begin() and .end() denote the locally owned index range:
      auto e_min_old =
          *std::min_element(internal_energy_.begin(), internal_energy_.end());
      e_min_old = Utilities::MPI::min(e_min_old, mpi_communicator_);

      // FIXME: create a meaningful relaxation based on global mesh size min.
      constexpr Number eps = std::numeric_limits<Number>::epsilon();
      e_min_old *= (1. - 1000. * eps);

      /*
       * Step 1: Solve velocity update:
       */
      {
        Scope scope(computing_timer_, "time step [P] 1 - update velocities");

        LIKWID_MARKER_START("time_step_parabolic_1");

        VelocityMatrix<dim, Number, Number> velocity_operator;
        velocity_operator.initialize(
            *parabolic_system_, *offline_data_, matrix_free_, density_, tau);

        const auto tolerance_velocity =
            (tolerance_linfty_norm_ ? velocity_rhs_.linfty_norm()
                                    : velocity_rhs_.l2_norm()) *
            tolerance_;

        /*
         * Multigrid might lack robustness for some cases, so in case it takes
         * too many iterations we better switch to the more robust plain
         * conjugate gradient method.
         */
        try {
          if (!use_gmg_velocity_)
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

          const auto &dof_handler = offline_data_->dof_handler();
          PreconditionMG<dim, bvt_float, MGTransferVelocity<dim, float>>
              preconditioner(dof_handler, mg, mg_transfer_velocity_);

          SolverControl solver_control(gmg_max_iter_vel_, tolerance_velocity);
          SolverCG<BlockVector> solver(solver_control);
          solver.solve(
              velocity_operator, velocity_, velocity_rhs_, preconditioner);

          /* update exponential moving average */
          n_iterations_velocity_ =
              0.9 * n_iterations_velocity_ + 0.1 * solver_control.last_step();

        } catch (SolverControl::NoConvergence &) {

          SolverControl solver_control(1000, tolerance_velocity);
          SolverCG<BlockVector> solver(solver_control);
          solver.solve(
              velocity_operator, velocity_, velocity_rhs_, diagonal_matrix);

          /* update exponential moving average, counting also GMG iterations */
          n_iterations_velocity_ *= 0.9;
          n_iterations_velocity_ +=
              0.1 * (use_gmg_velocity_ ? gmg_max_iter_vel_ : 0) +
              0.1 * solver_control.last_step();
        }

        LIKWID_MARKER_STOP("time_step_parabolic_1");
      }

      /*
       * Step 2: Build internal energy right hand side:
       */
      {
        Scope scope(computing_timer_,
                    "time step [P] 2 - update internal energy");

        LIKWID_MARKER_START("time_step_parabolic_2");

        /* Compute m_i K_i^{n+1/2}:  (5.5) */
        matrix_free_.template cell_loop<ScalarVector, BlockVector>(
            [this](const auto &data,
                   auto &dst,
                   const auto &src,
                   const auto cell_range) {
              FEEvaluation<dim, order_fe, order_quad, dim, Number> velocity(
                  data);
              FEEvaluation<dim, order_fe, order_quad, 1, Number> energy(data);

              const auto mu = parabolic_system_->mu();
              const auto lambda = parabolic_system_->lambda();

              for (unsigned int cell = cell_range.first;
                   cell < cell_range.second;
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
            internal_energy_rhs_,
            velocity_,
            /* zero destination */ true);

        const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();

        RYUJIN_PARALLEL_REGION_BEGIN

        auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
          using T = decltype(sentinel);
          unsigned int stride_size = get_stride_size<T>;

          const auto view = hyperbolic_system_->template view<dim, T>();

          RYUJIN_OMP_FOR
          for (unsigned int i = left; i < right; i += stride_size) {
            const auto rhs_i = get_entry<T>(internal_energy_rhs_, i);
            const auto m_i = get_entry<T>(lumped_mass_matrix, i);
            const auto rho_i = get_entry<T>(density_, i);
            const auto e_i = get_entry<T>(internal_energy_, i);

            const auto U_i = old_U.template get_tensor<T>(i);
            const auto V_i = view.momentum(U_i) / rho_i;

            dealii::Tensor<1, dim, T> V_i_new;
            for (unsigned int d = 0; d < dim; ++d) {
              V_i_new[d] = get_entry<T>(velocity_.block(d), i);
            }

            /*
             * For backward Euler we have to add this algebraic correction
             * to ensure conservation of total energy.
             */
            const auto correction = Number(0.5) * (V_i - V_i_new).norm_square();

            /* rhs_i contains already m_i K_i^{n+1/2} */
            const auto result = m_i * rho_i * (e_i + correction) + tau * rhs_i;
            write_entry<T>(internal_energy_rhs_, result, i);
          }
        };

        /* Parallel non-vectorized loop: */
        loop(Number(), n_regular, n_owned);
        /* Parallel vectorized SIMD loop: */
        loop(VA(), 0, n_regular);

        RYUJIN_PARALLEL_REGION_END

        /*
         * Set up "strongly enforced" boundary conditions that are not stored
         * in the AffineConstraints map: We enforce Neumann conditions (i.e.,
         * insulating boundary conditions) everywhere except for Dirichlet
         * boundaries where we have to enforce prescribed conditions:
         */

        const auto &boundary_map = offline_data_->boundary_map();

        for (auto entry : boundary_map) {
          const auto i = entry.first;
          if (i >= n_owned)
            continue;

          const auto id = std::get<3>(entry.second);
          const auto position = std::get<4>(entry.second);

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
         * Zero out constrained degrees of freedom due to periodic boundary
         * conditions. These boundary conditions are enforced by modifying
         * the stencil - consequently we have to remove constrained dofs from
         * the linear system.
         */
        affine_constraints.set_zero(internal_energy_rhs_);

        /*
         * Update MG matrices all 4 time steps; this is a balance because more
         * refreshes will render the approximation better, at some additional
         * cost.
         */
        if (use_gmg_internal_energy_ && reinitialize_gmg) {
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
                tau * parabolic_system_->cv_inverse_kappa(),
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

        LIKWID_MARKER_STOP("time_step_parabolic_2");
      }

      /*
       * Step 2: Solve internal energy update:
       */
      {
        Scope scope(computing_timer_,
                    "time step [P] 2 - update internal energy");

        LIKWID_MARKER_START("time_step_parabolic_2");

        EnergyMatrix<dim, Number, Number> energy_operator;
        const auto &kappa = parabolic_system_->cv_inverse_kappa();
        energy_operator.initialize(
            *offline_data_, matrix_free_, density_, tau * kappa);

        const auto tolerance_internal_energy =
            (tolerance_linfty_norm_ ? internal_energy_rhs_.linfty_norm()
                                    : internal_energy_rhs_.l2_norm()) *
            tolerance_;

        try {
          if (!use_gmg_internal_energy_)
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

          const auto &dof_handler = offline_data_->dof_handler();
          PreconditionMG<dim, vt_float, MGTransferEnergy<dim, float>>
              preconditioner(dof_handler, mg, mg_transfer_energy_);

          SolverControl solver_control(gmg_max_iter_en_,
                                       tolerance_internal_energy);
          SolverCG<ScalarVector> solver(solver_control);
          solver.solve(energy_operator,
                       internal_energy_,
                       internal_energy_rhs_,
                       preconditioner);

          /* update exponential moving average */
          n_iterations_internal_energy_ = 0.9 * n_iterations_internal_energy_ +
                                          0.1 * solver_control.last_step();

        } catch (SolverControl::NoConvergence &) {

          SolverControl solver_control(1000, tolerance_internal_energy);
          SolverCG<ScalarVector> solver(solver_control);
          solver.solve(energy_operator,
                       internal_energy_,
                       internal_energy_rhs_,
                       diagonal_matrix);

          /* update exponential moving average, counting also GMG iterations */
          n_iterations_internal_energy_ *= 0.9;
          n_iterations_internal_energy_ +=
              0.1 * (use_gmg_internal_energy_ ? gmg_max_iter_en_ : 0) +
              0.1 * solver_control.last_step();
        }

        /*
         * Check for local minimum principle on internal energy:
         */
        {
          // .begin() and .end() denote the locally owned index range:
          auto e_min_new = *std::min_element(internal_energy_.begin(),
                                             internal_energy_.end());
          e_min_new = Utilities::MPI::min(e_min_new, mpi_communicator_);

          if (e_min_new < e_min_old) {
#ifdef DEBUG_OUTPUT
            std::cout << std::fixed << std::setprecision(16);
            std::cout << "Bounds violation: internal energy (critical)!\n"
                      << "\t\te_min_old:         " << e_min_old << "\n"
                      << "\t\te_min_old (delta): "
                      << negative_part(e_min_new - e_min_old) << "\n"
                      << "\t\te_min_new:         " << e_min_new << "\n"
                      << std::endl;
#endif
            restart_needed = true;
          }
        }

        LIKWID_MARKER_STOP("time_step_parabolic_2");
      }

      /*
       * Step 3: Copy vectors
       *
       * FIXME: Memory access is suboptimal...
       */
      {
        Scope scope(computing_timer_, "time step [P] 3 - write back vectors");

        RYUJIN_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START("time_step_parabolic_3");

        auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
          using T = decltype(sentinel);
          unsigned int stride_size = get_stride_size<T>;

          const auto view = hyperbolic_system_->template view<dim, T>();

          RYUJIN_OMP_FOR
          for (unsigned int i = left; i < right; i += stride_size) {
            auto U_i = old_U.template get_tensor<T>(i);
            const auto rho_i = view.density(U_i);

            Tensor<1, dim, T> m_i_new;
            for (unsigned int d = 0; d < dim; ++d) {
              m_i_new[d] = rho_i * get_entry<T>(velocity_.block(d), i);
            }

            const auto rho_e_i_new = rho_i * get_entry<T>(internal_energy_, i);

            const auto E_i_new = rho_e_i_new + 0.5 * m_i_new * m_i_new / rho_i;

            for (unsigned int d = 0; d < dim; ++d)
              U_i[1 + d] = m_i_new[d];
            U_i[1 + dim] = E_i_new;

            new_U.template write_tensor<T>(U_i, i);
          }
        };

        /* Parallel non-vectorized loop: */
        loop(Number(), n_regular, n_owned);
        /* Parallel vectorized SIMD loop: */
        loop(VA(), 0, n_regular);

        LIKWID_MARKER_STOP("time_step_parabolic_3");
        RYUJIN_PARALLEL_REGION_END

        new_U.update_ghost_values();
      }

      CALLGRIND_STOP_INSTRUMENTATION;

      if (restart_needed) {
        switch (id_violation_strategy) {
        case IDViolationStrategy::warn:
          n_warnings_++;
          break;
        case IDViolationStrategy::raise_exception:
          n_restarts_++;
          throw Restart();
        }
      }
    }


    template <typename Description, int dim, typename Number>
    void ParabolicSolver<Description, dim, Number>::print_solver_statistics(
        std::ostream &output) const
    {
      output << "        [ " << std::setprecision(2) << std::fixed
             << n_iterations_velocity_
             << (use_gmg_velocity_ ? " GMG vel -- " : " CG vel -- ")
             << n_iterations_internal_energy_
             << (use_gmg_internal_energy_ ? " GMG int ]" : " CG int ]")
             << std::endl;
    }

  } // namespace NavierStokes
} /* namespace ryujin */
