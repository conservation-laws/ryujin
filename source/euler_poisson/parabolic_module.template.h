//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include "laplace_operator.h"
#include "parabolic_module.h"

#include <convenience_macros.h>
#include <instrumentation.h>
#include <openmp.h>
#include <scope.h>
#include <simd.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools_interpolate.h>


namespace ryujin
{
  namespace EulerPoisson
  {
    using namespace dealii;

    template <int dim, typename Number>
    ParabolicModule<dim, Number>::ParabolicModule(
        const MPIEnsemble &mpi_ensemble,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const ParabolicSystem &parabolic_system,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection)
        : ParameterAcceptor(subsection)
        , mpi_ensemble_(mpi_ensemble)
        , computing_timer_(computing_timer)
        , hyperbolic_system_(&hyperbolic_system)
        , parabolic_system_(&parabolic_system)
        , offline_data_(&offline_data)
        , initial_values_(&initial_values)
        , id_violation_strategy_(IDViolationStrategy::warn)
        , cycle_(0)
        , n_iterations_gauss_(0)
        , n_iterations_step_(0)
        , n_restarts_(0)
        , n_corrections_(0)
        , n_warnings_(0)
        , potential_initialized_(false)
        , t_background_density_(std::numeric_limits<Number>::lowest())
        , t_magnetic_field_(std::numeric_limits<Number>::lowest())
    {
      gauss_law_restart_strategy_ = GaussLawRestartStrategy::no_restart;
      add_parameter("gauss law restart strategy",
                    gauss_law_restart_strategy_,
                    "Strategy used when restarting the gauss law. Options are "
                    "\'no restart\', \'full restart\', \'correction\', "
                    "\'static no restart\', and \'static full restart\'.");

      gmg_max_iter_ = 15;
      add_parameter("multigrid - max iter",
                    gmg_max_iter_,
                    "Maximal number of CG iterations with GMG smoother");

      gmg_smoother_range_ = 8.;
      add_parameter("multigrid - chebyshev range",
                    gmg_smoother_range_,
                    "Chebyshev smoother: eigenvalue range parameter");

      gmg_smoother_max_eig_ = 2.0;
      add_parameter("multigrid - chebyshev max eig",
                    gmg_smoother_max_eig_,
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

      ElectrostaticConfigurationLibrary::
          populate_electrostatic_configuration_list<dim, Number>(
              electrostatic_configuration_list_,
              parabolic_system_->subsection());

      const auto populate = [this]() {
        bool initialized = false;
        for (auto &it : electrostatic_configuration_list_)

          if (it->name() == parabolic_system_->electrostatic_configuration()) {
            selected_electrostatic_configuration_ = it;
            initialized = true;
            break;
          }

        AssertThrow(initialized,
                    dealii::ExcMessage(
                        "Could not find an electrostatic configuration "
                        "description with name \"" +
                        parabolic_system_->electrostatic_configuration() +
                        "\""));
      };

      ParameterAcceptor::parse_parameters_call_back.connect(populate);
      populate();
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::prepare()" << std::endl;
#endif
      /*
       * The cycle_ variabe is only used for gmg reinitialization, simply
       * reset it to zero on prepare().
       */
      cycle_ = 0;

      const auto &discretization = offline_data_->discretization();
      AssertThrow(discretization.ansatz() == Ansatz::dg_q1 ||
                      discretization.ansatz() == Ansatz::cg_q1,
                  dealii::ExcMessage("The Euler-Poisson module currently only "
                                     "supports cG/dg Q1 finite elements."));

      AssertThrow(!offline_data_->dof_handler().has_hp_capabilities(),
                  dealii::ExcMessage(
                      "The Euler-Poisson module currently does not support "
                      "DoFHandlers set up with hp capabilities."));

      potential_initialized_ = false;

      /*
       * (Re)initialize matrix free object:
       */

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
          MatrixFree<dim, Number>::AdditionalData::none;

      // First index CG, second index hyperbolic ansatz
      std::vector<const dealii::DoFHandler<dim> *> dof_handlers = {
          &offline_data_->dof_handler_cg(), &offline_data_->dof_handler()};

      create_constraints();
      std::vector<const dealii::AffineConstraints<Number> *>
          affine_constraints = {&affine_constraints_potential_,
                                &offline_data_->affine_constraints()};

      // First index full quadrature, second index lumped quadrature
      std::vector<dealii::Quadrature<1>> quadratures = {
          discretization.quadrature_1d()[0],
          discretization.nodal_quadrature_1d()[0]};

      matrix_free_.reinit(discretization.mapping(),
                          dof_handlers,
                          affine_constraints,
                          quadratures,
                          additional_data);

      /*
       * (Re)initialize operators and preconditioners:
       */

      laplace_operator_.initialize(matrix_free_);
      laplace_operator_.compute_diagonal(diagonal_preconditioner_);
      update_operator_.initialize(matrix_free_, density_, magnetic_field_);

      typename decltype(multigrid_preconditioner_)::MultigridParameters
          parameters{gmg_max_iter_,
                     gmg_smoother_range_,
                     gmg_smoother_max_eig_,
                     gmg_smoother_degree_,
                     gmg_smoother_n_cg_iter_,
                     gmg_min_level_,
                     tolerance_};

      multigrid_preconditioner_.initialize(
          *offline_data_,
          selected_electrostatic_configuration_->dirichlet_boundaries(),
          parameters);

      /*
       * (Re)initialize auxiliary vectors:
       */

      const auto &potential_partitioner =
          matrix_free_.get_dof_info(0).vector_partitioner;
      potential_rhs_.reinit(potential_partitioner);

      const auto &scalar_partitioner =
          matrix_free_.get_dof_info(1).vector_partitioner;
      density_.reinit(scalar_partitioner);
      background_density_.reinit(scalar_partitioner);

      magnetic_field_.reinit(dim == 2 ? 1 : dim);
      for (unsigned int i = 0; i < magnetic_field_.n_blocks(); ++i)
        magnetic_field_.block(i).reinit(scalar_partitioner);

      velocity_rhs_.reinit(dim);
      for (unsigned int i = 0; i < dim; ++i)
        velocity_rhs_.block(i).reinit(scalar_partitioner);

      /*
       * Populate background fields:
       */

      if (!selected_electrostatic_configuration_->is_time_dependent()) {
        update_background_density(Number(0.));
        update_magnetic_field(Number(0.));
      }
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::reinit_state_vector(
        StateVector &state_vector) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::reinit_state_vector()"
                << std::endl;
#endif

      auto &[U, precomputed, V] = state_vector;
      V.reinit(1);

      auto &potential = V.block(0);
      const auto &partitioner = matrix_free_.get_dof_info(0).vector_partitioner;
      potential.reinit(partitioner);
      potential = 0.;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::prepare_state_vector(
        StateVector &state_vector, Number t) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::prepare_state_vector()"
                << std::endl;
#endif

      /*
       * We (re)compute the potential on the first step and if the restart
       * strategy is set to full_restart or static_full_restart.
       */

      AssertThrow(gauss_law_restart_strategy_ !=
                      GaussLawRestartStrategy::correction,
                  dealii::ExcNotImplemented());

      if (!potential_initialized_ ||
          (gauss_law_restart_strategy_ ==
           GaussLawRestartStrategy::full_restart) ||
          (gauss_law_restart_strategy_ ==
           GaussLawRestartStrategy::static_full_restart)) {

        compute_potential(t, state_vector);

        if (!potential_initialized_ &&
            parabolic_system_->magnetic_drift_limit())
          enforce_magnetic_drift_velocity(state_vector);
        potential_initialized_ = true;
      }
    }


    template <int dim, typename Number>
    template <int stages>
    void ParabolicModule<dim, Number>::backward_euler_step(
        const StateVector &old_state_vector,
        const Number old_t,
        std::array<std::reference_wrapper<const StateVector>,
                   stages> /*stage_state_vectors*/,
        const std::array<Number, stages> /*stage_weights*/,
        StateVector &new_state_vector,
        Number tau) const
    {
      step(old_state_vector,
           old_t,
           new_state_vector,
           tau,
           /*crank_nicolson_extrapolation = */ false);
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::crank_nicolson_step(
        const StateVector &old_state_vector,
        const Number old_t,
        StateVector &new_state_vector,
        Number tau) const
    {
      try {
        /* Backward Euler step to half time step, and extrapolate: */

        step(old_state_vector,
             old_t,
             new_state_vector,
             tau / Number(2.),
             /*crank_nicolson_extrapolation = */ true);

      } catch (Correction) {

        /*
         * Under very rare circumstances we might fail to perform a Crank
         * Nicolson step because the extrapolation step produced
         * inadmissible states. We could correct the update now by
         * performing a limiting step (either convex limiting, or flux
         * corrected transport)... but *meh*, just perform a backward Euler
         * step:
         */
        step(old_state_vector,
             old_t,
             new_state_vector,
             tau,
             /*crank_nicolson_extrapolation = */ false);
      }
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::create_constraints()
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::create_constraints()"
                << std::endl;
#endif

      const auto &discretization = offline_data_->discretization();
      const auto &dof_handler = offline_data_->dof_handler_cg();

      affine_constraints_potential_.clear();

#if DEAL_II_VERSION_GTE(9, 6, 0)
      const auto locally_relevant =
          DoFTools::extract_locally_relevant_dofs(dof_handler);
#else
      IndexSet locally_relevant;
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant);
#endif

#if DEAL_II_VERSION_GTE(9, 6, 0)
      const IndexSet &locally_owned = dof_handler.locally_owned_dofs();
      affine_constraints_potential_.reinit(locally_owned, locally_relevant);
#else
      affine_constraints_potential_.reinit(locally_relevant);
#endif

      DoFTools::make_hanging_node_constraints(offline_data_->dof_handler_cg(),
                                              affine_constraints_potential_);

      /*
       * Enforce periodic boundary conditions. We assume that the mesh is in
       * "normal configuration."
       */

      const auto &periodic_faces =
          discretization.triangulation().get_periodic_face_map();

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

        if constexpr (std::is_same_v<Number, double>) {
          DoFTools::make_periodicity_constraints(
              dof_cell_left->face(left.second),
              dof_cell_right->face(right.second),
              affine_constraints_potential_,
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

      for (const auto &it :
           selected_electrostatic_configuration_->dirichlet_boundaries())
        DoFTools::make_zero_boundary_constraints(
            offline_data_->dof_handler_cg(), it, affine_constraints_potential_);

      affine_constraints_potential_.close();
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::update_background_density(
        const Number t) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::update_background_density()"
                << std::endl;
#endif

      /*
       * Skip updating the background density if t > 0 and if the fields
       * are time independent:
       */
      if (!selected_electrostatic_configuration_->is_time_dependent() &&
          (t > Number(0.)))
        return;

      /*
       * Skip updating if we have already populated the background density
       * for the chosen time t.
       */
      if (std::abs(t_background_density_ - t) < 1.e-12)
        return;

#ifdef DEBUG_OUTPUT
      std::cout << "        updating to t = " << t << std::endl;
#endif

      Scope scope(computing_timer_,
                  "time step [X]   - interpolate data vectors");

      const auto &discretization = offline_data_->discretization();
      background_density_.zero_out_ghost_values();
      dealii::VectorTools::interpolate(
          discretization.mapping(),
          offline_data_->dof_handler(),
          dealii::ScalarFunctionFromFunctionObject<dim, Number>(
              [&](const dealii::Point<dim> &p) {
                return selected_electrostatic_configuration_
                    ->background_density(p, t);
              }),
          background_density_);
      background_density_.update_ghost_values();

      t_background_density_ = t;
    }


    template <int dim, typename Number>
    void
    ParabolicModule<dim, Number>::update_magnetic_field(const Number t) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::update_magnetic_field()"
                << std::endl;
#endif

      /*
       * Skip updating the background density if t > 0 and if the fields
       * are time independent:
       */
      if (!selected_electrostatic_configuration_->is_time_dependent() &&
          (t > Number(0.)))
        return;

      /*
       * Skip updating if we have already populated the background density
       * for the chosen time t.
       */
      if (std::abs(t_magnetic_field_ - t) < 1.e-12)
        return;

#ifdef DEBUG_OUTPUT
      std::cout << "        updating to t = " << t << std::endl;
#endif

      Scope scope(computing_timer_,
                  "time step [X]   - interpolate data vectors");

      const auto &discretization = offline_data_->discretization();
      for (unsigned int k = 0; k < (dim == 2 ? 1 : dim); ++k) {
        magnetic_field_.block(k).zero_out_ghost_values();
        dealii::VectorTools::interpolate(
            discretization.mapping(),
            offline_data_->dof_handler(),
            to_function<dim, Number>(
                [&](const dealii::Point<dim> &p) {
                  return selected_electrostatic_configuration_->magnetic_field(
                      p, t);
                },
                k),
            magnetic_field_.block(k));
      }
      magnetic_field_.update_ghost_values();

      t_magnetic_field_ = t;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::compute_potential(
        const Number t, StateVector &state_vector) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::compute_potential()"
                << std::endl;
#endif
      auto &U = std::get<0>(state_vector);
      auto &V = std::get<2>(state_vector);
      auto &potential = V.block(0);

      using VA = VectorizedArray<Number>;
      constexpr auto simd_length = VA::size();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const unsigned int n_regular = n_owned / simd_length * simd_length;

      constexpr unsigned int order_fe = 1;
      constexpr unsigned int order_quad = 2;

      /*
       * -----------------------------------------------------------------------
       * Step 1a: build right hand side for Gauss law
       * -----------------------------------------------------------------------
       */

      Scope scope(computing_timer_, "time step [P] 1 - enforce Gauss law");
      LIKWID_MARKER_START("time_step_parabolic_1a");

      update_background_density(t);

      RYUJIN_PARALLEL_REGION_BEGIN

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;
        const auto view = hyperbolic_system_->template view<dim, T>();

        RYUJIN_OMP_FOR
        for (unsigned int i = left; i < right; i += stride_size) {
          const auto U_i = U.template get_tensor<T>(i);
          const auto rho_i = view.density(U_i);
          write_entry<T>(density_, rho_i, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_regular, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_regular);

      RYUJIN_PARALLEL_REGION_END

      density_.update_ghost_values();

      const auto body = [this](const auto &data,
                               auto &dst,
                               const auto &src,
                               const auto range) {
        FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
            fee_potential(data, /*CG*/ 0, /*lumped quadrature*/ 1);
        FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
            fee_density(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);
        FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
            fee_background(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);


        const Number alpha = parabolic_system_->alpha();

        for (unsigned int cell = range.first; cell < range.second; ++cell) {
          fee_potential.reinit(cell);
          fee_density.reinit(cell);
          fee_background.reinit(cell);

          fee_density.gather_evaluate(src, dealii::EvaluationFlags::values);
          fee_background.gather_evaluate(background_density_,
                                         dealii::EvaluationFlags::values);

          for (unsigned int q = 0; q < fee_potential.n_q_points; ++q) {
            const auto density_q = fee_density.get_value(q);
            const auto background_q = fee_background.get_value(q);

            const auto value = alpha * (density_q + background_q);
            fee_potential.submit_value(value, q);
          }
          fee_potential.integrate_scatter(dealii::EvaluationFlags::values, dst);
        }
      };

      matrix_free_.template cell_loop<ScalarVector, ScalarVector>(
          body,
          potential_rhs_,
          density_,
          /*zero destination*/ true);

      LIKWID_MARKER_STOP("time_step_parabolic_1a");

      /*
       * -----------------------------------------------------------------------
       * Step 1b: solve Poisson problem
       * -----------------------------------------------------------------------
       */

      LIKWID_MARKER_START("time_step_parabolic_1b");

      const auto tolerance =
          (tolerance_linfty_norm_ ? potential_rhs_.linfty_norm()
                                  : potential_rhs_.l2_norm()) *
          tolerance_;

      matrix_free_.get_affine_constraints(0).distribute(potential);
      matrix_free_.get_affine_constraints(0).set_zero(potential_rhs_);

      try {
        SolverControl solver_control(gmg_max_iter_, tolerance);
        SolverCG<ScalarVector> solver(solver_control);
        solver.solve(laplace_operator_,
                     potential,
                     potential_rhs_,
                     multigrid_preconditioner_);


        if (potential_initialized_) {
          /* update exponential moving average */
          n_iterations_gauss_ =
              0.9 * n_iterations_gauss_ + 0.1 * solver_control.last_step();
        } else {
          n_iterations_gauss_ = solver_control.last_step();
        }

      } catch (SolverControl::NoConvergence &) {
        SolverControl solver_control(1000, tolerance);
        SolverCG<ScalarVector> solver(solver_control);

        solver.solve(laplace_operator_,
                     potential,
                     potential_rhs_,
                     diagonal_preconditioner_);

        if (potential_initialized_) {
          /* update exponential moving average */
          n_iterations_gauss_ *= 0.9;
          n_iterations_gauss_ +=
              0.1 * gmg_max_iter_ + 0.1 * solver_control.last_step();
        } else {
          n_iterations_gauss_ = gmg_max_iter_ + solver_control.last_step();
        }

        /* update exponential moving average, counting also GMG iterations */
      }

      matrix_free_.get_affine_constraints(0).distribute(potential);

      LIKWID_MARKER_STOP("time_step_parabolic_1b");
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::enforce_magnetic_drift_velocity(
        StateVector &state_vector) const
    {
#ifdef DEBUG_OUTPUT
      std::cout
          << "ParabolicModule<dim, Number>::enforce_magnetic_drift_velocity()"
          << std::endl;
#endif

      auto &U = std::get<0>(state_vector);
      auto &V = std::get<2>(state_vector);
      auto &potential = V.block(0);

      using VA = VectorizedArray<Number>;
      constexpr auto simd_length = VA::size();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const unsigned int n_regular = n_owned / simd_length * simd_length;

      const auto &lumped_mass_matrix_inverse =
          offline_data_->lumped_mass_matrix_inverse();

      constexpr unsigned int order_fe = 1;
      constexpr unsigned int order_quad = 2;

      /*
       * -----------------------------------------------------------------------
       * Step 1c: enforce magnetic drift velocity
       * -----------------------------------------------------------------------
       */

      LIKWID_MARKER_START("time_step_parabolic_1b");

      update_magnetic_field(Number(0.));

      /* Project gradient of potential into velocity space: */

      const auto body_velocity =
          [](const auto &data, auto &dst, const auto &src, const auto range) {
            FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
                fee_pot(data, /*CG*/ 0, /*lumped quadrature*/ 1);
            FEEvaluation<dim, order_fe, order_quad, /*components*/ dim, Number>
                fee_vel(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);

            for (unsigned int cell = range.first; cell < range.second; ++cell) {
              fee_pot.reinit(cell);
              fee_vel.reinit(cell);

              fee_pot.gather_evaluate(src, dealii::EvaluationFlags::gradients);
              for (unsigned int q = 0; q < fee_pot.n_q_points; ++q) {
                fee_vel.submit_value(fee_pot.get_gradient(q), q);
              }
              fee_vel.integrate_scatter(dealii::EvaluationFlags::values, dst);
            }
          };

      matrix_free_.template cell_loop<BlockVector, ScalarVector>(
          body_velocity,
          velocity_rhs_,
          potential,
          /*zero destination*/ true);

      RYUJIN_PARALLEL_REGION_BEGIN

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;
        const auto view = hyperbolic_system_->template view<dim, T>();

        RYUJIN_OMP_FOR
        for (unsigned int i = left; i < right; i += stride_size) {
          const auto m_i_inv = get_entry<T>(lumped_mass_matrix_inverse, i);

          auto U_i = U.template get_tensor<T>(i);
          const auto rho_i = view.density(U_i);
          const auto m_i = view.momentum(U_i);
          const auto v_i = m_i / rho_i;

          dealii::Tensor<1, (dim == 2 ? 1 : dim), T> magnetic_field;
          for (unsigned int d = 0; d < (dim == 2 ? 1 : dim); ++d)
            magnetic_field[d] = get_entry<T>(magnetic_field_.block(d), i);

          dealii::Tensor<1, dim, T> grad_phi;
          for (unsigned int d = 0; d < dim; ++d)
            grad_phi[d] = m_i_inv * get_entry<T>(velocity_rhs_.block(d), i);

          auto new_v_i = v_i;

          if constexpr (dim == 2) {
            new_v_i = -magnetic_field[0] * cross_product_2d(grad_phi) /
                      magnetic_field.norm_square();

          } else if constexpr (dim == 3) {
            new_v_i = -cross_product_3d(grad_phi, magnetic_field) /
                      magnetic_field.norm_square();
          }

          for (unsigned int d = 0; d < dim; ++d)
            U_i[1 + d] = rho_i * new_v_i[d];

          /*
           * Update the total energy accordingly:
           */
          if constexpr (view.have_energy_equation) {
            U_i[1 + dim] += Number(0.5) * rho_i *
                            (new_v_i.norm_square() - v_i.norm_square());
          }

          U.template write_tensor<T>(U_i, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_regular, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_regular);

      RYUJIN_PARALLEL_REGION_END




      LIKWID_MARKER_STOP("time_step_parabolic_1c");
    }


    template <int dim, typename Number>
    void
    ParabolicModule<dim, Number>::step(const StateVector &old_state_vector,
                                       const Number t,
                                       StateVector &new_state_vector,
                                       Number tau [[maybe_unused]],
                                       const bool crank_nicolson_extrapolation
                                       [[maybe_unused]]) const
    {
#ifdef DEBUG_OUTPUT
      std::cout << "ParabolicModule<dim, Number>::step()" << std::endl;
      std::cout << "        perform time-step with tau = " << tau << std::endl;
      if (crank_nicolson_extrapolation)
        std::cout << "        and extrapolate to t + 2 * tau" << std::endl;
#endif

      const Number alpha = parabolic_system_->alpha();

      const auto &old_U = std::get<0>(old_state_vector);
      const auto &old_V = std::get<2>(old_state_vector);
      const auto &old_potential = old_V.block(0);

      auto &new_U = std::get<0>(new_state_vector);
      auto &new_V = std::get<2>(new_state_vector);
      auto &new_potential = new_V.block(0);

      using VA = VectorizedArray<Number>;
      constexpr auto simd_length = VA::size();
      const unsigned int n_owned = offline_data_->n_locally_owned();
      const unsigned int n_regular = n_owned / simd_length * simd_length;

      const auto &lumped_mass_matrix_inverse =
          offline_data_->lumped_mass_matrix_inverse();

      constexpr unsigned int order_fe = 1;
      constexpr unsigned int order_quad = 2;

      /*
       * Initialize the new potential with the old one:
       */

      new_potential = old_potential;

      /*
       * If the Gauss law restart strategy is "static full restart" or
       * "static no restart", we skip updating the potential.
       */
      if ((gauss_law_restart_strategy_ !=
           GaussLawRestartStrategy::static_no_restart) &&
          (gauss_law_restart_strategy_ !=
           GaussLawRestartStrategy::static_full_restart)) {

        /*
         * ---------------------------------------------------------------------
         * Step 2a: build right hand side for potential update
         *
         * The right-hand side reads:
         *   (\nabla \varphi^n, \nabla \chi) +
         *   \tau \alpha \langle \rho^n B^{-1} v^n, \nabla \chi \rangle
         *
         * In case of a time-dependent background density, we add a term
         * \theta \alpha \langle \rho_b^{n+1} - \rho_b^n, \chi \rangle to
         * account for the time dependence. Here, t_{n+1} is the final time
         * t_n + tau, or t_n + 2 * tau (in case of Crank Nicolson). This
         * ensures that we are consistent with the Gauß law involution
         * "-\Delta \varphi^{n+1} = \alpha \rho^{n+1}."
         * ---------------------------------------------------------------------
         */

        Scope scope(computing_timer_, "time step [P] 2 - update potential");
        LIKWID_MARKER_START("time_step_parabolic_2a");

        /* Query the magnetic field at the time t + tau: */
        update_magnetic_field(t + tau);

        /*
         * Write out density and assemble velocity part. We need density_
         * to be set to the correct density for UpdateOperator::vmult()
         */

        RYUJIN_PARALLEL_REGION_BEGIN

        auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
          using T = decltype(sentinel);
          unsigned int stride_size = get_stride_size<T>;
          const auto view = hyperbolic_system_->template view<dim, T>();

          RYUJIN_OMP_FOR
          for (unsigned int i = left; i < right; i += stride_size) {
            const auto U_i = old_U.template get_tensor<T>(i);
            const auto rho_i = view.density(U_i);
            const auto m_i = view.momentum(U_i);

            dealii::Tensor<1, (dim == 2 ? 1 : dim), T> magnetic_field;
            for (unsigned int d = 0; d < (dim == 2 ? 1 : dim); ++d)
              magnetic_field[d] = get_entry<T>(magnetic_field_.block(d), i);

            const auto velocity_rhs =
                tau * alpha * rho_i *
                apply_B_n_inverse(magnetic_field, tau, m_i / rho_i);

            write_entry<T>(density_, rho_i, i);
            for (unsigned int d = 0; d < dim; ++d)
              write_entry<T>(velocity_rhs_.block(d), velocity_rhs[d], i);
          }
        };

        /* Parallel non-vectorized loop: */
        loop(Number(), n_regular, n_owned);
        /* Parallel vectorized SIMD loop: */
        loop(VA(), 0, n_regular);

        RYUJIN_PARALLEL_REGION_END

        density_.update_ghost_values();

        /* Apply Laplace operator to right hand side: */

        const auto body_laplace = [](const auto &data,
                                     auto &dst,
                                     const auto &src,
                                     const auto range) {
          FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number> fee(
              data, /*CG*/ 0, /*full quadrature*/ 0);

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            fee.reinit(cell);
            fee.gather_evaluate(src, dealii::EvaluationFlags::gradients);

            for (unsigned int q = 0; q < fee.n_q_points; ++q) {
              const auto grad_potential = fee.get_gradient(q);
              fee.submit_gradient(grad_potential, q);
            }
            fee.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
          }
        };

        matrix_free_.template cell_loop<ScalarVector, ScalarVector>(
            body_laplace,
            potential_rhs_,
            old_potential,
            /*zero destination*/ true);

        /* Apply Velocity contribution to right hand side: */

        const auto body_velocity = [](const auto &data,
                                      auto &dst,
                                      const auto &src,
                                      const auto range) {
          FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
              fee_pot(data, /*CG*/ 0, /*lumped quadrature*/ 1);
          FEEvaluation<dim, order_fe, order_quad, /*components*/ dim, Number>
              fee_vel(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            fee_pot.reinit(cell);
            fee_vel.reinit(cell);

            fee_vel.gather_evaluate(src, dealii::EvaluationFlags::values);

            for (unsigned int q = 0; q < fee_pot.n_q_points; ++q) {
              if constexpr (dim == 1) {
                decltype(fee_pot.get_gradient(q)) velocity_rhs;
                velocity_rhs[0] = fee_vel.get_value(q);
                fee_pot.submit_gradient(velocity_rhs, q);
              } else {
                fee_pot.submit_gradient(fee_vel.get_value(q), q);
              }
            }
            fee_pot.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
          }
        };

        matrix_free_.template cell_loop<ScalarVector, BlockVector>(
            body_velocity,
            potential_rhs_,
            velocity_rhs_,
            /*zero destination*/ false);

        LIKWID_MARKER_STOP("time_step_parabolic_2a");

        /* Time-dependent background density: */

        if (selected_electrostatic_configuration_->is_time_dependent()) {

          /*
           * Subtract background density at time t_n:
           */

          update_background_density(t);

          Number factor = (crank_nicolson_extrapolation ? -0.5 : -1.0) * alpha;

          const auto body = [&factor](const auto &data,
                                      auto &dst,
                                      const auto &src,
                                      const auto range) {
            FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
                fee_potential(data, /*CG*/ 0, /*lumped quadrature*/ 1);
            FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
                fee_background(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);

            for (unsigned int cell = range.first; cell < range.second; ++cell) {
              fee_potential.reinit(cell);
              fee_background.reinit(cell);
              fee_background.gather_evaluate(src, EvaluationFlags::values);

              for (unsigned int q = 0; q < fee_potential.n_q_points; ++q) {
                const auto background_q = fee_background.get_value(q);
                fee_potential.submit_value(factor * background_q, q);
              }
              fee_potential.integrate_scatter(EvaluationFlags::values, dst);
            }
          };

          matrix_free_.template cell_loop<ScalarVector, ScalarVector>(
              body,
              potential_rhs_,
              background_density_,
              /*zero destination*/ false);

          /*
           * Add background density at time t_{n+1}:
           */

          update_background_density(
              t + (crank_nicolson_extrapolation ? 2. : 1.) * tau);

          factor *= -1.;

          matrix_free_.template cell_loop<ScalarVector, ScalarVector>(
              body,
              potential_rhs_,
              background_density_,
              /*zero destination*/ false);
        }

        /*
         * ---------------------------------------------------------------------
         * Step 2b: solve modified poisson problem
         * ---------------------------------------------------------------------
         */

        LIKWID_MARKER_START("time_step_parabolic_2b");

        update_operator_.set_alpha(alpha);
        update_operator_.set_theta_tau(tau);

        const auto tolerance =
            (tolerance_linfty_norm_ ? potential_rhs_.linfty_norm()
                                    : potential_rhs_.l2_norm()) *
            tolerance_;

        matrix_free_.get_affine_constraints(0).distribute(new_potential);
        matrix_free_.get_affine_constraints(0).set_zero(potential_rhs_);

        try {
          SolverControl solver_control(gmg_max_iter_, tolerance);
          SolverCG<ScalarVector> solver(solver_control);
          solver.solve(update_operator_,
                       new_potential,
                       potential_rhs_,
                       multigrid_preconditioner_);

          /* update exponential moving average */
          n_iterations_step_ =
              0.9 * n_iterations_step_ + 0.1 * solver_control.last_step();

        } catch (SolverControl::NoConvergence &) {
          SolverControl solver_control(1000, tolerance);
          SolverCG<ScalarVector> solver(solver_control);

          solver.solve(update_operator_,
                       new_potential,
                       potential_rhs_,
                       diagonal_preconditioner_);

          /* update exponential moving average, counting also GMG iterations */
          n_iterations_step_ *= 0.9;
          n_iterations_step_ +=
              0.1 * gmg_max_iter_ + 0.1 * solver_control.last_step();
        }

        matrix_free_.get_affine_constraints(0).distribute(new_potential);

        LIKWID_MARKER_STOP("time_step_parabolic_2b");
      }

      /*
       * ---------------------------------------------------------------------
       * Step 2c: update velocity vector field; Crank-Nicolson extrapolation
       * ---------------------------------------------------------------------
       */

      LIKWID_MARKER_START("time_step_parabolic_2c");

      /* Project gradient of potential into velocity space: */

      const auto body_velocity =
          [](const auto &data, auto &dst, const auto &src, const auto range) {
            FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
                fee_pot(data, /*CG*/ 0, /*lumped quadrature*/ 1);
            FEEvaluation<dim, order_fe, order_quad, /*components*/ dim, Number>
                fee_vel(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);

            for (unsigned int cell = range.first; cell < range.second; ++cell) {
              fee_pot.reinit(cell);
              fee_vel.reinit(cell);

              fee_pot.gather_evaluate(src, dealii::EvaluationFlags::gradients);
              for (unsigned int q = 0; q < fee_pot.n_q_points; ++q) {
                fee_vel.submit_value(fee_pot.get_gradient(q), q);
              }
              fee_vel.integrate_scatter(dealii::EvaluationFlags::values, dst);
            }
          };

      matrix_free_.template cell_loop<BlockVector, ScalarVector>(
          body_velocity,
          velocity_rhs_,
          new_potential,
          /*zero destination*/ true);

      /*
       * Now that we have written out the gradients, copy over the old
       * state vector and perform the Crank-Nicolson extrapolation step on
       * the potential:
       */

      new_U = old_U;

      if (crank_nicolson_extrapolation) {
        new_potential *= Number(2.);
        new_potential -= old_potential;
      }

      /*
       * Update the momentum and total energy:
       */

      RYUJIN_PARALLEL_REGION_BEGIN

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;
        const auto view = hyperbolic_system_->template view<dim, T>();

        RYUJIN_OMP_FOR
        for (unsigned int i = left; i < right; i += stride_size) {
          const auto m_i_inv = get_entry<T>(lumped_mass_matrix_inverse, i);

          const auto old_U_i = old_U.template get_tensor<T>(i);
          const auto rho_i = view.density(old_U_i);
          const auto old_m_i = view.momentum(old_U_i);
          const auto old_v_i = old_m_i / rho_i;

          dealii::Tensor<1, (dim == 2 ? 1 : dim), T> magnetic_field;
          for (unsigned int d = 0; d < (dim == 2 ? 1 : dim); ++d)
            magnetic_field[d] = get_entry<T>(magnetic_field_.block(d), i);

          dealii::Tensor<1, dim, T> grad_phi;
          for (unsigned int d = 0; d < dim; ++d)
            grad_phi[d] = m_i_inv * get_entry<T>(velocity_rhs_.block(d), i);

          auto new_v_i =
              apply_B_n_inverse(magnetic_field, tau, old_v_i - tau * grad_phi);

          /* Perform an extrapolation step: */
          if (crank_nicolson_extrapolation)
            new_v_i = Number(2.) * new_v_i - old_v_i;

          auto new_U_i = old_U_i;
          for (unsigned int d = 0; d < dim; ++d)
            new_U_i[1 + d] = rho_i * new_v_i[d];

          /*
           * Update the total energy accordingly:
           */
          if constexpr (view.have_energy_equation) {
            new_U_i[1 + dim] += Number(0.5) * rho_i *
                                (new_v_i.norm_square() - old_v_i.norm_square());
          }

          new_U.template write_tensor<T>(new_U_i, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_regular, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_regular);

      RYUJIN_PARALLEL_REGION_END

      LIKWID_MARKER_STOP("time_step_parabolic_2c");

      return;
    }


    template <int dim, typename Number>
    void ParabolicModule<dim, Number>::print_solver_statistics(
        std::ostream &output) const
    {
      output << "        [ " << std::setprecision(2) << std::fixed //
             << n_iterations_gauss_ << " GMG gauss -- "            //
             << n_iterations_step_ << " GMG step ]" << std::endl;
    }

  } // namespace EulerPoisson
} /* namespace ryujin */
