//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/euler_poisson/electrostatic_configuration_library.h>
#include <ryujin/euler_poisson/laplace_operator.h>
#include <ryujin/euler_poisson/parabolic_system.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/typed/initial_values.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <limits>
#include <memory>
#include <vector>

namespace ryujin
{
  namespace EulerPoisson
  {
    /**
     * Controls the chosen Gauss law restart strategy.
     *
     * @ingroup EulerPoissonEquations
     */
    enum GaussLawRestartStrategy : dealii::types::boundary_id {
      /**
       * Initialize the potential by solving
       * \f[
       *   -\Delta \varphi = \alpha (\rho + \rho_b)
       * \f]
       * once at the first timestep. Then, solve
       * \f[
       *   -\partial_t \Delta \varphi = - \alpha \nabla \cdot \boldsymbol m +
       *   \partial_t \rho_b
       * \f]
       * at each parabolic substep. Do not restart the Gauss law. This is
       * the default strategy.
       */
      no_restart,

      /**
       * Initialize the potential by solving
       * \f[
       *   -\Delta \varphi = \alpha (\rho + \rho_b)
       * \f]
       * once at the first timestep. Then, solve
       * \f[
       *   -\partial_t \Delta \varphi = - \alpha \nabla \cdot \boldsymbol m +
       *   \partial_t \rho_b
       * \f]
       * at each parabolic substep. Finally, at the beginning of each
       * subsequent time step, restart the Gauss law at the beginning of
       * each time step by solving again
       * \f[
       *   -\Delta \varphi = \alpha (\rho + \rho_b).
       * \f]
       *
       */
      full_restart,

      /**
       * Solve
       * \f[
       *   -\partial_t \Delta \varphi = -\alpha \nabla \cdot \boldsymbol m
       *   + \partial_t \rho_b
       * \f]
       * at each parabolic substep. Then, correct the Gauss law violation
       * by artificially relaxing the kinetic energy.
       *
       * FIXME documentation
       */
      correction,

      /**
       * Initialize the potential by solving
       * \f[
       * -\Delta \varphi = \alpha (\rho + \rho_b)
       * \f]
       * once at the first timestep. Afterwards, do not update the potential
       * any more.
       */
      static_no_restart,

      /**
       * Solve
       * \f[
       * -\Delta \varphi = \alpha (\rho + \rho_b)
       * \f]
       * at the beginning of each timestep. Do not update the potential
       * during a parabolic substep.
       */
      static_full_restart,

    };

  } // namespace EulerPoisson
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(
    ryujin::EulerPoisson::GaussLawRestartStrategy,
    LIST({ryujin::EulerPoisson::GaussLawRestartStrategy::no_restart,
          "no restart"},
         {ryujin::EulerPoisson::GaussLawRestartStrategy::full_restart,
          "full restart"},
         {ryujin::EulerPoisson::GaussLawRestartStrategy::correction,
          "correction"},
         {ryujin::EulerPoisson::GaussLawRestartStrategy::static_no_restart,
          "static no restart"},
         {ryujin::EulerPoisson::GaussLawRestartStrategy::static_full_restart,
          "static full restart"}));
#endif

namespace ryujin
{
  namespace EulerPoisson
  {
    /**
     * Implicit backward-Euler time stepping for the parabolic limiting
     * equation for the Euler-Poisson system
     *
     * @ingroup EulerPoissonEquations
     */
    template <typename HyperbolicDescription, int dim, typename Number>
    class ParabolicModule final : public Interface::ParabolicModule<dim, Number>
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using StageVectors =
          typename Interface::ParabolicModule<dim, Number>::StageVectors;

      using StageWeights =
          typename Interface::ParabolicModule<dim, Number>::StageWeights;

      using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

      using InitialValues =
          typename HyperbolicDescription::template InitialValues<dim, Number>;

      using View = typename HyperbolicSystem::template View<dim, Number>;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      using BlockHostVector =
          dealii::LinearAlgebra::distributed::BlockVector<Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      ParabolicModule(const MPIEnsemble &mpi_ensemle,
                      const OfflineData<dim, Number> &offline_data,
                      const HyperbolicSystem &hyperbolic_system,
                      const ParabolicSystem &parabolic_system,
                      const InitialValues &initial_values,
                      const std::string &subsection = "/ParabolicModule");

      /**
       * Prepare time stepping. A call to @p prepare() allocates temporary
       * storage and is necessary before any of the following time-stepping
       * functions can be called.
       */
      void prepare() override;

      //@}
      /**
       * @name Functions for performing explicit time steps
       */
      //@{

      /**
       * (Re)initialize the parabolic state vector component of the state
       * vector.
       *
       * @note This routine does not modify the hyperbolic state vector or
       * the precomputed vector component.
       */
      void
      reinit_state_vector(StateVector<Number> &state_vector) const override;

      /**
       * This function preprocesses a given state vector @p U in preparation
       * for a high order IMEX time step. This function exists because some
       * time stepping variants have to precompute quantities before we can
       * perform an IMEX step. In addition, this function is called whenever
       * we perform a mesh transfer or output operation.
       */
      void prepare_state_vector(StateVector<Number> &state_vector,
                                Number t) const override;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p t and a time-step size @p tau perform an implicit
       * backward euler step (and store the result in @p new_state_vector).
       *
       * The function takes an optional list of states @p
       * stage_state_vectors together with a list of weights @p
       * stage_weights to construct a modified high-order right-hand side /
       * flux.
       */
      Number backward_euler_step(
          const StateVector<Number> &old_state_vector,
          StageVectors stage_state_vectors,
          StageWeights stage_weights,
          StateVector<Number> &new_state_vector,
          Number t,
          Number tau,
          Number tau_max = std::numeric_limits<Number>::max()) const override;

      /**
       * Given a reference to a previous state vector @p old_U at time @p
       * t and a time-step size @p tau perform an implicit Crank-Nicolson
       * step (and store the result in @p new_U).
       *
       * This variant is used in the TimeIntegrator class for the Strang
       * split variants.
       */
      Number crank_nicolson_step(
          const StateVector<Number> &old_state_vector,
          StateVector<Number> &new_state_vector,
          Number t,
          Number tau,
          Number tau_max = std::numeric_limits<Number>::max()) const override;

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Print a status line with solver statistics. This function is used
       * for constructing the status message displayed periodically in the
       * Scheduler.
       */
      void print_solver_statistics(std::ostream &output) const override;

    private:
      //@}
      /**
       * @name Run time options
       */
      //@{

      GaussLawRestartStrategy gauss_law_restart_strategy_;

      unsigned int gmg_max_iter_;
      double gmg_smoother_range_;
      double gmg_smoother_max_eig_;
      unsigned int gmg_smoother_degree_;
      unsigned int gmg_smoother_n_cg_iter_;
      unsigned int gmg_min_level_;

      Number tolerance_;
      bool tolerance_linfty_norm_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      // FIXME: refactor
      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      const MPIEnsemble &mpi_ensemble_;

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
      dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;
      dealii::ObserverPointer<const ryujin::OfflineData<dim, Number>>
          offline_data_;
      dealii::ObserverPointer<const InitialValues> initial_values_;

      std::vector<std::unique_ptr<ElectrostaticConfiguration<dim, Number>>>
          electrostatic_configuration_list_;

      ElectrostaticConfiguration<dim, Number>
          *selected_electrostatic_configuration_ = nullptr;

      mutable unsigned int cycle_;
      mutable double n_iterations_gauss_;
      mutable double n_iterations_step_;

      dealii::MatrixFree<dim, Number> matrix_free_;
      dealii::AffineConstraints<Number> affine_constraints_potential_;

      LaplaceOperator<dim, Number> laplace_operator_;
      dealii::DiagonalMatrix<ScalarHostVector<Number>> diagonal_preconditioner_;
      MGSmoother<dim, Number> multigrid_preconditioner_;
      UpdateOperator<dim, Number> update_operator_;

      mutable ScalarHostVector<Number> potential_rhs_;
      mutable ScalarHostVector<Number> density_;
      mutable ScalarHostVector<Number> background_density_;
      mutable BlockHostVector magnetic_field_;
      mutable BlockHostVector velocity_rhs_;

      mutable bool potential_initialized_;
      mutable Number t_background_density_;
      mutable Number t_magnetic_field_;
      //@}
      /**
       * @name Internal methods
       */
      //@{

      /**
       * The implementation of the backward_euler_step() function for a
       * fixed number of stages @p stages. The backward_euler_step()
       * function dispatches to it with the number of supplied stage state
       * vectors.
       */
      template <int stages>
      void backward_euler_step(const StateVector<Number> &old_state_vector,
                               StageVectors stage_state_vectors,
                               StageWeights stage_weights,
                               StateVector<Number> &new_state_vector,
                               Number t,
                               Number tau) const;

      /**
       * Set up affine constraints. Internally used in prepare().
       */
      void create_constraints();

      /**
       * Update the background density vector to time t.
       *
       * @note: The vector is always updated for t = 0, but the update is
       * skipped for t > 0 if the background fields are time independent.
       */
      void update_background_density(const Number t) const;

      /**
       * Update the (background) magnetic field vector to time t.
       *
       * @note: The vector is always updated for t = 0, but the update is
       * skipped for t > 0 if the background fields are time independent.
       */
      void update_magnetic_field(const Number t) const;

      /**
       * Compute the potential phi (the last component of the state_vector)
       * for a given density (the first component of the state_vector).
       */
      void compute_potential(const Number t,
                             StateVector<Number> &state_vector) const;

      /**
       * Set the velocity field equal to the magnetic magnetic drift
       * velocity computed from the potential.
       */
      void
      enforce_magnetic_drift_velocity(StateVector<Number> &state_vector) const;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p t and a time-step size @p tau perform a backward
       * Euler time step (and store the result in @p new_state_vector).
       *
       * If the boolean @crank_nicolson_extrapolation is set to true, then
       * we perform a final extrapolation on the primitive state for time
       * t + 2 * tau.
       */
      void step(const StateVector<Number> &old_state_vector,
                StateVector<Number> &new_state_vector,
                Number t,
                Number tau,
                const bool crank_nicolson_extrapolation) const;

      //@}
    };
  } // namespace EulerPoisson
} // namespace ryujin
