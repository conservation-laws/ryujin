//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "../euler/hyperbolic_system.h"

#include "electrostatic_configuration_library.h"
#include "laplace_operator.h"
#include "parabolic_system.h"

#include <hyperbolic_module.h>
#include <initial_values.h>
#include <mpi_ensemble.h>
#include <offline_data.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>

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
       * $$
       *   -\Delta \varphi = \alpha (\rho + \rho_b)
       * $$
       * once at the first timestep. Then, solve
       * $$
       *   -\partial_t \Delta \varphi = - \alpha \nabla \cdot \boldsymbol m +
       *   \partial_t \rho_b
       * $$
       * at each parabolic substep. Do not restart the Gauss law. This is
       * the default strategy.
       */
      no_restart,

      /**
       * Initialize the potential by solving
       * $$
       *   -\Delta \varphi = \alpha (\rho + \rho_b)
       * $$
       * once at the first timestep. Then, solve
       * $$
       *   -\partial_t \Delta \varphi = - \alpha \nabla \cdot \boldsymbol m +
       *   \partial_t \rho_b
       * $$
       * at each parabolic substep. Finally, at the beginning of each
       * subsequent time step, restart the Gauss law at the beginning of
       * each time step by solving again
       * $$
       *   -\Delta \varphi = \alpha (\rho + \rho_b).
       * $$
       *
       */
      full_restart,

      /**
       * Solve
       * $$
       *   -\partial_t \Delta \varphi = -\alpha \nabla \cdot \boldsymbol m
       *   + \partial_t \rho_b
       * $$
       * at each parabolic substep. Then, correct the Gauss law violation
       * by artificially relaxing the kinetic energy.
       *
       * FIXME documentation
       */
      correction,

      /**
       * Initialize the potential by solving
       * $$
       * -\Delta \varphi = \alpha (\rho + \rho_b)
       * $$
       * once at the first timestep. Afterwards, do not update the potential
       * any more.
       */
      static_no_restart,

      /**
       * Solve
       * $$
       * -\Delta \varphi = \alpha (\rho + \rho_b)
       * $$
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
    /* Forward declaration: */
    struct Description;

    /**
     * Implicit backward-Euler time stepping for the parabolic limiting
     * equation for the Euler-Poisson system
     *
     * @ingroup ParabolicModule
     */
    template <int dim, typename Number>
    class ParabolicModule final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using HyperbolicSystem = Euler::HyperbolicSystem;

      using View = Euler::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = EulerPoisson::ParabolicSystem;

      using StateVector = typename View::StateVector;

      using ScalarVector = Vectors::ScalarVector<Number>;

      using BlockVector = Vectors::BlockVector<Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      //@}

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      ParabolicModule(
          const MPIEnsemble &mpi_ensemle,
          std::map<std::string, dealii::Timer> &computing_timer,
          const OfflineData<dim, Number> &offline_data,
          const HyperbolicSystem &hyperbolic_system,
          const ParabolicSystem &parabolic_system,
          const InitialValues<Description, dim, Number> &initial_values,
          const std::string &subsection = "/ParabolicModule");

      /**
       * Prepare time stepping. A call to @p prepare() allocates temporary
       * storage and is necessary before any of the following time-stepping
       * functions can be called.
       */
      void prepare();

      //@}
      /**
       * @name Functons for performing explicit time steps
       */
      //@{

      /**
       * (Re)initialize the parabolic state vector component of the state
       * vector.
       *
       * @note This routine does not modify the hyperbolic state vector or
       * the precomputed vector component.
       */
      void reinit_state_vector(StateVector &state_vector) const;

      /**
       * This function preprocesses a given state vector @p U in preparation
       * for a high order IMEX time step. This function exists because some
       * time stepping variants have to precompute quantities before we can
       * perform an IMEX step. In addition, this function is called whenever
       * we perform a mesh transfer or output operation.
       */
      void prepare_state_vector(StateVector &state_vector, Number t) const;

      /**
       * Given a reference to a previous state vector @p old_U at time
       * @p old_t and a time-step size @p tau perform an implicit backward
       * euler step (and store the result in @p new_U).
       *
       * The function takes an optional array of states @p stage_U together
       * with a an array of weights @p stage_weights to construct a modified
       * high-order right-hand side / flux.
       */
      template <int stages>
      void
      backward_euler_step(const StateVector &old_state_vector,
                          const Number old_t,
                          std::array<std::reference_wrapper<const StateVector>,
                                     stages> stage_state_vectors,
                          const std::array<Number, stages> stage_weights,
                          StateVector &new_state_vector,
                          Number tau) const;

      /**
       * Given a reference to a previous state vector @p old_U at time @p
       * old_t and a time-step size @p tau perform an implicit Crank-Nicolson
       * step (and store the result in @p new_U).
       *
       * This variant is used in the TimeIntegrator class for the Strang
       * split variants.
       */
      void crank_nicolson_step(const StateVector &old_state_vector,
                               const Number old_t,
                               StateVector &new_state_vector,
                               Number tau) const;

      /**
       * Sets the invariant domain violation strategy.
       */
      void set_id_violation_strategy(const IDViolationStrategy &strategy) const
      {
        id_violation_strategy_ = strategy;
      }

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Print a status line with solver statistics. This function is used
       * for constructing the status message displayed periodically in the
       * TimeLoop.
       */
      void print_solver_statistics(std::ostream &output) const;

      /**
       * The number of restarts signalled by the step() function.
       */
      ACCESSOR_READ_ONLY(n_restarts)

      /**
       * The number of corrections performed by the step() function. This
       * function exists to mirror the ParabolicModule interface and will
       * always return 0.
       */
      ACCESSOR_READ_ONLY(n_corrections)

      /**
       * The number of ID violation warnings encounterd in the step()
       * function.
       */
      ACCESSOR_READ_ONLY(n_warnings)

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
       * @name Low-level implementation
       */
      //@{

      /**
       * Set up affine constraints. Internally used in prepare().
       */
      void create_constraints();

      /**
       * Compute the potential phi (the last component of the state_vector)
       * for a given density (the first component of the state_vector).
       */
      void compute_potential(const Number t, StateVector &state_vector) const;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p old_t and a time-step size @p tau perform a backward
       * Euler time step (and store the result in @p new_state_vector).
       *
       * If the boolean @crank_nicolson_extrapolation is set to true, then
       * we perform a final extrapolation on the primitive state for time
       * t + 2 * tau.
       */
      void step(const StateVector &old_state_vector,
                const Number old_t,
                StateVector &new_state_vector,
                Number tau,
                const bool crank_nicolson_extrapolation) const;

      //@}
      /**
       * @name Internal data
       */
      //@{

      // FIXME: refactor
      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      const MPIEnsemble &mpi_ensemble_;
      std::map<std::string, dealii::Timer> &computing_timer_;

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
      dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;
      dealii::ObserverPointer<const ryujin::OfflineData<dim, Number>>
          offline_data_;
      dealii::ObserverPointer<
          const ryujin::InitialValues<Description, dim, Number>>
          initial_values_;

      ElectrostaticConfigurationLibrary::
          electrostatic_configuration_list_type<dim, Number>
              electrostatic_configuration_list_;

      std::shared_ptr<ElectrostaticConfigurationLibrary::
                          ElectrostaticConfiguration<dim, Number>>
          selected_electrostatic_configuration_;

      mutable IDViolationStrategy id_violation_strategy_;

      mutable unsigned int cycle_;
      mutable double n_iterations_gauss_;
      mutable double n_iterations_step_;

      mutable unsigned int n_restarts_;
      mutable unsigned int n_corrections_;
      mutable unsigned int n_warnings_;

      dealii::MatrixFree<dim, Number> matrix_free_;
      dealii::AffineConstraints<Number> affine_constraints_potential_;

      LaplaceOperator<dim, Number, Number> laplace_operator_;
      dealii::DiagonalMatrix<ScalarVector> diagonal_preconditioner_;
      MGSmoother<dim, Number> multigrid_preconditioner_;

      mutable bool potential_initialized_;
      mutable ScalarVector potential_rhs_;
      mutable ScalarVector density_;
      mutable ScalarVector background_density_;
      mutable BlockVector magnetic_field_;
      //@}
    };
  } // namespace EulerPoisson
} /* namespace ryujin */
