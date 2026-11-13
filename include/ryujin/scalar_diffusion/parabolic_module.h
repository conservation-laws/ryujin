//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/discretization/offline_data.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/scalar_diffusion/diffusion_operator.h>
#include <ryujin/scalar_diffusion/parabolic_system.h>
#include <ryujin/typed/initial_values.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <limits>

namespace ryujin
{
  namespace ScalarDiffusion
  {
    /**
     * Implicit time stepping for the power-law diffusion model
     * \f$\partial_t u = \nabla\cdot(k(u)\nabla u)\f$, see ParabolicSystem.
     *
     * The spatial discretization uses the lumped mass matrix and the
     * stiffness matrix \f$\mathbb K(\textbf k)\f$ with the finite element
     * interpolant of the nodal diffusivity \f$\textbf k\f$. The parabolic
     * operator and its quasi-linearization around a state
     * \f$\textbf U^n\f$ thus read
     * \f{align}
     *   \textbf G(\textbf U) = -\mathbb K\big(k(\textbf U)\big)\textbf U,
     *   \qquad
     *   \textbf G^{\text{lin}}(\textbf U^n;\textbf U) =
     *   -\mathbb K\big(k(\textbf U^n)\big)\textbf U.
     * \f}
     * The linear system of a backward Euler step is solved with a
     * matrix-free, Jacobi preconditioned conjugate gradient method.
     *
     * Hanging node and periodicity constraints are supported. All
     * boundaries are equipped with the natural (no flux) boundary
     * condition.
     *
     * @note In contrast to @cite ErnGuermond2023 the parabolic update is
     * not limited.
     *
     * @ingroup ScalarDiffusionEquations
     */
    template <int dim, typename Number = double>
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

      using HyperbolicDescription = ScalarConservation::HyperbolicDescription;

      using HyperbolicSystem = ScalarConservation::HyperbolicSystem;

      using InitialValues =
          typename HyperbolicDescription::template InitialValues<dim, Number>;

      using View = ScalarConservation::HyperbolicSystemView<dim, Number>;

      using ParabolicSystem = ScalarDiffusion::ParabolicSystem;

      static constexpr auto problem_dimension = View::problem_dimension;

      //@}
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      ParabolicModule(const MPIEnsemble &mpi_ensemble,
                      const OfflineData<dim, dim, Number> &offline_data,
                      const HyperbolicSystem &hyperbolic_system,
                      const ParabolicSystem &parabolic_system,
                      const InitialValues &initial_values,
                      const std::string &subsection = "ParabolicModule");

      /**
       * Prepare time stepping: set up the matrix-free object and allocate
       * temporary vectors.
       */
      void prepare() override;

      //@}
      /**
       * @name Functions for performing implicit time steps
       */
      //@{

      /**
       * Nothing to do: the parabolic system has no components of its own.
       */
      void
      reinit_state_vector(StateVector<Number> & /*state_vector*/) const override
      {
      }

      /**
       * Nothing to do.
       */
      void prepare_state_vector(StateVector<Number> & /*state_vector*/,
                                Number /*t*/) const override
      {
      }

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p t and a time-step size @p tau perform an implicit
       * backward Euler step (and store the result in @p new_state_vector).
       *
       * The stage state vectors and weights enter as described in
       * Interface::ParabolicModule::backward_euler_step(). If no stage
       * state vectors are supplied the diffusivity is evaluated with
       * @p old_state_vector.
       */
      Number backward_euler_step(
          const StateVector<Number> &old_state_vector,
          StageVectors stage_state_vectors,
          StageWeights stage_weights_explicit,
          StageWeights stage_weights_implicit,
          StateVector<Number> &new_state_vector,
          Number t,
          Number tau,
          Number tau_max = std::numeric_limits<Number>::max()) const override;

      /**
       * Given a reference to a previous state vector @p old_state_vector
       * at time @p t and a time-step size @p tau perform an implicit
       * Crank-Nicolson step (and store the result in @p new_state_vector).
       *
       * For a nonlinear diffusivity the step is the linearly implicit
       * midpoint rule: a backward Euler step to the half time step with
       * the diffusivity evaluated at @p old_state_vector serves as a
       * predictor for the diffusivity at the midpoint.
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

      //@}

    private:
      /**
       * @name Run time options
       */
      //@{

      Number tolerance_;

      //@}
      /**
       * @name Internal data
       */
      //@{

      const MPIEnsemble &mpi_ensemble_;

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
      dealii::ObserverPointer<const ParabolicSystem> parabolic_system_;
      dealii::ObserverPointer<const ryujin::OfflineData<dim, dim, Number>>
          offline_data_;
      dealii::ObserverPointer<const InitialValues> initial_values_;

      mutable double n_iterations_;

      mutable dealii::MatrixFree<dim, Number> matrix_free_;

      mutable ScalarHostVector<Number> diffusivity_;
      mutable ScalarHostVector<Number> right_hand_side_;
      mutable ScalarHostVector<Number> solution_;
      mutable ScalarHostVector<Number> temp_;
      mutable ScalarHostVector<Number> temp_stiffness_;

      mutable DiffusionOperator<dim, Number> diffusion_operator_;

      //@}
      /**
       * @name Internal methods
       */
      //@{

      /**
       * Store the nodal values of the scalar state of @p state_vector in
       * @p dst. Constrained degrees of freedom are set to zero.
       */
      void extract(const StateVector<Number> &state_vector,
                   ScalarHostVector<Number> &dst) const;

      /**
       * Reinitialize the diffusion operator with the diffusivity of the
       * nodal state @p u.
       */
      void reinit_operator(const ScalarHostVector<Number> &u) const;

      /**
       * Solve \f$(\mathbb M^L + \sigma\mathbb K)\,\textbf u = \textbf r\f$
       * for the right hand side right_hand_side_ and the currently set
       * diffusivity. The result is stored in solution_, which also has to
       * hold the initial guess. For \f$\sigma=0\f$ the lumped mass matrix
       * is inverted directly.
       */
      void solve(const Number sigma) const;

      //@}
    };

  } // namespace ScalarDiffusion
} // namespace ryujin
