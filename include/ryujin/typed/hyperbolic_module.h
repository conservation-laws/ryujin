//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/mpi_ensemble.h>
#include <ryujin/base/observer_pointer.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/offline_data.h>
#include <ryujin/linear_algebra/sparse_matrix.h>
#include <ryujin/linear_algebra/state_vector.h>
#include <ryujin/typed/initial_values.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

#include <limits>
#include <ostream>
#include <utility>

namespace ryujin
{
  /**
   * Explicit forward Euler time-stepping for hyperbolic systems with
   * convex limiting.
   *
   * This module is described in detail in @cite ryujin-2021-1, Alg. 1.
   *
   * @ingroup Typed
   */
  template <typename HyperbolicDescription, int dim, typename Number = double>
  class HyperbolicModule final : public Interface::HyperbolicModule<dim, Number>
  {
  public:
    /**
     * @name Typedefs and constexpr constants
     */
    //@{

    using StageVectors =
        typename Interface::HyperbolicModule<dim, Number>::StageVectors;

    using StageWeights =
        typename Interface::HyperbolicModule<dim, Number>::StageWeights;

    using HyperbolicSystem = typename HyperbolicDescription::HyperbolicSystem;

    using View = typename HyperbolicSystem::template View<dim, Number>;

    using Indicator =
        typename HyperbolicDescription::template Indicator<Number>;

    using Limiter = typename HyperbolicDescription::template Limiter<Number>;

    using WaveSpeedEstimator =
        typename HyperbolicDescription::template WaveSpeedEstimator<Number>;

    static constexpr auto problem_dimension = View::problem_dimension;

    using state_type = typename View::state_type;

    static constexpr auto n_precomputed_values = View::n_precomputed_values;

    using precomputed_type = typename View::precomputed_type;

    static constexpr auto n_initial_precomputed_values =
        View::n_initial_precomputed_values;

    using initial_precomputed_type = typename View::initial_precomputed_type;

    //@}
    /**
     * @name Constructor and setup
     */
    //@{

    /**
     * Constructor
     */
    HyperbolicModule(
        const MPIEnsemble &mpi_ensemble,
        const Interface::OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const InitialValues<HyperbolicDescription, dim, Number> &initial_values,
        const std::string &subsection = "/HyperbolicModule");

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
     * (Re)initialize the hyperbolic state vector component and the
     * precomputed vector component of the state vector.
     *
     * @note This routine does not modify the parabolic state vector component.
     */
    void reinit_state_vector(StateVector<Number> &state_vector) const override;

    /**
     * This function preprocesses a given state vector @p U in preparation
     * for an explicit euler step performed by the step() function. The
     * function performs the following tasks:
     *
     *  - We update the @p U component of the state vector by enforcing
     *    boundary conditions for the supplied time time @p t. We then
     *    updates ghost ranges on @p U so that the state vector is
     *    consistent across MPI ranks.
     *
     *  - The function then runs the precomputation loop that populates the
     *    "precomputed values" component of the state vector and
     *    distributes the result over all MPI ranks by updating ghost
     *    ranges of the precomputed values vector.
     */
    void prepare_state_vector(StateVector<Number> &state_vector,
                              Number t) const override;

    void
    fill_precomputed_values(StateVector<Number> &state_vector) const override;

    /**
     * Given a reference to a previous state vector @p old_state_vector
     * perform an explicit euler step (and store the result in @p
     * new_state_vector). The
     * function returns the chosen time step size tau with which the update
     * was performed.
     *
     * The time step is performed with either tau_max (if @p tau is set to
     * 0), or tau (if @p tau is nonzero). Here, tau_max is the minimum of
     * the specified parameter @p tau_max and the computed maximal time
     * step size according to the CFL condition. @p tau is the last
     * parameter of the function.
     *
     * The function takes an optional list of states @p
     * stage_state_vectors together with a list of weights @p
     * stage_weights to construct a
     * modified high-order flux. The standard high-order flux reads
     * (cf. @cite ryujin-2021-1, Eq. 12):
     * \f{align}
     *   \newcommand{\bF}{{\boldsymbol F}}
     *   \newcommand{\bU}{{\boldsymbol U}}
     *   \newcommand\bUni{\bU^n_i}
     *   \newcommand\bUnj{\bU^n_j}
     *   \newcommand{\polf}{{\mathbb f}}
     *   \newcommand\Ii{\mathcal{I}(i)}
     *   \newcommand{\bc}{{\boldsymbol c}}
     *   \sum_{j\in\Ii} \frac{m_{ij}}{m_{j}}
     *   \;
     *   \frac{m_{j}}{\tau_n}\big(
     *   \tilde\bU_j^{H,n+1} - \bU_j^{n}\big)
     *   \;=\;
     *   \bF^n_i + \sum_{j\in\Ii}d_{ij}^{H,n}\big(\bUnj-\bUni\big),
     *   \qquad\text{with}\quad
     *   \bF^n_i\;:=\;
     *   \sum_{j\in\Ii}\Big(-(\polf(\bUni)+\polf(\bUnj)) \cdot\bc_{ij}\Big).
     * \f}
     * Instead, the function replaces \f$\bF^n_i\f$ in the right-hand side
     * by the modified high-order flux:
     * \f{align}
     *   \newcommand{\bF}{{\boldsymbol F}}
     *   \newcommand{\bU}{{\boldsymbol U}}
     *   \newcommand\bUnis{\bU^{s,n}_i}
     *   \newcommand\bUnjs{\bU^{s,n}_j}
     *   \newcommand{\polf}{{\mathbb f}}
     *   \newcommand\Ii{\mathcal{I}(i)}
     *   \newcommand{\bc}{{\boldsymbol c}}
     *   \tilde{\bF}^n_i\;:=\;
     *   \big(1-\sum_{s=1}^{\text{stages}}\omega_s\big)\bF^n_i
     *   \;+\;
     *   \sum_{s=1}^{\text{stages}}\omega_s \bF^{s,n}_i
     *   \qquad\text{with}\quad
     *   \bF^{s,n}_i\;:=\;
     *   \sum_{j\in\Ii}\Big(-(\polf(\bUnis)+\polf(\bUnjs)) \cdot\bc_{ij}\Big).
     * \f}
     * where \f$\omega_s\f$ denotes the weights for the given stages
     * \f$\bU^{s,n}\f$. The high-order viscosity term
     * \f$d_{ij}^{H,n}(\bU^n_j-\bU^n_i)\f$ is always evaluated with the
     * old state. Source terms, if present, are combined with the same
     * weights.
     *
     * @note The routine only performs an explicit update step on the
     * locally owned dof index range. It neither updates the precomputed
     * block of the state vector, sets boundary conditions (prior) to the
     * update step, nor automatically updates the ghost range of the
     * vector. It is thus necessary to call
     * HyperbolicModule::prepare_state_vector() on @p old_state_vector
     * prior to calling the step function.
     */
    Number
    step(const StateVector<Number> &old_state_vector,
         StageVectors stage_state_vectors,
         StageWeights stage_weights,
         StateVector<Number> &new_state_vector,
         Number tau = Number(0.),
         Number tau_max = std::numeric_limits<Number>::max()) const override;

    //@}
    /**
     * @name Information and statistics
     */
    //@{

    /**
     * Print a status line with solver statistics. The explicit update
     * has no solver statistics to report.
     */
    void print_solver_statistics(std::ostream & /*output*/) const override
    {
      // do nothing
    }

    /**
     * Return a reference to the OfflineData object
     */
    ACCESSOR_READ_ONLY(offline_data)

    /**
     * Return a reference to the HyperbolicSystem object
     */
    ACCESSOR_READ_ONLY(hyperbolic_system)

    /**
     * Return a reference to the precomputed initial data vector
     */
    ACCESSOR_READ_ONLY(initial_precomputed)

    /**
     * Return a reference to alpha vector storing indicator values. Note
     * that the values stored in alpha correspond to the last step executed
     * by this class.
     */
    ACCESSOR_READ_ONLY(alpha)

  private:
    //@}
    /**
     * @name Internal methods
     */
    //@{

    /**
     * The implementation of the step() function for a fixed number of
     * stages @p stages. The step() function dispatches to it with the
     * number of supplied stage state vectors.
     */
    template <int stages>
    Number step(const StateVector<Number> &old_state_vector,
                StageVectors stage_state_vectors,
                StageWeights stage_weights,
                StateVector<Number> &new_state_vector,
                Number tau,
                Number tau_max) const;

    /**
     * Update all boundary values of the hyperbolic state vector component
     * @p U for the supplied time @p t.
     *
     * The boundary map and the initial state functions are host-only data
     * structures. We therefore gather all boundary states of @p U into a
     * compact, mirrored buffer, apply boundary conditions on the host
     * memory space, and scatter the updated states back. This avoids
     * transferring the entire state vector to the host memory space and
     * back on every time step.
     *
     * @note The function does not update ghost values.
     */
    template <typename MemorySpace>
    void apply_boundary_conditions(MultiComponentVector<Number> &U,
                                   const Number t) const;

    //@}
    /**
     * @name Run time options
     */
    //@{
    Indicator indicator_;

    Limiter limiter_;

    WaveSpeedEstimator wave_speed_estimator_;

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPIEnsemble &mpi_ensemble_;

    dealii::ObserverPointer<const Interface::OfflineData<dim, Number>>
        offline_data_;
    dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::ObserverPointer<
        const InitialValues<HyperbolicDescription, dim, Number>>
        initial_values_;

    MultiComponentVector<Number> initial_precomputed_;

    mutable ScalarVector<Number> alpha_;

    static constexpr auto n_bounds =
        decltype(std::declval<const Limiter &>()
                     .template view<dim, Number>())::n_bounds;
    mutable MultiComponentVector<Number> bounds_;

    mutable MultiComponentVector<Number> r_;

    /* A buffer holding all boundary states of the hyperbolic state vector. */
    mutable Mirrored<Number *> boundary_states_{
        "hyperbolic_module_boundary_states"};

    mutable SparseMatrix<Number> dij_matrix_;
    mutable SparseMatrix<Number> lij_matrix_;
    mutable SparseMatrix<Number> lij_matrix_next_;
    mutable SparseMatrix<Number, problem_dimension> pij_matrix_;

    //@}
  };

} // namespace ryujin
