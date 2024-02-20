//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"
#include "initial_values.h"
#include "offline_data.h"
#include "simd.h"
#include "sparse_matrix_simd.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_matrix.templates.h>
#include <deal.II/lac/vector.h>

#include <functional>

namespace ryujin
{
  /**
   * An enum controlling the behavior on detection of an invariant domain
   * or CFL violation. Such a case might occur for either aggressive CFL
   * numbers > 1, and/or later stages in the Runge Kutta scheme when the
   * time step tau is prescribed.
   *
   * The invariant domain violation is detected in the limiter and
   * typically implies that the low-order update is already out of
   * bounds. We further do a quick sanity check whether the computed
   * step size tau_max and the prescribed step size tau are within an
   * acceptable tolerance of about 10%.
   *
   * @ingroup TimeLoop
   */
  enum class IDViolationStrategy {
    /**
     * Warn about an invariant domain violation but take no further
     * action.
     */
    warn,

    /**
     * Raise a Restart exception on domain violation. This exception can be
     * caught in TimeIntegrator and various different actions (adapt CFL
     * and retry) can be taken depending on chosen strategy.
     */
    raise_exception,
  };


  /**
   * A class signalling a restart, thrown in HyperbolicModule::single_step and
   * caught at various places.
   *
   * @ingroup TimeLoop
   */
  class Restart final
  {
  };


  /**
   * Explicit forward Euler time-stepping for hyperbolic systems with
   * convex limiting.
   *
   * This module is described in detail in @cite ryujin-2021-1, Alg. 1.
   *
   * @ingroup HyperbolicModule
   */
  template <typename Description, int dim, typename Number = double>
  class HyperbolicModule final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc HyperbolicSystem::View
     */
    using HyperbolicSystemView =
        typename HyperbolicSystem::template View<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystemView::problem_dimension;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = typename HyperbolicSystemView::state_type;

    /**
     * @copydoc HyperbolicSystem::flux_type
     */
    using flux_type = typename HyperbolicSystemView::flux_type;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * @copydoc HyperbolicSystem::View::vector_type
     */
    using vector_type = typename HyperbolicSystemView::vector_type;

    /**
     * @copydoc HyperbolicSystem::n_precomputed_values
     */
    static constexpr unsigned int n_precomputed_values =
        HyperbolicSystemView::n_precomputed_values;

    /**
     * @copydoc HyperbolicSystemView::n_precomputation_cycles
     */
    static constexpr unsigned int n_precomputation_cycles =
        HyperbolicSystemView::n_precomputation_cycles;

    /**
     * Typedef for a MultiComponentVector storing precomputed values.
     */
    using precomputed_vector_type =
        typename HyperbolicSystemView::precomputed_vector_type;

    /**
     * @copydoc HyperbolicSystem::n_precomputed_initial_values
     */
    static constexpr unsigned int n_precomputed_initial_values =
        HyperbolicSystemView::n_precomputed_initial_values;

    /**
     * Typedef for a MultiComponentVector storing precomputed initial_values.
     */
    using precomputed_initial_vector_type =
        typename HyperbolicSystemView::precomputed_initial_vector_type;


    /**
     * Constructor.
     */
    HyperbolicModule(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicSystem &hyperbolic_system,
        const InitialValues<Description, dim, Number> &initial_values,
        const std::string &subsection = "/HyperbolicModule");

    /**
     * Prepare time stepping. A call to @p prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector @p old_U perform an
     * explicit euler step (and store the result in @p new_U). The
     * function returns the computed maximal time step size tau_max
     * according to the CFL condition.
     *
     * The time step is performed with either tau_max (if @p tau is set
     * to 0), or tau (if @p tau is nonzero). Here, tau_max is the
     * computed maximal time step size and @p tau is the last parameter
     * of the function.
     *
     * The function takes an optional array of states @p stage_U together
     * with a an array of weights @p stage_weights to construct a
     * modified high-order flux. The standard high-order flux reads
     * (cf @cite ryujin-2021-1, Eq. 12):
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
     * Instead, the function assembles the modified high-order flux:
     * \f{align}
     *   \newcommand{\bF}{{\boldsymbol F}}
     *   \newcommand{\bU}{{\boldsymbol U}}
     *   \newcommand\bUnis{\bU^{s,n}_i}
     *   \newcommand\bUnjs{\bU^{s,n}_j}
     *   \newcommand{\polf}{{\mathbb f}}
     *   \newcommand\Ii{\mathcal{I}(i)}
     *   \newcommand{\bc}{{\boldsymbol c}}
     *   \tilde{\bF}^n_i\;:=\;
     *   \big(1-\sum_{s=\{1:\text{stages}\}}\omega_s\big)\bF^n_i
     *   \;+\;
     *   \sum_{s=\{1:stages\}}\omega_s \bF^{s,n}_i
     *   \qquad\text{with}\quad
     *   \bF^{s,n}_i\;:=\;
     *   \sum_{j\in\Ii}\Big(-(\polf(\bUnis)+\polf(\bUnjs)) \cdot\bc_{ij}\Big).
     * \f}
     * where \f$\omega_s\f$ denotes the weigths for the given stages
     * \f$\bU^{s,n}\f$.
     *
     * @note The routine does not automatically update ghost vectors of the
     * distributed vector @p new_U. It is best to simply call
     * HyperbolicModule::apply_boundary_conditions() on the appropriate vector
     * immediately after performing a time step.
     */
    template <int stages>
    Number
    step(const vector_type &old_U,
         std::array<std::reference_wrapper<const vector_type>, stages> stage_U,
         std::array<std::reference_wrapper<const precomputed_vector_type>,
                    stages> stage_precomputed,
         const std::array<Number, stages> stage_weights,
         vector_type &new_U,
         precomputed_vector_type &new_precomputed,
         Number tau = Number(0.)) const;

    /**
     * This function postprocesses a given state @p U to conform with all
     * prescribed boundary conditions at time @p t. This implies that on
     * slip (and no-slip) boundaries the normal momentum is set to zero; on
     * Dirichlet boundaries the appropriate state at time @p t is
     * substituted; and on "flexible" boundaries depending on the fact
     * whether we have supersonic or subsonic inflow/outflow the
     * appropriate Riemann invariant is prescribed. See @cite ryujin-2021-3
     * for details.
     *
     * @note The routine does update ghost vectors of the distributed
     * vector @p U
     */
    void apply_boundary_conditions(vector_type &U, Number t) const;

    /**
     * Sets the relative CFL number used for computing an appropriate
     * time-step size to the given value. The CFL number must be a positive
     * value. If chosen to be within the interval \f$(0,1)\f$ then the
     * low-order update and limiting stages guarantee invariant domain
     * preservation.
     */
    void cfl(Number new_cfl) const
    {
      Assert(cfl_ > Number(0.), dealii::ExcInternalError());
      cfl_ = new_cfl;
    }

    /**
     * Returns the relative CFL number used for computing an appropriate
     * time-step size.
     */
    ACCESSOR_READ_ONLY(cfl)

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
    ACCESSOR_READ_ONLY(precomputed_initial)

    /**
     * Return a reference to alpha vector storing indicator values. Note
     * that the values stored in alpha correspond to the last step executed
     * by this class. This value can be recomputed for a given state vector
     * by setting precompute_only_ to true and calling the step() function.
     */
    ACCESSOR_READ_ONLY(alpha)

    /**
     * The number of restarts issued by the step() function.
     */
    ACCESSOR_READ_ONLY(n_restarts)

    /**
     * The number of ID violation warnings encounterd in the step()
     * function.
     */
    ACCESSOR_READ_ONLY(n_warnings)

    // FIXME: refactor to function
    mutable bool precompute_only_;

    // FIXME: refactor to function
    mutable IDViolationStrategy id_violation_strategy_;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{
    typename Description::template Indicator<dim, Number>::Parameters
        indicator_parameters_;

    typename Description::template Limiter<dim, Number>::Parameters
        limiter_parameters_;

    bool cfl_with_boundary_dofs_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicSystem> hyperbolic_system_;
    dealii::SmartPointer<const InitialValues<Description, dim, Number>>
        initial_values_;

    mutable Number cfl_;

    mutable unsigned int n_restarts_;

    mutable unsigned int n_warnings_;

    precomputed_initial_vector_type precomputed_initial_;

    mutable scalar_type alpha_;

    static constexpr auto n_bounds =
        Description::template Limiter<dim, Number>::n_bounds;
    mutable MultiComponentVector<Number, n_bounds> bounds_;

    mutable vector_type r_;

    mutable SparseMatrixSIMD<Number> dij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_next_;
    mutable SparseMatrixSIMD<Number, problem_dimension> pij_matrix_;

    //@}
  };

} /* namespace ryujin */
