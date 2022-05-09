//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <hyperbolic_system.h>
#include <indicator.h>
#include <limiter.h>

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
     * Raise a ryujin::Restart exception on domain violation. This
     * exception can be caught in TimeIntegrator and various different
     * actions (adapt CFL and retry) can be taken depending on chosen
     * strategy.
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
   * This module is described in detail in @cite ryujin-2022-1, Alg. 1.
   *
   * @ingroup HyperbolicModule
   */
  template <int dim, typename Number = double>
  class HyperbolicModule final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystem::problem_dimension<dim>;

    /**
     * @copydoc HyperbolicSystem::state_type
     */
    using state_type = HyperbolicSystem::state_type<dim, Number>;

    /**
     * @copydoc HyperbolicSystem::flux_type
     */
    using flux_type = HyperbolicSystem::flux_type<dim, Number>;

    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * Constructor.
     */
    HyperbolicModule(const MPI_Comm &mpi_communicator,
                     std::map<std::string, dealii::Timer> &computing_timer,
                     const ryujin::OfflineData<dim, Number> &offline_data,
                     const ryujin::HyperbolicSystem &hyperbolic_system,
                     const ryujin::InitialValues<dim, Number> &initial_values,
                     const std::string &subsection = "HyperbolicModule");

    /**
     * Prepare time stepping. A call to @ref prepare() allocates temporary
     * storage and is necessary before any of the following time-stepping
     * functions can be called.
     */
    void prepare();

    /**
     * @name Functons for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector @ref old_U perform an
     * explicit euler step (and store the result in @ref new_U). The
     * function returns the computed maximal time step size tau_max
     * according to the CFL condition.
     *
     * The time step is performed with either tau_max (if @ref tau is set
     * to 0), or tau (if @ref tau is nonzero). Here, tau_max is the
     * computed maximal time step size and @ref tau is the last parameter
     * of the function.
     *
     * The function takes an optional array of states @ref stage_U together
     * with a an array of weights @ref stage_weights to construct a
     * modified high-order flux. The standard high-order flux reads
     * (cf @cite ryujin-2022-1, Eq. 12):
     * \f{align}
     *   \newcommand{\bR}{{\boldsymbol R}}
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
     *   \bR^n_i,
     *   \qquad\text{with}\quad
     *   \bR^n_i\;:=\;
     *   \sum_{j\in\Ii}\Big(-\polf(\bUnj) \cdot\bc_{ij}
     *   +d_{ij}^{H,n}\big(\bUnj-\bUni\big)\Big).
     * \f}
     * Instead, the function assembles the modified high-order flux:
     * \f{align}
     *   \newcommand{\bR}{{\boldsymbol R}}
     *   \newcommand{\bU}{{\boldsymbol U}}
     *   \newcommand\bUnis{\bU^{s,n}_i}
     *   \newcommand\bUnjs{\bU^{s,n}_j}
     *   \newcommand{\polf}{{\mathbb f}}
     *   \newcommand\Ii{\mathcal{I}(i)}
     *   \newcommand{\bc}{{\boldsymbol c}}
     *   \tilde{\bR}^n_i\;:=\;
     *   \big(1-\sum_{s=\{1:\text{stages}\}}\omega_s\big)\bR^n_i
     *   \;+\;
     *   \sum_{s=\{1:stages\}}\omega_s
     *   \sum_{j\in\Ii}\Big(-\polf(\bUnis)-\polf(\bUnjs) \cdot\bc_{ij}
     *   +d_{ij}^{H,s,n}\big(\bUnjs-\bUnis\big)\Big),
     * \f}
     * where \f$\omega_s\f$ denotes the weigths for the given stages.
     *
     * @note The routine does not automatically update ghost vectors of the
     * distributed vector @ref new_U. It is best to simply call
     * HyperbolicModule::apply_boundary_conditions() on the appropriate vector
     * immediately after performing a time step.
     */
    template <int stages>
    Number
    step(const vector_type &old_U,
         std::array<std::reference_wrapper<const vector_type>, stages> stage_U,
         const std::array<Number, stages> stage_weights,
         vector_type &new_U,
         Number tau = Number(0.)) const;

    /**
     * This function postprocesses a given state @ref U to conform with all
     * prescribed boundary conditions at time @ref t. This implies that on
     * slip (and no-slip) boundaries the normal momentum is set to zero; on
     * Dirichlet boundaries the appropriate state at time @ref t is
     * substituted; and on "flexible" boundaries depending on the fact
     * whether we have supersonic or subsonic inflow/outflow the
     * appropriate Riemann invariant is prescribed. See @cite ryujin-2022-3
     * for details.
     *
     * @note The routine does update ghost vectors of the distributed
     * vector @ref U
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

    mutable IDViolationStrategy id_violation_strategy_;

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    unsigned int limiter_iter_;
    Number limiter_newton_tolerance_;
    unsigned int limiter_newton_max_iter_;

    bool cfl_with_boundary_dofs_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::HyperbolicSystem> hyperbolic_system_;
    ACCESSOR_READ_ONLY(hyperbolic_system)
    dealii::SmartPointer<const ryujin::InitialValues<dim, Number>>
        initial_values_;

    mutable Number cfl_;
    ACCESSOR_READ_ONLY(cfl)

    mutable unsigned int n_restarts_;
    ACCESSOR_READ_ONLY(n_restarts)

    mutable unsigned int n_warnings_;
    ACCESSOR_READ_ONLY(n_warnings)

    static constexpr auto n_prec = HyperbolicSystem::n_precomputed_values<dim>;
    mutable MultiComponentVector<Number, n_prec> hyperbolic_system_prec_values_;
    ACCESSOR_READ_ONLY(hyperbolic_system_prec_values)

    static constexpr auto n_ind = Indicator<dim, Number>::n_precomputed_values;
    mutable MultiComponentVector<Number, n_ind> indicator_prec_values_;

    mutable scalar_type alpha_;
    ACCESSOR_READ_ONLY(alpha)

    static constexpr auto n_lim = Limiter<dim, Number>::n_precomputed_values;
    mutable MultiComponentVector<Number, n_lim> limiter_prec_values_;

    static constexpr auto n_bounds = Limiter<dim, Number>::n_bounds;
    mutable MultiComponentVector<Number, n_bounds> bounds_;

    mutable vector_type r_;

    mutable SparseMatrixSIMD<Number> dij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_;
    mutable SparseMatrixSIMD<Number> lij_matrix_next_;
    mutable SparseMatrixSIMD<Number, problem_dimension> pij_matrix_;

    //@}
  };

  } /* namespace ryujin */
