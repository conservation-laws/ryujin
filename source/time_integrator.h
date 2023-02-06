//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2022 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <hyperbolic_system.h>

#include "convenience_macros.h"
#include "hyperbolic_module.h"
#include "offline_data.h"
#include "patterns_conversion.h"

namespace ryujin
{
  /**
   * Controls the chosen invariant domain / CFL recovery strategy.
   */
  enum class CFLRecoveryStrategy {
    /**
     * Step with the chosen "cfl max" value and do nothing in case an
     * invariant domain and or CFL condition violation is detected.
     */
    none,

    /**
     * Step with the chosen "cfl max" value and, in case an invariant
     * domain and or CFL condition violation is detected, the time step
     * is repeated with "cfl min". If this is unsuccessful as well, a
     * warning is emitted.
     */
    bang_bang_control,
  };


  /**
   * Controls the chosen time-stepping scheme.
   */
  enum class TimeSteppingScheme {
    /**
     * The strong stability preserving Runge Kutta method of order 3,
     * SSPRK(3,3;1/3), with the following butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   1            & 1            & 0 \\
     *   \tfrac{1}{2} & \tfrac{1}{4} & \tfrac{1}{4} & 0\\
     *   \hline
     *   1            & \tfrac{1}{6} & \tfrac{1}{6} & \tfrac{2}{3}
     * \end{array}
     * \f}
     */
    ssprk_33,

    /**
     * The explicit Runge-Kutta method RK(2,2;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{2} & \tfrac{1}{2} & 0 \\
     *   \hline
     *   1            & 0  & 1
     * \end{array}
     * \f}
     */
    erk_22,

    /**
     * The explicit Runge-Kutta method RK(3,3;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{3} & \tfrac{1}{3} & 0 \\
     *   \tfrac{2}{3} & 0            & \tfrac{2}{3} & 0 \\
     *   \hline
     *   1            & \tfrac{1}{4} & 0            & \tfrac{3}{4}
     * \end{array}
     * \f}
     */
    erk_33,

    /**
     * The explicit Runge-Kutta method RK(4,3;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{4} & \tfrac{1}{4} & 0 \\
     *   \tfrac{1}{2} & 0            & \tfrac{1}{2} & 0 \\
     *   \tfrac{3}{4} & 0            & \tfrac{1}{4} & \tfrac{1}{2}  & 0 \\
     *   \hline
     *   1            & 0            & \tfrac{2}{3} & -\tfrac{1}{3} &
     * \tfrac{2}{3} \end{array} \f}
     */
    erk_43,
  };
} // namespace ryujin

#ifndef DOXYGEN
DECLARE_ENUM(ryujin::CFLRecoveryStrategy,
             LIST({ryujin::CFLRecoveryStrategy::none, "none"},
                  {ryujin::CFLRecoveryStrategy::bang_bang_control,
                   "bang bang control"}));

DECLARE_ENUM(ryujin::TimeSteppingScheme,
             LIST({ryujin::TimeSteppingScheme::ssprk_33, "ssprk 33"},
                  {ryujin::TimeSteppingScheme::erk_22, "erk 22"},
                  {ryujin::TimeSteppingScheme::erk_33, "erk 33"},
                  {ryujin::TimeSteppingScheme::erk_43, "erk 43"}));
#endif

namespace ryujin
{
  /**
   * The TimeIntegrator class implements IMEX timestepping strategies based
   * on explicit and diagonally-implicit Runge Kutta schemes.
   *
   * @ingroup TimeLoop
   */
  template <typename Description, int dim, typename Number = double>
  class TimeIntegrator final : public dealii::ParameterAcceptor
  {
  public:
    /**
     * @copydoc HyperbolicSystem
     */
    using HyperbolicSystem = typename Description::HyperbolicSystem;

    /**
     * @copydoc HyperbolicSystem::problem_dimension
     */
    static constexpr unsigned int problem_dimension =
        HyperbolicSystem::template problem_dimension<dim>;

    /**
     * @copydoc HyperbolicSystem::n_precomputed_values
     */
    static constexpr unsigned int n_precomputed_values =
        HyperbolicSystem::template n_precomputed_values<dim>;

    /**
     * Typedef for a MultiComponentVector storing the state U.
     */
    using vector_type = MultiComponentVector<Number, problem_dimension>;

    /**
     * Typedef for a MultiComponentVector storing precomputed values.
     */
    using precomputed_type = MultiComponentVector<Number, n_precomputed_values>;

    /**
     * Constructor.
     */
    TimeIntegrator(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const OfflineData<dim, Number> &offline_data,
        const HyperbolicModule<Description, dim, Number> &hyperbolic_module,
        const std::string &subsection = "TimeIntegrator");

    /**
     * Prepare time integration. A call to @ref prepare() allocates
     * temporary storage and is necessary before any of the following
     * time-stepping functions can be called.
     */
    void prepare();

    /**
     * @name Functions for performing explicit time steps
     */
    //@{

    /**
     * Given a reference to a previous state vector U performs an explicit
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * @note This function switches between different Runge-Kutta methods
     * depending on chosen runtime parameters.
     *
     * @note Depending on chosen run time parameters different CFL
     * adaptation and recovery strategies for invariant domain violations
     * are used.
     */
    Number step(vector_type &U, Number t, unsigned int cycle);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * third-order strong-stability preserving Runge-Kutta SSPRK(3,3,1/3)
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     *
     * If the parameter @ref tau is set to a nonzero value then the
     * supplied value is used for time stepping instead of the computed
     * maximal time step size.
     */
    Number step_ssprk_33(vector_type &U, Number t, Number tau = Number(0.));

    /**
     * Given a reference to a previous state vector U performs an explicit
     * second-order Runge-Kutta SSPRK(2,2,1) time step (and store the
     * result in U). The function returns the chosen time step size tau.
     */
    Number step_erk_22(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * third-order Runge-Kutta SSPRK(3,3,1) time step (and store the
     * result in U). The function returns the chosen time step size tau.
     */
    Number step_erk_33(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * 4 stage third-order Runge-Kutta SSPRK(4,3,1) time step (and store
     * the result in U). The function returns the chosen time step size
     * tau.
     */
    Number step_erk_43(vector_type &U, Number t);

  private:
    //@}
    /**
     * @name Run time options
     */
    //@{

    Number cfl_min_;
    Number cfl_max_;

    CFLRecoveryStrategy cfl_recovery_strategy_;

    TimeSteppingScheme time_stepping_scheme_;

    //@}

    //@}
    /**
     * @name Internal data
     */
    //@{

    const MPI_Comm &mpi_communicator_;
    std::map<std::string, dealii::Timer> &computing_timer_;

    dealii::SmartPointer<const OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const HyperbolicModule<Description, dim, Number>>
        hyperbolic_module_;

    std::vector<vector_type> U_;
    std::vector<precomputed_type> precomputed_;

    //@}
  };

} /* namespace ryujin */
