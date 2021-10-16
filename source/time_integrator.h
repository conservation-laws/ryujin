//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "convenience_macros.h"

#include "dissipation_module.h"
#include "euler_module.h"
#include "offline_data.h"

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
     * The explicit Runge-Kutta method RK(3,3;1) with the butcher tableau
     * \f{align*}
     * \begin{array}{c|ccc}
     *   0            & 0 \\
     *   \tfrac{1}{3} & \tfrac{1}{3} & 0 \\
     *   \tfrac{2}{3} & 0            & \tfrac{2}{3} & 0\\
     *   \hline
     *   1            & \tfrac{1}{4} & 0            & \tfrac{3}{4}
     * \end{array}
     * \f}
     */
    erk_33
  };
}

#ifndef doxygen
/*
 * Boilerplate for automatic translation between runtime parameter string
 * and enum classes:
 */
DEAL_II_NAMESPACE_OPEN
template<>
struct Patterns::Tools::Convert<ryujin::CFLRecoveryStrategy> {
  using T = typename ryujin::CFLRecoveryStrategy;

  static std::unique_ptr<Patterns::PatternBase> to_pattern()
  {
    return std::make_unique<Patterns::Selection>("none|bang bang control");
  }

  static std::string to_string(const T &t, const Patterns::PatternBase &)
  {
    switch (t) {
    case ryujin::CFLRecoveryStrategy::none:
      return "none";
    case ryujin::CFLRecoveryStrategy::bang_bang_control:
      return "bang bang control";
    }
  }

  static T
  to_value(const std::string &s,
           const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
  {
    AssertThrow(pattern.match(s), ExcNoMatch(s, pattern.description()))

    if (s == "none")
      return ryujin::CFLRecoveryStrategy::none;
    else if (s == "bang bang control")
      return ryujin::CFLRecoveryStrategy::bang_bang_control;
    else {
      AssertThrow(false, ExcInternalError());
    }
  }
};

template<>
struct Patterns::Tools::Convert<ryujin::TimeSteppingScheme> {
  using T = typename ryujin::TimeSteppingScheme;

  static std::unique_ptr<Patterns::PatternBase> to_pattern()
  {
    return std::make_unique<Patterns::Selection>("ssprk 33|erk 33");
  }

  static std::string to_string(const T &t, const Patterns::PatternBase &)
  {
    switch (t) {
    case ryujin::TimeSteppingScheme::ssprk_33:
      return "ssprk 33";
    case ryujin::TimeSteppingScheme::erk_33:
      return "erk 33";
    }
  }

  static T
  to_value(const std::string &s,
           const Patterns::PatternBase &pattern = *Convert<T>::to_pattern())
  {
    AssertThrow(pattern.match(s), ExcNoMatch(s, pattern.description()));
    if (s == "ssprk 33")
      return ryujin::TimeSteppingScheme::ssprk_33;
    else if (s == "erk 33")
      return ryujin::TimeSteppingScheme::erk_33;
    else {
      AssertThrow(false, ExcInternalError());
    }
  }
};
DEAL_II_NAMESPACE_CLOSE
#endif


namespace ryujin
{
  /**
   * TODO documentation
   */
  template <int dim, typename Number = double>
  class TimeIntegrator final : public dealii::ParameterAcceptor
  {
  public:

    std::string cfl_recovery_strategy_string_;

    /**
     * @copydoc OfflineData::vector_type
     */
    using vector_type = typename OfflineData<dim, Number>::vector_type;

    /**
     * Constructor.
     */
    TimeIntegrator(
        const MPI_Comm &mpi_communicator,
        std::map<std::string, dealii::Timer> &computing_timer,
        const ryujin::OfflineData<dim, Number> &offline_data,
        const ryujin::EulerModule<dim, Number> &euler_module,
        const ryujin::DissipationModule<dim, Number> &dissipation_module,
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
    Number step(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * third-order strong-stability preserving Runge-Kutta SSPRK(3,3,1/3)
     * time step (and store the result in U). The function returns the
     * chosen time step size tau.
     */
    Number step_ssprk_33(vector_type &U, Number t);

    /**
     * Given a reference to a previous state vector U performs an explicit
     * third-order Runge-Kutta SSPRK(3,3,1) time step (and store the
     * result in U). The function returns the chosen time step size tau.
     */
    Number step_erk_33(vector_type &U, Number t);

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

    dealii::SmartPointer<const ryujin::OfflineData<dim, Number>> offline_data_;
    dealii::SmartPointer<const ryujin::EulerModule<dim, Number>> euler_module_;

    dealii::SmartPointer<const ryujin::DissipationModule<dim, Number>>
        dissipation_module_;

    SparseMatrixSIMD<Number> dummy_; /* kept uninitialized */
    std::vector<SparseMatrixSIMD<Number>> temp_dij_;
    std::vector<vector_type> temp_U_;

    //@}
  };

} /* namespace ryujin */
