//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/discretization/butcher_tableau.h>
#include <ryujin/discretization/time_integrator.h>
#include <ryujin/discretization/time_stepping.h>

#include <span>
#include <utility>

namespace ryujin
{
  using namespace dealii;

  template <int dim, typename Number>
  TimeIntegrator<dim, Number>::TimeIntegrator(
      const MPIEnsemble &mpi_ensemble,
      const Interface::HyperbolicSystem &hyperbolic_system,
      const Interface::ParabolicSystem &parabolic_system,
      const Interface::HyperbolicModule<dim, Number> &hyperbolic_module,
      const Interface::ParabolicModule<dim, Number> &parabolic_module,
      const std::string &subsection /*= "TimeIntegrator"*/)
      : Interface::TimeIntegrator<dim, Number>(subsection)
      , mpi_ensemble_(mpi_ensemble)
      , hyperbolic_system_(&hyperbolic_system)
      , parabolic_system_(&parabolic_system)
      , hyperbolic_module_(&hyperbolic_module)
      , parabolic_module_(&parabolic_module)
  {
    if (parabolic_system_->is_identity())
      time_stepping_scheme_ = TimeSteppingScheme::erk_33;
    else
      time_stepping_scheme_ = TimeSteppingScheme::strang_erk_33_cn;
    this->add_parameter(
        "time stepping scheme",
        time_stepping_scheme_,
        "Time stepping scheme: ssprk 22, ssprk 33, erk 11, erk 22, "
        "erk 33, erk 43, erk "
        "54, strang ssprk 33 cn, strang erk 33 cn, strang erk 43 cn, "
        "strang erk 33 dirk 22, strang erk 33 dirk 23, imex 11, imex 22, "
        "imex 33, gsa imex 22, gsa imex 43");

    cfl_min_ = Number(0.45);
    this->add_parameter(
        "cfl min",
        cfl_min_,
        "Minimal admissible relative CFL constant. How this parameter is used "
        "depends on the chosen CFL recovery strategy");

    cfl_max_ = Number(0.90);
    this->add_parameter(
        "cfl max",
        cfl_max_,
        "Maximal admissible relative CFL constant. How this parameter is used "
        "depends on the chosen CFL recovery strategy");

    cfl_recovery_strategy_ = CFLRecoveryStrategy::cruise_control;
    this->add_parameter(
        "cfl recovery strategy",
        cfl_recovery_strategy_,
        "CFL/invariant domain violation recovery strategy: none, "
        "bang bang control, cruise control");

    acceptable_tau_max_ratio_ = Number(2.0);
    this->add_parameter(
        "acceptable tau_max ratio",
        acceptable_tau_max_ratio_,
        "Maximal acceptable discrepancy between computed tau_max of "
        "a (sub)step and enforced time-step size tau. If the ratio "
        "is violated then a restart will be singnalled.");

    tau_max_ = std::numeric_limits<Number>::max();
    this->add_parameter("tau_max", tau_max_, "Largest time step size allowed.");
  }


  template <int dim, typename Number>
  void TimeIntegrator<dim, Number>::prepare()
  {
    RYUJIN_DEBUG_ANNOUNCE();

    this->time_stepping_scheme_name_ =
        Patterns::Tools::Convert<TimeSteppingScheme>::to_string(
            time_stepping_scheme_);

    /*
     * Resize temporary storage to appropriate sizes and determine whether
     * the scheme performs implicit steps with the parabolic module:
     */

    /*
     * The Runge-Kutta methods of the selected scheme: a single method, or
     * for a Strang splitting an explicit method for the hyperbolic and a
     * diagonally implicit method for the parabolic subproblem. The strong
     * stability preserving schemes are convex combinations of forward
     * Euler steps and have no method.
     */

    using enum TimeSteppingScheme;
    using M = RungeKuttaMethod;

    bool ssprk = false;
    bool strang = false;
    RungeKuttaMethod method{};
    RungeKuttaMethod parabolic_method{};

    switch (time_stepping_scheme_) {
    case ssprk_22:
      [[fallthrough]];
    case ssprk_33:
      ssprk = true;
      break;
    case erk_11:
      method = M::erk_11;
      break;
    case erk_22:
      method = M::erk_22;
      break;
    case erk_33:
      method = M::erk_33;
      break;
    case erk_43:
      method = M::erk_43;
      break;
    case erk_54:
      method = M::erk_54;
      break;
    case strang_ssprk_33_cn:
      ssprk = true;
      strang = true;
      parabolic_method = M::dirk_midpoint;
      break;
    case strang_erk_33_cn:
      strang = true;
      method = M::erk_33;
      parabolic_method = M::dirk_midpoint;
      break;
    case strang_erk_43_cn:
      strang = true;
      method = M::erk_43;
      parabolic_method = M::dirk_midpoint;
      break;
    case strang_erk_33_dirk_22:
      strang = true;
      method = M::erk_33;
      parabolic_method = M::dirk_22;
      break;
    case strang_erk_33_dirk_23:
      strang = true;
      method = M::erk_33;
      parabolic_method = M::dirk_23;
      break;
    case imex_11:
      method = M::imex_11;
      break;
    case imex_22:
      method = M::imex_22;
      break;
    case imex_33:
      method = M::imex_33;
      break;
    case gsa_imex_22:
      method = M::gsa_imex_22;
      break;
    case gsa_imex_43:
      method = M::gsa_imex_43;
      break;
    }

    /*
     * Resize temporary storage: a Runge-Kutta method needs
     * n_temporary_vectors(), a strong stability preserving scheme needs a
     * single vector. The result is stored in the last vector. A Strang
     * splitting runs the stages of its two methods one after the other,
     * and needs an additional vector for its intermediate state.
     */

    std::size_t n_stages = 1;
    bool imex = false;
    this->efficiency_ = Number(1.);

    const bool implicit_flux = hyperbolic_module_->have_implicit_flux();

    Assert(!implicit_flux || !parabolic_system_->is_identity(),
           dealii::ExcMessage("An equation that splits off an implicit flux "
                              "requires a parabolic module that integrates "
                              "it."));

    if (!ssprk) {
      tableau_ = butcher_tableau<Number>(method);
      imex = tableau_.type() == ButcherTableau<Number>::Type::imex;
      n_stages = n_temporary_vectors(tableau_, *hyperbolic_module_);
      this->efficiency_ = Number(1.) / tableau_.maximal_increment();
    }

    if (strang) {
      parabolic_tableau_ = butcher_tableau<Number>(parabolic_method);
      n_stages = std::max(
          n_stages,
          n_temporary_vectors(parabolic_tableau_, *hyperbolic_module_));
      this->efficiency_ *= Number(2.);
    }

    temp_.resize(n_stages + (strang ? 2 : 1));

    /* Whether the scheme performs implicit steps with the parabolic module: */
    const bool implicit = imex || strang;

    /* Make sure the user selects an appropriate time-stepping scheme: */

    AssertThrow(
        implicit || parabolic_system_->is_identity(),
        dealii::ExcMessage(
            "The selected equation consists of a hyperbolic and nontrivial "
            "parabolic subsystem and requires an IMEX timestepping "
            "scheme such as »strang erk 33 cn«."));

    AssertThrow(
        !implicit || !parabolic_system_->is_identity(),
        dealii::ExcMessage(
            "The selected equation has a trivial parabolic subsystem and "
            "should not be run with an IMEX timestepping scheme."));

    AssertThrow(
        !ssprk || !implicit_flux,
        dealii::ExcMessage(
            "The selected equation splits off an implicit flux and cannot "
            "be used with an SSPRK scheme."));

    /* Initialize temporary vectors: */

    for (auto &it : temp_) {
      hyperbolic_module_->reinit_state_vector(it);
      parabolic_module_->reinit_state_vector(it);
    }

    /* Reset CFL to starting value, set maximal acceptable tau_max ratio: */

    AssertThrow(cfl_min_ > 0., ExcMessage("cfl min must be a positive value"));
    AssertThrow(cfl_max_ >= cfl_min_,
                ExcMessage("cfl max must be greater than or equal to cfl min"));

    AssertThrow(
        acceptable_tau_max_ratio_ >= 1.0,
        ExcMessage(
            "acceptable tau_max ratio must be greater than or equal to 1."));

    hyperbolic_module_->set_cfl(cfl_max_);
    hyperbolic_module_->set_acceptable_tau_max_ratio(acceptable_tau_max_ratio_);
  }


  /*
   * -------------------------------------------------------------------------
   * High level step function implementing various CFLRecoveryStrategy
   * -------------------------------------------------------------------------
   */


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step(
      StateVector<Number> &state_vector,
      Number t,
      Number tau,
      Number tau_max /*=std::numeric_limits<Number>::max()*/)
  {
    tau_max = std::min(tau_max_, tau_max);

    RYUJIN_DEBUG_ANNOUNCE();

#ifdef DEBUG_OUTPUT
    std::cout << "        enforcing tau_max <= " << tau_max << std::endl;
#endif

    const auto single_step = [&]() {
      /*
       * The result of a successful step is stored in the last temporary
       * vector.
       */
      auto &result = temp_.back();

      const auto &hyperbolic = *hyperbolic_module_;
      const auto &parabolic = *parabolic_module_;
      const auto temp = std::span(temp_).first(temp_.size() - 1);

      using enum TimeSteppingScheme;
      switch (time_stepping_scheme_) {
      case ssprk_22:
        tau = step_ssprk_22(
            hyperbolic, temp[0], state_vector, result, t, tau, tau_max);
        break;
      case ssprk_33:
        tau = step_ssprk_33(
            hyperbolic, temp[0], state_vector, result, t, tau, tau_max);
        break;
      case strang_ssprk_33_cn:
        [[fallthrough]];
      case strang_erk_33_cn:
        [[fallthrough]];
      case strang_erk_43_cn:
        [[fallthrough]];
      case strang_erk_33_dirk_22:
        [[fallthrough]];
      case strang_erk_33_dirk_23: {
        /*
         * The explicit and the implicit step share all but the last
         * temporary vector of temp, the last one holds the intermediate
         * state of the Strang split:
         */
        const auto inner_temp = temp.first(temp.size() - 1);

        const auto explicit_step = [&](const StateVector<Number> &old_U,
                                       StateVector<Number> &new_U,
                                       const Number time,
                                       const Number step_size,
                                       const Number step_size_max) {
          if (time_stepping_scheme_ == strang_ssprk_33_cn)
            return step_ssprk_33(hyperbolic,
                                 inner_temp[0],
                                 old_U,
                                 new_U,
                                 time,
                                 step_size,
                                 step_size_max);
          return step_runge_kutta(tableau_,
                                  hyperbolic,
                                  parabolic,
                                  inner_temp,
                                  old_U,
                                  new_U,
                                  time,
                                  step_size,
                                  step_size_max);
        };

        const auto implicit_step = [&](const StateVector<Number> &old_U,
                                       StateVector<Number> &new_U,
                                       const Number time,
                                       const Number step_size,
                                       const Number step_size_max) {
          return step_runge_kutta(parabolic_tableau_,
                                  hyperbolic,
                                  parabolic,
                                  inner_temp,
                                  old_U,
                                  new_U,
                                  time,
                                  step_size,
                                  step_size_max);
        };

        tau = step_strang<dim, Number>(hyperbolic,
                                       explicit_step,
                                       implicit_step,
                                       temp.back(),
                                       state_vector,
                                       result,
                                       t,
                                       tau,
                                       tau_max);
        break;
      }
      default:
        tau = step_runge_kutta(tableau_,
                               hyperbolic,
                               parabolic,
                               temp,
                               state_vector,
                               result,
                               t,
                               tau,
                               tau_max);
      }

      std::swap(state_vector, result);
      return tau;
    };

    if (cfl_recovery_strategy_ != CFLRecoveryStrategy::none) {
      hyperbolic_module_->set_id_violation_strategy(
          IDViolationStrategy::raise_exception);
      parabolic_module_->set_id_violation_strategy(
          IDViolationStrategy::raise_exception);
      hyperbolic_module_->set_cfl(cfl_max_);
    }

    try {
      return single_step();

    } catch (const Restart &restart) {

      AssertThrow(cfl_recovery_strategy_ != CFLRecoveryStrategy::none,
                  dealii::ExcInternalError());

      hyperbolic_module_->set_id_violation_strategy(IDViolationStrategy::warn);
      parabolic_module_->set_id_violation_strategy(IDViolationStrategy::warn);

      /* A prescribed time step size cannot be adjusted: */
      if (tau != Number(0.))
        throw;

      if (cfl_recovery_strategy_ == CFLRecoveryStrategy::bang_bang_control) {
        /* Retry with cfl_min instead of cfl_max: */
#ifdef DEBUG_OUTPUT
        std::cout
            << "        restart with bang bang control: setting cfl to cfl_min"
            << std::endl;
#endif
        hyperbolic_module_->set_cfl(cfl_min_);
      }

      if (cfl_recovery_strategy_ == CFLRecoveryStrategy::cruise_control) {
        /* Retry with the suggested tau_max: */
#ifdef DEBUG_OUTPUT
        std::cout
            << "        restart with cruise control: using suggested_tau_max"
            << std::endl;
#endif
        /*
         * The step functions report the suggested tau_max in terms of the
         * step size of the full time step:
         */
        tau_max = std::min(tau_max, Number(restart.suggested_tau_max));
      }

      return single_step();
    }
  }
} // namespace ryujin
