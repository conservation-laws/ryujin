//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/discretization/time_integrator.h>

#include <discretization/butcher_tableau.h>

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
        "imex 11, imex 22, imex 33, gsa imex 22, gsa imex 43");

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

    /* Resize temporary storage to appropriate sizes: */

    switch (time_stepping_scheme_) {
    case TimeSteppingScheme::ssprk_22:
      [[fallthrough]];
    case TimeSteppingScheme::ssprk_33:
      temp_.resize(2);
      this->efficiency_ = 1.;
      break;
    case TimeSteppingScheme::strang_ssprk_33_cn:
      temp_.resize(3);
      this->efficiency_ = 2.;
      break;
    default: {
      /*
       * A Runge-Kutta scheme described by a Butcher tableau with s
       * stages: we need storage for all stages except the first one. An
       * implicit-explicit scheme in addition needs a vector for the
       * hyperbolic prediction, and a Strang splitting a vector for the
       * result of the Crank-Nicolson step.
       */
      const auto tableau = butcher_tableau<Number>(time_stepping_scheme_);
      const auto n_stages = tableau.c.size() - 1;
      const bool strang =
          time_stepping_scheme_ == TimeSteppingScheme::strang_erk_33_cn ||
          time_stepping_scheme_ == TimeSteppingScheme::strang_erk_43_cn;
      temp_.resize(n_stages + (tableau.is_imex() || strang ? 1 : 0));
      this->efficiency_ =
          Number(strang ? 2. : 1.) / tableau.maximal_increment();
    } break;
    }

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

    /* Make sure the user selects an appropriate time-stepping scheme: */

    switch (time_stepping_scheme_) {
    case TimeSteppingScheme::ssprk_22:
      [[fallthrough]];
    case TimeSteppingScheme::ssprk_33:
      [[fallthrough]];
    case TimeSteppingScheme::erk_11:
      [[fallthrough]];
    case TimeSteppingScheme::erk_22:
      [[fallthrough]];
    case TimeSteppingScheme::erk_33:
      [[fallthrough]];
    case TimeSteppingScheme::erk_43:
      [[fallthrough]];
    case TimeSteppingScheme::erk_54: {
      AssertThrow(
          parabolic_system_->is_identity(),
          dealii::ExcMessage(
              "The selected equation consists of a hyperbolic and nontrivial "
              "parabolic subsystem and requires an IMEX timestepping "
              "scheme such as »strang erk 33 cn«."));
      break;
    }
    case TimeSteppingScheme::imex_11:
      [[fallthrough]];
    case TimeSteppingScheme::imex_22:
      [[fallthrough]];
    case TimeSteppingScheme::imex_33:
      [[fallthrough]];
    case TimeSteppingScheme::gsa_imex_22:
      [[fallthrough]];
    case TimeSteppingScheme::gsa_imex_43:
      [[fallthrough]];
    case TimeSteppingScheme::strang_ssprk_33_cn:
      [[fallthrough]];
    case TimeSteppingScheme::strang_erk_33_cn:
      [[fallthrough]];
    case TimeSteppingScheme::strang_erk_43_cn: {
      AssertThrow(
          !parabolic_system_->is_identity(),
          dealii::ExcMessage(
              "The selected equation has a trivial parabolic subsystem and "
              "should not be run with an IMEX timestepping scheme."));
      break;
    }
    }
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
      switch (time_stepping_scheme_) {
      case TimeSteppingScheme::ssprk_22:
        return step_ssprk_22(state_vector, t, tau, tau_max);
      case TimeSteppingScheme::ssprk_33:
        return step_ssprk_33(state_vector, t, tau, tau_max);
      case TimeSteppingScheme::strang_ssprk_33_cn:
        return step_strang_ssprk_33_cn(state_vector, t, tau, tau_max);
      case TimeSteppingScheme::strang_erk_33_cn:
        [[fallthrough]];
      case TimeSteppingScheme::strang_erk_43_cn:
        return step_strang_runge_kutta_cn(state_vector, t, tau, tau_max);
      default:
        tau = step_runge_kutta(state_vector, temp_.back(), t, tau, tau_max);
        std::swap(state_vector, temp_.back());
        return tau;
      }
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
        //
        // Multiply the suggested tau_max value with the efficiency.
        //
        // We have to account for the fact that the e Restart exception is
        // thrown within a substep of the hyperbolic or parabolic module.
        // This implies that the suggested_tau_max is computed for that
        // particular substep and not for the full combined method (where
        // tau_max can be larger). We thus multiply tau_max with the
        // efficiency factor.
        //
        tau_max = std::min(
            tau_max, this->efficiency_ * Number(restart.suggested_tau_max));
      }

      return single_step();
    }
  }


  /*
   * -------------------------------------------------------------------------
   * Concrete implementation of ERK / IMEX time stepping strategies.
   * -------------------------------------------------------------------------
   */


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step_ssprk_22(
      StateVector<Number> &state_vector, Number t, Number tau, Number tau_max)
  {
    /* SSP-RK2, see @cite Shu1988, Eq. 2.15. */

    RYUJIN_DEBUG_ANNOUNCE();

    Assert(this->efficiency_ == 1., dealii::ExcInternalError());

    /* Step 1: T0 = U_old + tau * L(U_old) at t -> t + tau */
    tau = hyperbolic_module_->forward_euler_step(
        state_vector, {}, {}, temp_[0], t, tau, tau_max);

    /* Step 2: T1 = T0 + tau L(T0) at time t + tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[0], {}, {}, temp_[1], t + 1.0 * tau, tau);

    /* Step 2: convex combination: T1 = 1/2 U_old + 1/2 T1 at time t + tau */
    sadd<selected_memory_space_t>(
        temp_[1], Number(1.0 / 2.0), Number(1.0 / 2.0), state_vector);

    std::swap(state_vector, temp_[1]);
    return tau;
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step_ssprk_33(
      StateVector<Number> &state_vector, Number t, Number tau, Number tau_max)
  {
    /* SSP-RK3, see @cite Shu1988, Eq. 2.18. */

    RYUJIN_DEBUG_ANNOUNCE();

    Assert(this->efficiency_ == 1., dealii::ExcInternalError());

    /* Step 1: T0 = U_old + tau * L(U_old) at time t -> t + tau */
    tau = hyperbolic_module_->forward_euler_step(
        state_vector, {}, {}, temp_[0], t, tau, tau_max);

    /* Step 2: T1 = T0 + tau L(T0) at time t + tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[0], {}, {}, temp_[1], t + 1.0 * tau, tau);

    /* Step 2: convex combination T1 = 3/4 U_old + 1/4 T1 at time t + 0.5*tau */
    sadd<selected_memory_space_t>(
        temp_[1], Number(1.0 / 4.0), Number(3.0 / 4.0), state_vector);

    /* Step 3: T0 = T1 + tau L(T1) at time t + 0.5*tau -> t + 1.5*tau */
    hyperbolic_module_->prepare_state_vector(temp_[1], t + 0.5 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[1], {}, {}, temp_[0], t + 0.5 * tau, tau);

    /* Step 3: convex combination: T0 = 1/3 U_old + 2/3 T0 at time t + tau */
    sadd<selected_memory_space_t>(
        temp_[0], Number(2.0 / 3.0), Number(1.0 / 3.0), state_vector);

    std::swap(state_vector, temp_[0]);
    return tau;
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step_strang_ssprk_33_cn(
      StateVector<Number> &state_vector, Number t, Number tau, Number tau_max)
  {
    // FIXME: avoid code duplication with step_ssprk_33

    RYUJIN_DEBUG_ANNOUNCE();

    Assert(this->efficiency_ == 2., dealii::ExcInternalError());

    /* First explicit SSPRK 3 step with final result in temp_[0]: */

    tau = hyperbolic_module_->forward_euler_step(state_vector,
                                                 {},
                                                 {},
                                                 temp_[0],
                                                 t,
                                                 tau / this->efficiency_,
                                                 tau_max / this->efficiency_);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[0], {}, {}, temp_[1], t + 1.0 * tau, tau);
    sadd<selected_memory_space_t>(
        temp_[1], Number(1.0 / 4.0), Number(3.0 / 4.0), /*!*/ state_vector);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 0.5 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[1], {}, {}, temp_[0], t + 0.5 * tau, tau);
    sadd<selected_memory_space_t>(
        temp_[0], Number(2.0 / 3.0), Number(1.0 / 3.0), /*!*/ state_vector);

    /* Implicit Crank-Nicolson step with final result in temp_[2]: */

    try {
      parabolic_module_->crank_nicolson_step(temp_[0], temp_[2], t, 2.0 * tau);
    } catch (Restart &restart) {
      /* Adjust suggested_tau_max. We multiply with this->efficiency_ again
       * later */
      restart.suggested_tau_max /= this->efficiency_;
      throw;
    }

    /* Second SSPRK 3 step with final result in temp_[0]: */

    hyperbolic_module_->prepare_state_vector(/*!*/ temp_[2], t + 1.0 * tau);
    hyperbolic_module_->forward_euler_step(
        /*!*/ temp_[2], {}, {}, temp_[0], t + 1.0 * tau, tau);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 2.0 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[0], {}, {}, temp_[1], t + 2.0 * tau, tau);
    sadd<selected_memory_space_t>(
        temp_[1], Number(1.0 / 4.0), Number(3.0 / 4.0), /*!*/ temp_[2]);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 1.5 * tau);
    hyperbolic_module_->forward_euler_step(
        temp_[1], {}, {}, temp_[0], t + 1.5 * tau, tau);
    sadd<selected_memory_space_t>(
        temp_[0], Number(2.0 / 3.0), Number(1.0 / 3.0), /*!*/ temp_[2]);

    std::swap(state_vector, temp_[0]);
    return this->efficiency_ * tau;
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step_runge_kutta(
      const StateVector<Number> &old_state_vector,
      StateVector<Number> &new_state_vector,
      Number t,
      Number tau,
      Number tau_max)
  {
    RYUJIN_DEBUG_ANNOUNCE();

    using Tableau = ButcherTableau<Number>;

    const auto tableau = butcher_tableau<Number>(time_stepping_scheme_);
    const auto &c = tableau.c;
    const auto &a_explicit = tableau.a_explicit;
    const auto &a_implicit = tableau.a_implicit;

    const std::size_t n = c.size();
    const bool imex = tableau.is_imex();
    const auto start_stages = tableau.start_stages();
    const Number dc_max = tableau.maximal_increment();

    /*
     * Storage: the first stage is the old state and the last stage is the
     * new state. All other stages are stored in temp_. An
     * implicit-explicit scheme uses temp_[0] for the hyperbolic
     * prediction of a stage.
     */
    const std::size_t offset = imex ? 1 : 0;
    const auto stage = [&](const std::size_t l) -> StateVector<Number> & {
      return l == n - 1 ? new_state_vector : temp_[offset + l - 1];
    };
    const auto U = [&](const std::size_t l) -> const StateVector<Number> & {
      return l == 0 ? old_state_vector : stage(l);
    };

    std::vector<std::reference_wrapper<const StateVector<Number>>> vectors;
    std::vector<Number> weights_explicit;
    std::vector<Number> weights_implicit;

    /* The increment c_l - c_l' of the stage that is currently computed: */
    Number dc = dc_max;

    try {
      for (std::size_t l = 1; l < n; ++l) {
        /*
         * We compute the stage l as an increment of the stage l'. In terms
         * of the Butcher tableau the weight of the stage k is given by
         * (a_lk - a_l'k) / (c_l - c_l').
         */
        const std::size_t l_prime = start_stages[l];
        dc = c[l] - c[l_prime];

        const auto weight = [&](const auto &a, const std::size_t k) {
          return (Tableau::entry(a, l, k) - Tableau::entry(a, l_prime, k)) / dc;
        };

        /*
         * Explicit step: at time t + c_l' * tau -> t + c_l * tau
         *
         * The stage l' is the old state of the explicit step and does not
         * have to be supplied. We also skip all stages that do not
         * contribute.
         */

        vectors.clear();
        weights_explicit.clear();
        for (std::size_t k = 0; k < l; ++k) {
          const Number w = weight(a_explicit, k);
          if (k == l_prime || w == Number(0.))
            continue;
          vectors.push_back(U(k));
          weights_explicit.push_back(w);
        }

        auto &prediction = imex ? temp_[0] : stage(l);

        if (l == 1) {
          /*
           * The first explicit step determines the time-step size tau. If
           * the first stage is not the one with the largest increment we
           * have to reduce the CFL number accordingly so that the
           * time-step size is admissible for all stages.
           */
          const Number cfl = hyperbolic_module_->cfl();
          hyperbolic_module_->set_cfl(dc / dc_max * cfl);
          try {
            tau = hyperbolic_module_->forward_euler_step(old_state_vector,
                                                         {},
                                                         {},
                                                         prediction,
                                                         t,
                                                         dc * tau,
                                                         dc * tau_max);
          } catch (...) {
            hyperbolic_module_->set_cfl(cfl);
            throw;
          }
          hyperbolic_module_->set_cfl(cfl);
          tau /= dc;

        } else {
          hyperbolic_module_->forward_euler_step(U(l_prime),
                                                 vectors,
                                                 weights_explicit,
                                                 prediction,
                                                 t + c[l_prime] * tau,
                                                 dc * tau);
        }

        /*
         * Implicit step: at time t + c_l' * tau -> t + c_l * tau
         *
         * We supply all prior stages. The last implicit weight is the
         * weight of the new stage.
         */

        if (imex) {
          vectors.clear();
          weights_explicit.clear();
          weights_implicit.clear();
          for (std::size_t k = 0; k < l; ++k) {
            vectors.push_back(U(k));
            weights_explicit.push_back(weight(a_explicit, k));
            weights_implicit.push_back(weight(a_implicit, k));
          }
          weights_implicit.push_back(Tableau::entry(a_implicit, l, l) / dc);

          parabolic_module_->backward_euler_step(prediction,
                                                 vectors,
                                                 weights_explicit,
                                                 weights_implicit,
                                                 stage(l),
                                                 t + c[l_prime] * tau,
                                                 dc * tau);
        }

        /* Prepare the stage for its use in subsequent stages: */
        if (l < n - 1)
          hyperbolic_module_->prepare_state_vector(stage(l), t + c[l] * tau);
      }

    } catch (Restart &restart) {
      /*
       * The suggested tau_max of a restart refers to the stage that
       * signalled the restart. Adjust it to the largest increment; we
       * multiply with this->efficiency_ later.
       */
      restart.suggested_tau_max *= dc_max / dc;
      throw;
    }

    return tau;
  }


  template <int dim, typename Number>
  Number TimeIntegrator<dim, Number>::step_strang_runge_kutta_cn(
      StateVector<Number> &state_vector, Number t, Number tau, Number tau_max)
  {
    RYUJIN_DEBUG_ANNOUNCE();

    /*
     * Storage: the stages of the explicit Runge-Kutta scheme are stored in
     * all but the last two vectors of temp_.
     */
    auto &U_explicit = temp_[temp_.size() - 2];
    auto &U_implicit = temp_[temp_.size() - 1];

    /* First explicit Runge-Kutta step: at time t -> t + tau */
    tau = step_runge_kutta(
        state_vector, U_explicit, t, tau / Number(2.), tau_max / Number(2.));

    /* Implicit Crank-Nicolson step: at time t -> t + 2 * tau */
    try {
      parabolic_module_->crank_nicolson_step(
          U_explicit, U_implicit, t, Number(2.) * tau);
    } catch (Restart &restart) {
      /* Adjust suggested_tau_max. We multiply with this->efficiency_ again
       * later */
      restart.suggested_tau_max /= this->efficiency_;
      throw;
    }

    /* Second explicit Runge-Kutta step: at time t + tau -> t + 2 * tau */
    hyperbolic_module_->prepare_state_vector(U_implicit, t + tau);
    step_runge_kutta(U_implicit, U_explicit, t + tau, tau, tau);

    std::swap(state_vector, U_explicit);
    return Number(2.) * tau;
  }
} // namespace ryujin
