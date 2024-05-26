//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include "time_integrator.h"

namespace ryujin
{
  using namespace dealii;


  /**
   * TODO: clear out precomputed vector and also scale add V.
   */
  template <typename StateVector, typename Number>
  void
  sadd(StateVector &dst, const Number s, const Number b, const StateVector &src)
  {
    auto &dst_U = std::get<0>(dst);
    auto &src_U = std::get<0>(src);
    dst_U.sadd(s, b, src_U);
  }


  template <typename Description, int dim, typename Number>
  TimeIntegrator<Description, dim, Number>::TimeIntegrator(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicModule<Description, dim, Number> &hyperbolic_module,
      const ParabolicModule<Description, dim, Number> &parabolic_module,
      const std::string &subsection /*= "TimeIntegrator"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , hyperbolic_module_(&hyperbolic_module)
      , parabolic_module_(&parabolic_module)
  {
    cfl_min_ = Number(0.45);
    add_parameter(
        "cfl min",
        cfl_min_,
        "Minimal admissible relative CFL constant. How this parameter is used "
        "depends on the chosen CFL recovery strategy");

    cfl_max_ = Number(0.90);
    add_parameter(
        "cfl max",
        cfl_max_,
        "Maximal admissible relative CFL constant. How this parameter is used "
        "depends on the chosen CFL recovery strategy");

    cfl_recovery_strategy_ = CFLRecoveryStrategy::bang_bang_control;
    add_parameter("cfl recovery strategy",
                  cfl_recovery_strategy_,
                  "CFL/invariant domain violation recovery strategy: none, "
                  "bang bang control");

    if (ParabolicSystem::is_identity)
      time_stepping_scheme_ = TimeSteppingScheme::erk_33;
    else
      time_stepping_scheme_ = TimeSteppingScheme::strang_erk_33_cn;
    add_parameter("time stepping scheme",
                  time_stepping_scheme_,
                  "Time stepping scheme: ssprk 22, ssprk 33, erk 11, erk 22, "
                  "erk 33, erk 43, erk "
                  "54, strang ssprk 33 cn, strang erk 33 cn, strang erk 43 cn");
  }


  template <typename Description, int dim, typename Number>
  void TimeIntegrator<Description, dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::prepare()" << std::endl;
#endif

    /* Resize temporary storage to appropriate sizes: */

    switch (time_stepping_scheme_) {
    case TimeSteppingScheme::ssprk_22:
      temp_.resize(2);
      efficiency_ = 1.;
      break;
    case TimeSteppingScheme::ssprk_33:
      temp_.resize(2);
      efficiency_ = 1.;
      break;
    case TimeSteppingScheme::erk_11:
      temp_.resize(1);
      efficiency_ = 1.;
      break;
    case TimeSteppingScheme::erk_22:
      temp_.resize(2);
      efficiency_ = 2.;
      break;
    case TimeSteppingScheme::erk_33:
      temp_.resize(3);
      efficiency_ = 3.;
      break;
    case TimeSteppingScheme::erk_43:
      temp_.resize(4);
      efficiency_ = 4.;
      break;
    case TimeSteppingScheme::erk_54:
      temp_.resize(5);
      efficiency_ = 5.;
      break;
    case TimeSteppingScheme::strang_ssprk_33_cn:
      temp_.resize(3);
      efficiency_ = 2.;
      break;
    case TimeSteppingScheme::strang_erk_33_cn:
      temp_.resize(4);
      efficiency_ = 6.;
      break;
    case TimeSteppingScheme::strang_erk_43_cn:
      temp_.resize(4);
      efficiency_ = 8.;
      break;
    }

    /* Initialize temporary vectors: */

    for (auto &it : temp_) {
      auto &[U, precomputed, V] = it;
      U.reinit(offline_data_->state_vector_partitioner());
      precomputed.reinit(offline_data_->precomputed_vector_partitioner());
    }

    /* Reset CFL to canonical starting value: */

    AssertThrow(cfl_min_ > 0., ExcMessage("cfl min must be a positive value"));
    AssertThrow(cfl_max_ >= cfl_min_,
                ExcMessage("cfl max must be greater than or equal to cfl min"));

    hyperbolic_module_->cfl(cfl_max_);

    const auto check_whether_timestepping_makes_sense = [&]() {
      /*
       * Make sure the user selects an appropriate time-stepping scheme.
       */

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
            ParabolicSystem::is_identity,
            dealii::ExcMessage(
                "The selected equation consists of a hyperbolic and nontrivial "
                "parabolic subsystem and requires an IMEX timestepping "
                "scheme such as »strang erk 33 cn«."));
        break;
      }
      case TimeSteppingScheme::strang_ssprk_33_cn:
        [[fallthrough]];
      case TimeSteppingScheme::strang_erk_33_cn:
        [[fallthrough]];
      case TimeSteppingScheme::strang_erk_43_cn: {
        AssertThrow(
            !ParabolicSystem::is_identity,
            dealii::ExcMessage(
                "The selected equation has a trivial parabolic subsystem and "
                "should not be run with an IMEX timestepping scheme."));
        break;
      }
      }
    };

    check_whether_timestepping_makes_sense();
    this->parse_parameters_call_back.connect(
        check_whether_timestepping_makes_sense);
  }


  template <typename Description, int dim, typename Number>
  Number
  TimeIntegrator<Description, dim, Number>::step(StateVector &state_vector,
                                                 Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step()" << std::endl;
#endif

    const auto single_step = [&]() {
      switch (time_stepping_scheme_) {
      case TimeSteppingScheme::ssprk_22:
        return step_ssprk_22(state_vector, t);
      case TimeSteppingScheme::ssprk_33:
        return step_ssprk_33(state_vector, t);
      case TimeSteppingScheme::erk_11:
        return step_erk_11(state_vector, t);
      case TimeSteppingScheme::erk_22:
        return step_erk_22(state_vector, t);
      case TimeSteppingScheme::erk_33:
        return step_erk_33(state_vector, t);
      case TimeSteppingScheme::erk_43:
        return step_erk_43(state_vector, t);
      case TimeSteppingScheme::erk_54:
        return step_erk_54(state_vector, t);
      case TimeSteppingScheme::strang_ssprk_33_cn:
        return step_strang_ssprk_33_cn(state_vector, t);
      case TimeSteppingScheme::strang_erk_33_cn:
        return step_strang_erk_33_cn(state_vector, t);
      case TimeSteppingScheme::strang_erk_43_cn:
        return step_strang_erk_43_cn(state_vector, t);
      default:
        __builtin_unreachable();
      }
    };

    if (cfl_recovery_strategy_ == CFLRecoveryStrategy::bang_bang_control) {
      hyperbolic_module_->id_violation_strategy_ =
          IDViolationStrategy::raise_exception;
      parabolic_module_->id_violation_strategy_ =
          IDViolationStrategy::raise_exception;
      hyperbolic_module_->cfl(cfl_max_);
    }

    try {
      return single_step();

    } catch (Restart) {

      AssertThrow(cfl_recovery_strategy_ != CFLRecoveryStrategy::none,
                  dealii::ExcInternalError());

      if (cfl_recovery_strategy_ == CFLRecoveryStrategy::bang_bang_control) {
        hyperbolic_module_->id_violation_strategy_ = IDViolationStrategy::warn;
        parabolic_module_->id_violation_strategy_ = IDViolationStrategy::warn;
        hyperbolic_module_->cfl(cfl_min_);
        return single_step();
      }

      __builtin_unreachable();
    }
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_ssprk_22(
      StateVector &state_vector, Number t)
  {
    /* SSP-RK2, see @cite Shu1988, Eq. 2.15. */

    /* Step 1: T0 = U_old + tau * L(U_old) at t -> t + tau */
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    /* Step 2: T1 = T0 + tau L(T0) at time t + tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<0>(temp_[0], {}, {}, temp_[1], tau);

    /* Step 2: convex combination: T1 = 1/2 U_old + 1/2 T1 at time t + tau */
    sadd(temp_[1], Number(1.0 / 2.0), Number(1.0 / 2.0), state_vector);

    state_vector.swap(temp_[1]);
    return tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_ssprk_33(
      StateVector &state_vector, Number t)
  {
    /* SSP-RK3, see @cite Shu1988, Eq. 2.18. */

    /* Step 1: T0 = U_old + tau * L(U_old) at time t -> t + tau */
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    /* Step 2: T1 = T0 + tau L(T0) at time t + tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<0>(temp_[0], {}, {}, temp_[1], tau);

    /* Step 2: convex combination T1 = 3/4 U_old + 1/4 T1 at time t + 0.5*tau */
    sadd(temp_[1], Number(1.0 / 4.0), Number(3.0 / 4.0), state_vector);

    /* Step 3: T0 = T1 + tau L(T1) at time t + 0.5*tau -> t + 1.5*tau */
    hyperbolic_module_->prepare_state_vector(temp_[1], t + 0.5 * tau);
    hyperbolic_module_->template step<0>(temp_[1], {}, {}, temp_[0], tau);

    /* Step 3: convex combination: T0 = 1/3 U_old + 2/3 T0 at time t + tau */
    sadd(temp_[0], Number(2.0 / 3.0), Number(1.0 / 3.0), state_vector);

    state_vector.swap(temp_[0]);
    return tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_11(
      StateVector &state_vector, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_11()" << std::endl;
#endif

    /* Step 1: T0 <- {U_old, 1} at time t -> t + tau */
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    state_vector.swap(temp_[0]);
    return tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_22(
      StateVector &state_vector, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_22()" << std::endl;
#endif

    /* Step 1: T0 <- {U_old, 1} at time t -> t + tau */
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    /* Step 2: T1 <- {T0, 2} and {U_old, -1} at time t + tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{state_vector}}, {{Number(-1.)}}, temp_[1], tau);

    state_vector.swap(temp_[1]);
    return 2. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_33(
      StateVector &state_vector, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_33()" << std::endl;
#endif

    /* Step 1: T0 <- {U_old, 1} at time t -> t + tau */
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    /* Step 2: T1 <- {T0, 2} and {U_old, -1} at time t + 1*tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{state_vector}}, {{Number(-1.)}}, temp_[1], tau);

    /* Step 3: T2 <- {T1, 9/4} and {T0, -2} and {U_old, 3/4}
     * at time t + 2*tau -> t + 3*tau */
    hyperbolic_module_->prepare_state_vector(temp_[1], t + 2.0 * tau);
    hyperbolic_module_->template step<2>(temp_[1],
                                         {{state_vector, temp_[0]}},
                                         {{Number(0.75), Number(-2.)}},
                                         temp_[2],
                                         tau);

    state_vector.swap(temp_[2]);
    return 3. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_43(
      StateVector &state_vector, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_43()" << std::endl;
#endif

    /* Step 1: T0 <- {U_old, 1} at time t -> t + tau */
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    /* Step 2: T1 <- {T0, 2} and {U_old, -1} at time t + 1*tau -> t + 2*tau */
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{state_vector}}, {{Number(-1.)}}, temp_[1], tau);

    /* Step 3: T2 <- {T1, 2} and {T0, -1} at time t + 2*tau -> t + 3*tau */
    hyperbolic_module_->prepare_state_vector(temp_[1], t + 2.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[1], {{temp_[0]}}, {{Number(-1.)}}, temp_[2], tau);

    /* Step 4: T3 <- {T2, 8/3} and {T1,-10/3} and {T0, 5/3}
     * at time t + 3*tau -> t + 4*tau */
    hyperbolic_module_->prepare_state_vector(temp_[2], t + 3.0 * tau);
    hyperbolic_module_->template step<2>(temp_[2],
                                         {{temp_[0], temp_[1]}},
                                         {{Number(5. / 3.), Number(-10. / 3.)}},
                                         temp_[3],
                                         tau);

    state_vector.swap(temp_[3]);
    return 4. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_54(
      StateVector &state_vector, Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_54()" << std::endl;
#endif

    constexpr Number c = 0.2; /* equidistant c_i */
    constexpr Number a_21 = +0.2;
    constexpr Number a_31 = +0.26075582269554909;
    constexpr Number a_32 = +0.13924417730445096;
    constexpr Number a_41 = -0.25856517872570289;
    constexpr Number a_42 = +0.91136274166280729;
    constexpr Number a_43 = -0.05279756293710430;
    constexpr Number a_51 = +0.21623276431503774;
    constexpr Number a_52 = +0.51534223099602405;
    constexpr Number a_53 = -0.81662794199265554;
    constexpr Number a_54 = +0.88505294668159373;
    constexpr Number a_61 = -0.10511678454691901; /* aka b_1 */
    constexpr Number a_62 = +0.87880047152100838; /* aka b_2 */
    constexpr Number a_63 = -0.58903404061484477; /* aka b_3 */
    constexpr Number a_64 = +0.46213380485434047; /* aka b_4 */
    // constexpr Number a_65 = +0.35321654878641495; /* aka b_5 */

    /* Step 1: at time t -> t + 1*tau*/
    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau =
        hyperbolic_module_->template step<0>(state_vector, {}, {}, temp_[0]);

    /* Step 2: at time t -> 1*tau -> t + 2*tau*/
    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{state_vector}}, {{(a_31 - a_21) / c}}, temp_[1], tau);

    /* Step 3: at time t -> 2*tau -> t + 3*tau*/
    hyperbolic_module_->prepare_state_vector(temp_[1], t + 2.0 * tau);
    hyperbolic_module_->template step<2>(
        temp_[1],
        {{state_vector, temp_[0]}},
        {{(a_41 - a_31) / c, (a_42 - a_32) / c}},
        temp_[2],
        tau);

    /* Step 4: at time t -> 3*tau -> t + 4*tau*/
    hyperbolic_module_->prepare_state_vector(temp_[2], t + 3.0 * tau);
    hyperbolic_module_->template step<3>(
        temp_[2],
        {{state_vector, temp_[0], temp_[1]}},
        {{(a_51 - a_41) / c, (a_52 - a_42) / c, (a_53 - a_43) / c}},
        temp_[3],
        tau);

    /* Step 5: at time t -> 4*tau -> t + 5*tau*/
    hyperbolic_module_->prepare_state_vector(temp_[3], t + 4.0 * tau);
    hyperbolic_module_->template step<4>(
        temp_[3],
        {{state_vector, temp_[0], temp_[1], temp_[2]}},
        {{(a_61 - a_51) / c,
          (a_62 - a_52) / c,
          (a_63 - a_53) / c,
          (a_64 - a_54) / c}},
        temp_[4],
        tau);

    state_vector.swap(temp_[4]);
    return 5. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_strang_ssprk_33_cn(
      StateVector &state_vector, Number t)
  {
    // FIXME: avoid code duplication with step_ssprk_33

#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_strang_ssprk_33_cn()"
              << std::endl;
#endif

    /* First explicit SSPRK 3 step with final result in temp_[0]: */

    hyperbolic_module_->prepare_state_vector(/*!*/ state_vector, t);
    Number tau = hyperbolic_module_->template step<0>(
        /*!*/ state_vector, {}, {}, temp_[0]);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<0>(temp_[0], {}, {}, temp_[1], tau);
    sadd(temp_[1], Number(1.0 / 4.0), Number(3.0 / 4.0), /*!*/ state_vector);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 0.5 * tau);
    hyperbolic_module_->template step<0>(temp_[1], {}, {}, temp_[0], tau);
    sadd(temp_[0], Number(2.0 / 3.0), Number(1.0 / 3.0), /*!*/ state_vector);

    /* Implicit Crank-Nicolson step with final result in temp_[2]: */

    parabolic_module_->template step<0>(temp_[0], t, {}, {}, temp_[2], tau);
    sadd(temp_[2], Number(2.), Number(-1.), temp_[0]);

    /* Second SSPRK 3 step with final result in temp_[0]: */

    hyperbolic_module_->prepare_state_vector(/*!*/ temp_[2], t + 1.0 * tau);
    hyperbolic_module_->template step<0>(/*!*/ temp_[2], {}, {}, temp_[0], tau);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 2.0 * tau);
    hyperbolic_module_->template step<0>(temp_[0], {}, {}, temp_[1], tau);
    sadd(temp_[1], Number(1.0 / 4.0), Number(3.0 / 4.0), /*!*/ temp_[2]);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 1.5 * tau);
    hyperbolic_module_->template step<0>(temp_[1], {}, {}, temp_[0], tau);
    sadd(temp_[0], Number(2.0 / 3.0), Number(1.0 / 3.0), /*!*/ temp_[2]);

    state_vector.swap(temp_[0]);
    return 2.0 * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_strang_erk_33_cn(
      StateVector &state_vector, Number t)
  {
    // FIXME: refactor to avoid code duplication with step_erk_33

#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_strang_erk_33_cn()"
              << std::endl;
#endif

    /* First explicit ERK(3,3,1) step with final result in temp_[2]: */

    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau = hyperbolic_module_->template step<0>(
        /*!*/ state_vector, {}, {}, temp_[0]);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{/*!*/ state_vector}}, {{Number(-1.)}}, temp_[1], tau);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 2.0 * tau);
    hyperbolic_module_->template step<2>(temp_[1],
                                         {{/*!*/ state_vector, temp_[0]}},
                                         {{Number(0.75), Number(-2.)}},
                                         temp_[2],
                                         tau);

    /* Implicit Crank-Nicolson step with final result in temp_[3]: */

    parabolic_module_->template step<0>(
        temp_[2], t, {}, {}, temp_[3], 3.0 * tau);
    sadd(temp_[3], Number(2.), Number(-1.), temp_[2]);

    /* Second explicit ERK(3,3,1) 3 step with final result in temp_[2]: */

    hyperbolic_module_->prepare_state_vector(temp_[3], t + 3.0 * tau);
    hyperbolic_module_->template step<0>(
        /*!*/ temp_[3], {}, {}, temp_[0], tau);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 4.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{/*!*/ temp_[3]}}, {{Number(-1.)}}, temp_[1], tau);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 5.0 * tau);
    hyperbolic_module_->template step<2>(temp_[1],
                                         {{/*!*/ temp_[3], temp_[0]}},
                                         {{Number(0.75), Number(-2.)}},
                                         temp_[2],
                                         tau);

    state_vector.swap(temp_[2]);
    return 6. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_strang_erk_43_cn(
      StateVector &state_vector, Number t)
  {
    // FIXME: refactor to avoid code duplication with step_erk_43

#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_strang_erk_43_cn()"
              << std::endl;
#endif

    /* First explicit ERK(4,3,1) step with final result in temp_[3]: */

    hyperbolic_module_->prepare_state_vector(state_vector, t);
    Number tau = hyperbolic_module_->template step<0>(
        /*!*/ state_vector, {}, {}, temp_[0]);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 1.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{/*!*/ state_vector}}, {{Number(-1.)}}, temp_[1], tau);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 2.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[1], {{temp_[0]}}, {{Number(-1.)}}, temp_[2], tau);

    hyperbolic_module_->prepare_state_vector(temp_[2], t + 3.0 * tau);
    hyperbolic_module_->template step<2>(temp_[2],
                                         {{temp_[0], temp_[1]}},
                                         {{Number(5. / 3.), Number(-10. / 3.)}},
                                         temp_[3],
                                         tau);

    /* Implicit Crank-Nicolson step with final result in temp_[2]: */

    parabolic_module_->template step<0>(
        temp_[3], t, {}, {}, temp_[2], 4.0 * tau);
    sadd(temp_[2], Number(2.), Number(-1.), temp_[3]);

    /* Second explicit ERK(4,3,1) step with final result in temp_[3]: */

    hyperbolic_module_->prepare_state_vector(temp_[2], t + 4.0 * tau);
    hyperbolic_module_->template step<0>(
        /*!*/ temp_[2], {}, {}, temp_[0], tau);

    hyperbolic_module_->prepare_state_vector(temp_[0], t + 5.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[0], {{/*!*/ temp_[2]}}, {{Number(-1.)}}, temp_[1], tau);

    hyperbolic_module_->prepare_state_vector(temp_[1], t + 6.0 * tau);
    hyperbolic_module_->template step<1>(
        temp_[1], {{temp_[0]}}, {{Number(-1.)}}, temp_[2], tau);

    hyperbolic_module_->prepare_state_vector(temp_[2], t + 7.0 * tau);
    hyperbolic_module_->template step<2>(temp_[2],
                                         {{temp_[0], temp_[1]}},
                                         {{Number(5. / 3.), Number(-10. / 3.)}},
                                         temp_[3],
                                         tau);

    state_vector.swap(temp_[3]);
    return 8. * tau;
  }

} /* namespace ryujin */
