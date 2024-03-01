//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2024 by the ryujin authors
//

#pragma once

#include "time_integrator.h"

namespace ryujin
{
  using namespace dealii;

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
      U_.resize(2);
      precomputed_.resize(1);
      efficiency_ = 1.;
      break;
    case TimeSteppingScheme::ssprk_33:
      U_.resize(2);
      precomputed_.resize(1);
      efficiency_ = 1.;
      break;
    case TimeSteppingScheme::erk_11:
      U_.resize(1);
      precomputed_.resize(1);
      efficiency_ = 1.;
      break;
    case TimeSteppingScheme::erk_22:
      U_.resize(2);
      precomputed_.resize(2);
      efficiency_ = 2.;
      break;
    case TimeSteppingScheme::erk_33:
      U_.resize(3);
      precomputed_.resize(3);
      efficiency_ = 3.;
      break;
    case TimeSteppingScheme::erk_43:
      U_.resize(4);
      precomputed_.resize(4);
      efficiency_ = 4.;
      break;
    case TimeSteppingScheme::erk_54:
      U_.resize(5);
      precomputed_.resize(5);
      efficiency_ = 5.;
      break;
    case TimeSteppingScheme::strang_ssprk_33_cn:
      U_.resize(3);
      precomputed_.resize(1);
      efficiency_ = 2.;
      break;
    case TimeSteppingScheme::strang_erk_33_cn:
      U_.resize(4);
      precomputed_.resize(3);
      efficiency_ = 6.;
      break;
    case TimeSteppingScheme::strang_erk_43_cn:
      U_.resize(4); // FIXME
      precomputed_.resize(4);
      efficiency_ = 8.;
      break;
    }

    /* Initialize temporary vectors and matrices: */

    const auto &vector_partitioner = offline_data_->vector_partitioner();
    for (auto &it : U_)
      it.reinit(vector_partitioner);

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    for (auto &it : precomputed_)
      it.reinit_with_scalar_partitioner(scalar_partitioner);

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
  Number TimeIntegrator<Description, dim, Number>::step(vector_type &U,
                                                        Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step()" << std::endl;
#endif

    const auto single_step = [&]() {
      switch (time_stepping_scheme_) {
      case TimeSteppingScheme::ssprk_22:
        return step_ssprk_22(U, t);
      case TimeSteppingScheme::ssprk_33:
        return step_ssprk_33(U, t);
      case TimeSteppingScheme::erk_11:
        return step_erk_11(U, t);
      case TimeSteppingScheme::erk_22:
        return step_erk_22(U, t);
      case TimeSteppingScheme::erk_33:
        return step_erk_33(U, t);
      case TimeSteppingScheme::erk_43:
        return step_erk_43(U, t);
      case TimeSteppingScheme::erk_54:
        return step_erk_54(U, t);
      case TimeSteppingScheme::strang_ssprk_33_cn:
        return step_strang_ssprk_33_cn(U, t);
      case TimeSteppingScheme::strang_erk_33_cn:
        return step_strang_erk_33_cn(U, t);
      case TimeSteppingScheme::strang_erk_43_cn:
        return step_strang_erk_43_cn(U, t);
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
  Number TimeIntegrator<Description, dim, Number>::step_ssprk_22(vector_type &U,
                                                                 Number t)
  {
    /* SSP-RK3, see @cite Shu1988, Eq. 2.15. */

    /* Step 1: U1 = U_old + tau * L(U_old) at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: U2 = 1/2 U_old + 1/2 (U1 + tau L(U1)) at time t + tau */
    hyperbolic_module_->template step<0>(
        U_[0], {}, {}, {}, U_[1], precomputed_[0], tau);
    U_[1].sadd(Number(1. / 2.), Number(1. / 2.), U);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + tau);

    U.swap(U_[1]);
    return tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_ssprk_33(vector_type &U,
                                                                 Number t)
  {
    /* SSP-RK3, see @cite Shu1988, Eq. 2.18. */

    /* Step 1: U1 = U_old + tau * L(U_old) at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: U2 = 3/4 U_old + 1/4 (U1 + tau L(U1)) at time t + 0.5 * tau */
    hyperbolic_module_->template step<0>(
        U_[0], {}, {}, {}, U_[1], precomputed_[0], tau);
    U_[1].sadd(Number(1. / 4.), Number(3. / 4.), U);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 0.5 * tau);

    /* Step 3: U3 = 1/3 U_old + 2/3 (U2 + tau L(U2)) at final time t + tau */
    hyperbolic_module_->template step<0>(
        U_[1], {}, {}, {}, U_[0], precomputed_[0], tau);
    U_[0].sadd(Number(2. / 3.), Number(1. / 3.), U);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    U.swap(U_[0]);
    return tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_11(vector_type &U,
                                                               Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_11()" << std::endl;
#endif

    /* Step 1: U1 <- {U, 1} at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    U.swap(U_[0]);
    return tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_22(vector_type &U,
                                                               Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_22()" << std::endl;
#endif

    /* Step 1: U1 <- {U, 1} at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: U2 <- {U1, 2} and {U, -1} at time t + 2 tau */
    hyperbolic_module_->template step<1>(U_[0],
                                         {{U}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 2. * tau);

    U.swap(U_[1]);
    return 2. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_33(vector_type &U,
                                                               Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_33()" << std::endl;
#endif

    /* Step 1: U1 <- {U, 1} at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: U2 <- {U1, 2} and {U, -1} at time t + 2 tau */
    hyperbolic_module_->template step<1>(U_[0],
                                         {{U}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 2. * tau);

    /* Step 3: U3 <- {U2, 9/4} and {U1, -2} and {U, 3/4} at time t + 3 tau */
    hyperbolic_module_->template step<2>(U_[1],
                                         {{U, U_[0]}},
                                         {{precomputed_[0], precomputed_[1]}},
                                         {{Number(0.75), Number(-2.)}},
                                         U_[2],
                                         precomputed_[2],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 3. * tau);

    U.swap(U_[2]);
    return 3. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_43(vector_type &U,
                                                               Number t)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_erk_43()" << std::endl;
#endif

    /* Step 1: U1 <- {U, 1} at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: U2 <- {U1, 2} and {U, -1} at time t + 2 tau */
    hyperbolic_module_->template step<1>(U_[0],
                                         {{U}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 2. * tau);

    /* Step 3: U3 <- {U2, 2} and {U1, -1} at time t + 3 tau */
    hyperbolic_module_->template step<1>(U_[1],
                                         {{U_[0]}},
                                         {{precomputed_[1]}},
                                         {{Number(-1.)}},
                                         U_[2],
                                         precomputed_[2],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 3. * tau);

    /* Step 4: U4 <- {U3, 8/3} and {U2,-10/3} and {U1, 5/3} at time t + 4 tau */
    hyperbolic_module_->template step<2>(U_[2],
                                         {{U_[0], U_[1]}},
                                         {{precomputed_[1], precomputed_[2]}},
                                         {{Number(5. / 3.), Number(-10. / 3.)}},
                                         U_[3],
                                         precomputed_[3],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[3], t + 4. * tau);

    U.swap(U_[3]);
    return 4. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_erk_54(vector_type &U,
                                                               Number t)
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

    /* Step 1: */
    Number tau = hyperbolic_module_->template step<0>(
        U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: */
    hyperbolic_module_->template step<1>(U_[0],
                                         {{U}},
                                         {{precomputed_[0]}},
                                         {{(a_31 - a_21) / c}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 2. * tau);

    /* Step 3: */
    hyperbolic_module_->template step<2>(
        U_[1],
        {{U, U_[0]}},
        {{precomputed_[0], precomputed_[1]}},
        {{(a_41 - a_31) / c, (a_42 - a_32) / c}},
        U_[2],
        precomputed_[2],
        tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 3. * tau);

    /* Step 4: */
    hyperbolic_module_->template step<3>(
        U_[2],
        {{U, U_[0], U_[1]}},
        {{precomputed_[0], precomputed_[1], precomputed_[2]}},
        {{(a_51 - a_41) / c, (a_52 - a_42) / c, (a_53 - a_43) / c}},
        U_[3],
        precomputed_[3],
        tau);
    hyperbolic_module_->apply_boundary_conditions(U_[3], t + 4. * tau);

    /* Step 5: */
    hyperbolic_module_->template step<4>(
        U_[3],
        {{U, U_[0], U_[1], U_[2]}},
        {{precomputed_[0], precomputed_[1], precomputed_[2], precomputed_[3]}},
        {{(a_61 - a_51) / c,
          (a_62 - a_52) / c,
          (a_63 - a_53) / c,
          (a_64 - a_54) / c}},
        U_[4],
        precomputed_[4],
        tau);
    hyperbolic_module_->apply_boundary_conditions(U_[4], t + 5. * tau);

    U.swap(U_[4]);
    return 5. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_strang_ssprk_33_cn(
      vector_type &U, Number t)
  {
    // FIXME: avoid code duplication with step_ssprk_33

#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_strang_ssprk_33_cn()"
              << std::endl;
#endif

    /* First explicit SSPRK 3 step with final result in U_[0]: */

    Number tau = hyperbolic_module_->template step<0>(
        /*input*/ U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    hyperbolic_module_->template step<0>(
        U_[0], {}, {}, {}, U_[1], precomputed_[0], tau);
    U_[1].sadd(Number(1. / 4.), Number(3. / 4.), /*input*/ U);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 0.5 * tau);

    hyperbolic_module_->template step<0>(
        U_[1], {}, {}, {}, U_[0], precomputed_[0], tau);
    U_[0].sadd(Number(2. / 3.), Number(1. / 3.), /*input*/ U);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Implicit Crank-Nicolson step with final result in U_[2]: */

    parabolic_module_->crank_nicolson_step(U_[0], t, U_[2], 2.0 * tau);

    /* Second SSPRK 3 step with final result in U_[0]: */

    hyperbolic_module_->template step<0>(
        /*intermediate*/ U_[2], {}, {}, {}, U_[0], precomputed_[0], tau);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + 2.0 * tau);

    hyperbolic_module_->template step<0>(
        U_[0], {}, {}, {}, U_[1], precomputed_[0], tau);
    U_[1].sadd(Number(1. / 4.), Number(3. / 4.), /*intermediate*/ U_[2]);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 1.5 * tau);

    hyperbolic_module_->template step<0>(
        U_[1], {}, {}, {}, U_[0], precomputed_[0], tau);
    U_[0].sadd(Number(2. / 3.), Number(1. / 3.), /*intermediate*/ U_[2]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + 2.0 * tau);

    U.swap(U_[0]);
    return 2.0 * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_strang_erk_33_cn(
      vector_type &U, Number t)
  {
    // FIXME: refactor to avoid code duplication with step_erk_33

#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_strang_erk_33_cn()"
              << std::endl;
#endif

    /* First explicit ERK(3,3,1) step with final result in U_[2]: */

    Number tau = hyperbolic_module_->template step<0>(
        /*input*/ U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    hyperbolic_module_->template step<1>(U_[0],
                                         {{/*input*/ U}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 2. * tau);

    hyperbolic_module_->template step<2>(U_[1],
                                         {{/*input*/ U, U_[0]}},
                                         {{precomputed_[0], precomputed_[1]}},
                                         {{Number(0.75), Number(-2.)}},
                                         U_[2],
                                         precomputed_[2],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 3. * tau);

    /* Implicit Crank-Nicolson step with final result in U_[3]: */

    parabolic_module_->crank_nicolson_step(U_[2], t, U_[3], 6.0 * tau);

    /* First explicit SSPRK 3 step with final result in U_[2]: */

    hyperbolic_module_->template step<0>(
        /*intermediate*/ U_[3], {}, {}, {}, U_[0], precomputed_[0], tau);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + 4. * tau);

    hyperbolic_module_->template step<1>(U_[0],
                                         {{/*intermediate*/ U_[3]}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 5. * tau);

    hyperbolic_module_->template step<2>(U_[1],
                                         {{/*intermediate*/ U_[3], U_[0]}},
                                         {{precomputed_[0], precomputed_[1]}},
                                         {{Number(0.75), Number(-2.)}},
                                         U_[2],
                                         precomputed_[2],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 6. * tau);

    U.swap(U_[2]);
    return 6. * tau;
  }


  template <typename Description, int dim, typename Number>
  Number TimeIntegrator<Description, dim, Number>::step_strang_erk_43_cn(
      vector_type &U, Number t)
  {
    // FIXME: refactor to avoid code duplication with step_erk_43

#ifdef DEBUG_OUTPUT
    std::cout << "TimeIntegrator<dim, Number>::step_strang_erk_43_cn()"
              << std::endl;
#endif

    /* First explicit ERK(4,3,1) step with final result in U_[3]: */

    /* Step 1: U1 <- {U, 1} at time t + tau */
    Number tau = hyperbolic_module_->template step<0>(
        /*input*/ U, {}, {}, {}, U_[0], precomputed_[0]);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + tau);

    /* Step 2: U2 <- {U1, 2} and {U, -1} at time t + 2 tau */
    hyperbolic_module_->template step<1>(U_[0],
                                         {{/*input*/ U}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 2. * tau);

    /* Step 3: U3 <- {U2, 2} and {U1, -1} at time t + 3 tau */
    hyperbolic_module_->template step<1>(U_[1],
                                         {{U_[0]}},
                                         {{precomputed_[1]}},
                                         {{Number(-1.)}},
                                         U_[2],
                                         precomputed_[2],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 3. * tau);

    /* Step 4: U4 <- {U3, 8/3} and {U2,-10/3} and {U1, 5/3} at time t + 4 tau */
    hyperbolic_module_->template step<2>(U_[2],
                                         {{U_[0], U_[1]}},
                                         {{precomputed_[1], precomputed_[2]}},
                                         {{Number(5. / 3.), Number(-10. / 3.)}},
                                         U_[3],
                                         precomputed_[3],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[3], t + 4. * tau);

    /* Implicit Crank-Nicolson step with final result in U_[2]: */

    parabolic_module_->crank_nicolson_step(U_[3], t, U_[2], 8.0 * tau);

    /* First explicit SSPRK 3 step with final result in U_[3]: */

    /* Step 1: U1 <- {U, 1} at time t + tau */
    hyperbolic_module_->template step<0>(
        /*intermediate*/ U_[2], {}, {}, {}, U_[0], precomputed_[0], tau);
    hyperbolic_module_->apply_boundary_conditions(U_[0], t + 5. * tau);

    /* Step 2: U2 <- {U1, 2} and {U, -1} at time t + 2 tau */
    hyperbolic_module_->template step<1>(U_[0],
                                         {{/*intermediate*/ U_[2]}},
                                         {{precomputed_[0]}},
                                         {{Number(-1.)}},
                                         U_[1],
                                         precomputed_[1],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[1], t + 6. * tau);

    /* Step 3: U3 <- {U2, 2} and {U1, -1} at time t + 3 tau */
    hyperbolic_module_->template step<1>(U_[1],
                                         {{U_[0]}},
                                         {{precomputed_[1]}},
                                         {{Number(-1.)}},
                                         U_[2],
                                         precomputed_[2],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[2], t + 7. * tau);

    /* Step 4: U4 <- {U3, 8/3} and {U2,-10/3} and {U1, 5/3} at time t + 4 tau */
    hyperbolic_module_->template step<2>(U_[2],
                                         {{U_[0], U_[1]}},
                                         {{precomputed_[1], precomputed_[2]}},
                                         {{Number(5. / 3.), Number(-10. / 3.)}},
                                         U_[3],
                                         precomputed_[3],
                                         tau);
    hyperbolic_module_->apply_boundary_conditions(U_[3], t + 8. * tau);

    U.swap(U_[3]);
    return 8. * tau;
  }

} /* namespace ryujin */
