//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/convenience_macros.h>
#include <ryujin/discretization/time_stepping.h>

#include <functional>
#include <vector>

namespace ryujin
{
  template <int dim, typename Number>
  std::size_t
  n_temporary_vectors(const ButcherTableau<Number> &tableau,
                      const Interface::HyperbolicModule<dim, Number> &module)
  {
    /*
     * All stages except the first and the last one, and a vector for the
     * hyperbolic prediction of an implicit-explicit scheme, or of an
     * explicit scheme if the hyperbolic module splits off an implicit
     * flux, see step_runge_kutta():
     */
    using Type = typename ButcherTableau<Number>::Type;
    const auto type = tableau.type();
    const bool hyperbolic_prediction =
        type == Type::imex ||
        (type == Type::erk && module.have_implicit_flux());
    return tableau.c.size() - 2 + (hyperbolic_prediction ? 1 : 0);
  }


  template <int dim, typename Number>
  Number step_ssprk_22(const Interface::HyperbolicModule<dim, Number> &module,
                       StateVector<Number> &temp,
                       const StateVector<Number> &old_state_vector,
                       StateVector<Number> &new_state_vector,
                       Number t,
                       Number tau,
                       Number tau_max)
  {
    /* SSP-RK2, see @cite Shu1988, Eq. 2.15. */

    RYUJIN_DEBUG_ANNOUNCE();

    /* Step 1: T0 = U_old + tau * L(U_old) at t -> t + tau */
    tau = module.forward_euler_step(
        old_state_vector, {}, {}, temp, t, tau, tau_max);

    /* Step 2: U_new = T0 + tau L(T0) at time t + tau -> t + 2*tau */
    module.prepare_state_vector(temp, t + 1.0 * tau);
    module.forward_euler_step(
        temp, {}, {}, new_state_vector, t + 1.0 * tau, tau);

    /* Step 2: convex combination: U_new = 1/2 U_old + 1/2 U_new at t + tau */
    sadd<selected_memory_space_t>(new_state_vector,
                                  Number(1.0 / 2.0),
                                  Number(1.0 / 2.0),
                                  old_state_vector);

    return tau;
  }


  template <int dim, typename Number>
  Number step_ssprk_33(const Interface::HyperbolicModule<dim, Number> &module,
                       StateVector<Number> &temp,
                       const StateVector<Number> &old_state_vector,
                       StateVector<Number> &new_state_vector,
                       Number t,
                       Number tau,
                       Number tau_max)
  {
    /* SSP-RK3, see @cite Shu1988, Eq. 2.18. */

    RYUJIN_DEBUG_ANNOUNCE();

    /* Step 1: U_new = U_old + tau * L(U_old) at time t -> t + tau */
    tau = module.forward_euler_step(
        old_state_vector, {}, {}, new_state_vector, t, tau, tau_max);

    /* Step 2: T0 = U_new + tau L(U_new) at time t + tau -> t + 2*tau */
    module.prepare_state_vector(new_state_vector, t + 1.0 * tau);
    module.forward_euler_step(
        new_state_vector, {}, {}, temp, t + 1.0 * tau, tau);

    /* Step 2: convex combination T0 = 3/4 U_old + 1/4 T0 at time t + 0.5*tau */
    sadd<selected_memory_space_t>(
        temp, Number(1.0 / 4.0), Number(3.0 / 4.0), old_state_vector);

    /* Step 3: U_new = T0 + tau L(T0) at time t + 0.5*tau -> t + 1.5*tau */
    module.prepare_state_vector(temp, t + 0.5 * tau);
    module.forward_euler_step(
        temp, {}, {}, new_state_vector, t + 0.5 * tau, tau);

    /* Step 3: convex combination: U_new = 1/3 U_old + 2/3 U_new at t + tau */
    sadd<selected_memory_space_t>(new_state_vector,
                                  Number(2.0 / 3.0),
                                  Number(1.0 / 3.0),
                                  old_state_vector);

    return tau;
  }


  template <int dim, typename Number>
  Number step_runge_kutta(
      const ButcherTableau<Number> &tableau,
      const Interface::HyperbolicModule<dim, Number> &hyperbolic_module,
      const Interface::ParabolicModule<dim, Number> &parabolic_module,
      std::span<StateVector<Number>> temp,
      const StateVector<Number> &old_state_vector,
      StateVector<Number> &new_state_vector,
      Number t,
      Number tau,
      Number tau_max)
  {
    RYUJIN_DEBUG_ANNOUNCE();

    using Tableau = ButcherTableau<Number>;

    const auto &c = tableau.c;
    const auto &a_explicit = tableau.a_explicit;
    const auto &a_implicit = tableau.a_implicit;

    const std::size_t n = c.size();
    const auto type = tableau.type();
    const bool hyperbolic = type != Tableau::Type::dirk;
    const bool parabolic = type != Tableau::Type::erk;
    const bool imex = hyperbolic && parabolic;

    /*
     * If the hyperbolic module splits off an implicit flux, every
     * hyperbolic prediction (also of the explicit substep of a Strang
     * splitting) contains the implicit flux of the start stage, which the
     * parabolic module has to remove again:
     */
    const bool implicit_flux = hyperbolic_module.have_implicit_flux();
    const bool hyperbolic_prediction = imex || (hyperbolic && implicit_flux);
    const auto start_stages = tableau.start_stages();
    const Number dc_max = tableau.maximal_increment();

    Assert(hyperbolic || tau != Number(0.),
           dealii::ExcMessage("A diagonally implicit scheme cannot choose the "
                              "time step size: tau must be nonzero."));

    /*
     * Storage: the first stage is the old state and the last stage is the
     * new state. All other stages are stored in temp. A scheme with
     * hyperbolic prediction uses temp[0] for the hyperbolic prediction of
     * a stage.
     */
    const std::size_t offset = hyperbolic_prediction ? 1 : 0;
    Assert(temp.size() >= n_temporary_vectors(tableau, hyperbolic_module),
           dealii::ExcMessage("Insufficient number of temporary vectors."));
    const auto stage = [&](const std::size_t l) -> StateVector<Number> & {
      return l == n - 1 ? new_state_vector : temp[offset + l - 1];
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
         * For a diagonally implicit scheme the explicit weights are those
         * of the strictly lower part of the implicit tableau: the known
         * stages enter with the parabolic operator, only the new stage is
         * linearized, see Interface::ParabolicModule::backward_euler_step().
         */
        const auto weight_strictly_lower = [&](const std::size_t k) {
          const Number a_l_prime_k =
              k < l_prime ? Tableau::entry(a_implicit, l_prime, k) : Number(0.);
          return (Tableau::entry(a_implicit, l, k) - a_l_prime_k) / dc;
        };

        /*
         * Explicit step: at time t + c_l' * tau -> t + c_l * tau
         *
         * The stage l' is the old state of the explicit step and does not
         * have to be supplied. We also skip all stages that do not
         * contribute. A diagonally implicit scheme uses the stage l' as
         * prediction.
         */

        const StateVector<Number> *prediction = &U(l_prime);

        if (hyperbolic) {
          vectors.clear();
          weights_explicit.clear();
          for (std::size_t k = 0; k < l; ++k) {
            const Number w = weight(a_explicit, k);
            if (k == l_prime || w == Number(0.))
              continue;
            vectors.push_back(U(k));
            weights_explicit.push_back(w);
          }

          auto &explicit_result = hyperbolic_prediction ? temp[0] : stage(l);
          prediction = &explicit_result;

          if (l == 1) {
            /*
             * The first explicit step determines the time-step size tau.
             * If the first stage is not the one with the largest increment
             * we have to reduce the CFL number accordingly so that the
             * time-step size is admissible for all stages.
             */
            const Number cfl = hyperbolic_module.cfl();
            hyperbolic_module.set_cfl(dc / dc_max * cfl);
            try {
              tau = hyperbolic_module.forward_euler_step(old_state_vector,
                                                         {},
                                                         {},
                                                         explicit_result,
                                                         t,
                                                         dc * tau,
                                                         dc * tau_max);
            } catch (...) {
              hyperbolic_module.set_cfl(cfl);
              throw;
            }
            hyperbolic_module.set_cfl(cfl);
            tau /= dc;

          } else {
            hyperbolic_module.forward_euler_step(U(l_prime),
                                                 vectors,
                                                 weights_explicit,
                                                 explicit_result,
                                                 t + c[l_prime] * tau,
                                                 dc * tau);
          }
        }

        /*
         * Implicit step: at time t + c_l' * tau -> t + c_l * tau
         *
         * We supply all prior stages. If the hyperbolic module splits off
         * an implicit flux, the hyperbolic prediction contains the implicit
         * flux of the start stage l' only, and we supply the explicit
         * weights of that forward Euler step. The prediction of a
         * diagonally implicit scheme, the start stage l', contains no
         * implicit flux and all explicit weights vanish. If the weight of
         * the new stage vanishes the stage is computed without implicit
         * solve.
         */

        if (parabolic || hyperbolic_prediction) {
          vectors.clear();
          weights_explicit.clear();
          weights_implicit.clear();
          for (std::size_t k = 0; k < l; ++k) {
            vectors.push_back(U(k));
            if (implicit_flux)
              weights_explicit.push_back(Number(hyperbolic && k == l_prime));
            else if (imex)
              weights_explicit.push_back(weight(a_explicit, k));
            else
              weights_explicit.push_back(weight_strictly_lower(k));
            weights_implicit.push_back(weight(a_implicit, k));
          }

          const Number weight_new = Tableau::entry(a_implicit, l, l) / dc;

          if (weight_new == Number(0.)) {
            parabolic_module.stage_combination_step(*prediction,
                                                    vectors,
                                                    weights_explicit,
                                                    weights_implicit,
                                                    stage(l),
                                                    t + c[l_prime] * tau,
                                                    dc * tau);
          } else {
            weights_implicit.push_back(weight_new);
            parabolic_module.backward_euler_step(*prediction,
                                                 vectors,
                                                 weights_explicit,
                                                 weights_implicit,
                                                 stage(l),
                                                 t + c[l_prime] * tau,
                                                 dc * tau);
          }
        }

        /* Prepare the stage for its use in subsequent stages: */
        if (hyperbolic && l < n - 1)
          hyperbolic_module.prepare_state_vector(stage(l), t + c[l] * tau);
      }

    } catch (Restart &restart) {
      /*
       * The suggested tau_max of a restart refers to the stage that
       * signalled the restart. If the hyperbolic module chooses the
       * time-step size we convert it into the step size of the full time
       * step. A diagonally implicit scheme passes the suggestion through.
       */
      if (hyperbolic)
        restart.suggested_tau_max /= dc;
      throw;
    }

    return tau;
  }

  template <int dim, typename Number>
  Number step_strang(const Interface::HyperbolicModule<dim, Number> &module,
                     const StepFunction<Number> &explicit_step,
                     const StepFunction<Number> &implicit_step,
                     StateVector<Number> &temp,
                     const StateVector<Number> &old_state_vector,
                     StateVector<Number> &new_state_vector,
                     Number t,
                     Number tau,
                     Number tau_max)
  {
    RYUJIN_DEBUG_ANNOUNCE();

    /*
     * The explicit steps are performed with half the step size: we double
     * the suggested tau_max of a restart.
     */
    const auto explicit_half_step = [&](const StateVector<Number> &old_U,
                                        StateVector<Number> &new_U,
                                        const Number time,
                                        const Number step_size,
                                        const Number step_size_max) {
      try {
        return explicit_step(old_U, new_U, time, step_size, step_size_max);
      } catch (Restart &restart) {
        restart.suggested_tau_max *= 2.;
        throw;
      }
    };

    /* First explicit step: at time t -> t + tau */
    tau = explicit_half_step(old_state_vector,
                             new_state_vector,
                             t,
                             tau / Number(2.),
                             tau_max / Number(2.));

    /*
     * Implicit step: at time t -> t + 2 * tau. The step is performed with
     * the full step size; a restart passes through.
     */
    implicit_step(new_state_vector, temp, t, Number(2.) * tau, tau_max);

    /* Second explicit step: at time t + tau -> t + 2 * tau */
    module.prepare_state_vector(temp, t + tau);
    explicit_half_step(temp, new_state_vector, t + tau, tau, tau);

    return Number(2.) * tau;
  }
} // namespace ryujin
