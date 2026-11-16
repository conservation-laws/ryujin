//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/discretization/butcher_tableau.h>
#include <ryujin/interface/hyperbolic_module.h>
#include <ryujin/interface/parabolic_module.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <cstddef>
#include <functional>
#include <span>

namespace ryujin
{
  /**
   * @name Time-stepping schemes
   *
   * All step functions given below take a reference to a previous state
   * vector @p old_state_vector at time @p t, perform a time step and store
   * the result in @p new_state_vector. The old state vector is not
   * modified. The new state vector is not prepared.
   *
   * If @p tau is nonzero the step is performed with @p tau. If @p tau is 0
   * the time-step size is chosen such that it is less than or equal to
   * @p tau_max. The functions return the time-step size of the step.
   *
   * If a step signals a Restart, Restart::suggested_tau_max refers to the
   * step size of the full time step.
   *
   * The temporary vectors @p temp have to be initialized with
   * reinit_state_vector() of the hyperbolic and the parabolic module.
   *
   * @pre The @p old_state_vector has to be prepared with
   * Interface::HyperbolicModule::prepare_state_vector().
   *
   * @ingroup MeshAndOfflineData
   */
  //@{

  /**
   * The signature of a step function.
   */
  template <typename Number>
  using StepFunction =
      std::function<Number(const StateVector<Number> &old_state_vector,
                           StateVector<Number> &new_state_vector,
                           Number t,
                           Number tau,
                           Number tau_max)>;

  /**
   * Return the number of temporary vectors step_runge_kutta() needs for
   * the Butcher tableau @p tableau.
   */
  template <int dim, typename Number>
  std::size_t
  n_temporary_vectors(const ButcherTableau<Number> &tableau,
                      const Interface::HyperbolicModule<dim, Number> &module);

  /**
   * Perform all stages of the explicit, diagonally implicit, or
   * implicit-explicit Runge-Kutta scheme described by the Butcher tableau
   * @p tableau. Every stage consists of one explicit hyperbolic step
   * (unless the scheme is diagonally implicit) and one implicit parabolic
   * step (unless the scheme is explicit). The intermediate stages are
   * stored in the temporary vectors @p temp, see n_temporary_vectors().
   *
   * @note A diagonally implicit scheme does not choose the time-step size:
   * @p tau must be nonzero.
   */
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
      Number tau_max);

  /**
   * Perform an explicit second-order strong-stability preserving
   * Runge-Kutta SSPRK(2,2;1/2) time step, see @cite Shu1988, Eq. 2.15.
   */
  template <int dim, typename Number>
  Number step_ssprk_22(const Interface::HyperbolicModule<dim, Number> &module,
                       StateVector<Number> &temp,
                       const StateVector<Number> &old_state_vector,
                       StateVector<Number> &new_state_vector,
                       Number t,
                       Number tau,
                       Number tau_max);

  /**
   * Perform an explicit third-order strong-stability preserving
   * Runge-Kutta SSPRK(3,3;1/3) time step, see @cite Shu1988, Eq. 2.18.
   */
  template <int dim, typename Number>
  Number step_ssprk_33(const Interface::HyperbolicModule<dim, Number> &module,
                       StateVector<Number> &temp,
                       const StateVector<Number> &old_state_vector,
                       StateVector<Number> &new_state_vector,
                       Number t,
                       Number tau,
                       Number tau_max);

  /**
   * Perform a Strang split: a step with @p explicit_step with half the
   * step size, a step with @p implicit_step with the full step size, and
   * a second step with @p explicit_step with half the step size. The
   * function @p implicit_step is called with a nonzero step size.
   *
   * The intermediate state of the implicit step is stored in @p temp. The
   * step functions must not use @p temp, @p old_state_vector, or
   * @p new_state_vector as temporary storage.
   */
  template <int dim, typename Number>
  Number step_strang(const Interface::HyperbolicModule<dim, Number> &module,
                     const StepFunction<Number> &explicit_step,
                     const StepFunction<Number> &implicit_step,
                     StateVector<Number> &temp,
                     const StateVector<Number> &old_state_vector,
                     StateVector<Number> &new_state_vector,
                     Number t,
                     Number tau,
                     Number tau_max);

  //@}
} // namespace ryujin
