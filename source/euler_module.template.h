//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 by the ryujin authors
//

#ifndef EULER_MODULE_TEMPLATE_H
#define EULER_MODULE_TEMPLATE_H

#include "euler_module.h"
#include "openmp.h"
#include "scope.h"
#include "simd.h"

#include "indicator.h"
#include "riemann_solver.h"

#include <atomic>

#ifdef VALGRIND_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#endif

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_START(opt)
#define LIKWID_MARKER_STOP(opt)
#endif

namespace ryujin
{
  using namespace dealii;


  template <int dim, typename Number>
  EulerModule<dim, Number>::EulerModule(
      const MPI_Comm &mpi_communicator,
      std::map<std::string, dealii::Timer> &computing_timer,
      const ryujin::OfflineData<dim, Number> &offline_data,
      const ryujin::ProblemDescription &problem_description,
      const ryujin::InitialValues<dim, Number> &initial_values,
      const std::string &subsection /*= "EulerModule"*/)
      : ParameterAcceptor(subsection)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , problem_description_(&problem_description)
      , initial_values_(&initial_values)
      , n_restarts_(0)
  {
    cfl_update_ = Number(0.95);
    add_parameter(
        "cfl update", cfl_update_, "relative CFL constant used for update");

    cfl_max_ = Number(1.0);
    add_parameter(
        "cfl max", cfl_max_, "Maximal admissible relative CFL constant");

    time_step_order_ = 3;
    add_parameter(
        "time step order",
        time_step_order_,
        "Approximation order of time stepping method. Switches between Forward "
        "Euler, SSP Heun, and SSP Runge Kutta 3rd order");

    limiter_iter_ = 2;
    add_parameter(
        "limiter iterations", limiter_iter_, "Number of limiter iterations");
  }


  template <int dim, typename Number>
  void EulerModule<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::prepare()" << std::endl;
#endif

    /* Initialize vectors: */

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    second_variations_.reinit(scalar_partitioner);
    alpha_.reinit(scalar_partitioner);
    specific_entropies_.reinit(scalar_partitioner);
    evc_entropies_.reinit(scalar_partitioner);

    bounds_.reinit_with_scalar_partitioner(scalar_partitioner);

    const auto &vector_partitioner = offline_data_->vector_partitioner();
    r_.reinit(vector_partitioner);
    temp_euler_.reinit(vector_partitioner);
    temp_ssp_.reinit(vector_partitioner);

    /* Initialize matrices: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    dij_matrix_.reinit(sparsity_simd);
    lij_matrix_.reinit(sparsity_simd);
    lij_matrix_next_.reinit(sparsity_simd);
    pij_matrix_.reinit(sparsity_simd);
  }


  template <int dim, typename Number>
  Number
  EulerModule<dim, Number>::euler_step(vector_type &U, Number t, Number tau)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::euler_step()" << std::endl;
#endif

    CALLGRIND_START_INSTRUMENTATION

    using VA = VectorizedArray<Number>;

    /* Index ranges for the iteration over the sparsity pattern : */

    constexpr auto simd_length = VA::size();
    const unsigned int n_export_indices = offline_data_->n_export_indices();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();
    const unsigned int n_relevant = offline_data_->n_locally_relevant();

    /* References to precomputed matrices and the stencil: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &lumped_mass_matrix_inverse =
        offline_data_->lumped_mass_matrix_inverse();
    const auto &mass_matrix = offline_data_->mass_matrix();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();

    const auto &boundary_map = offline_data_->boundary_map();
    const Number measure_of_omega_inverse =
        Number(1.) / offline_data_->measure_of_omega();

    /* A monotonically increasing "channel" variable for mpi_tags: */
    unsigned int channel = 10;

    /*
     * Step 0: Precompute f(U) and the entropies of U
     */
    {
      Scope scope(computing_timer_, "time step [E] 0 - compute entropies");

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_0");

      const unsigned int size_regular = n_relevant / simd_length * simd_length;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < size_regular; i += simd_length) {
        const auto U_i = U.get_vectorized_tensor(i);
        simd_store(specific_entropies_,
                   problem_description_->specific_entropy(U_i),
                   i);

        const auto evc_entropy =
            Indicator<dim, double>::evc_entropy_ ==
                    Indicator<dim, double>::Entropy::mathematical
                ? problem_description_->mathematical_entropy(U_i)
                : problem_description_->harten_entropy(U_i);
        simd_store(evc_entropies_, evc_entropy, i);
      }

      for (unsigned int i = size_regular; i < n_relevant; ++i) {
        const auto U_i = U.get_tensor(i);

        specific_entropies_.local_element(i) =
            problem_description_->specific_entropy(U_i);

        evc_entropies_.local_element(i) =
            Indicator<dim, double>::evc_entropy_ ==
                    Indicator<dim, double>::Entropy::mathematical
                ? problem_description_->mathematical_entropy(U_i)
                : problem_description_->harten_entropy(U_i);
      }

      LIKWID_MARKER_STOP("time_step_0");
      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * Step 1: Compute off-diagonal d_ij, and alpha_i
     *
     * The computation of the d_ij is quite costly. So we do a trick to
     * save a bit of computational resources. Instead of computing all d_ij
     * entries for a row of a given local index i, we only compute d_ij for
     * which j > i,
     *
     *        llllrr
     *      l .xxxxx
     *      l ..xxxx
     *      l ...xxx
     *      l ....xx
     *      r ......
     *      r ......
     *
     *  and symmetrize in Step 2.
     *
     *  MM: We could save a bit more computational resources by only
     *  computing entries for which *IN A GLOBAL* enumeration j > i. But
     *  the index translation, subsequent symmetrization, and exchange
     *  sounds a bit too expensive...
     */

    {
      Scope scope(computing_timer_, "time step [E] 1 - compute d_ij, and alpha_i");

      SynchronizationDispatch synchronization_dispatch([&]() {
        alpha_.update_ghost_values_start(channel++);
        second_variations_.update_ghost_values_start(channel++);
      });

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_1");

      /* Stored thread locally: */
      RiemannSolver<dim, Number> riemann_solver_serial(*problem_description_);
      Indicator<dim, Number> indicator_serial(*problem_description_);

      /* Parallel non-vectorized loop: */
      RYUJIN_OMP_FOR_NOWAIT
      for (unsigned int i = n_internal; i < n_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom (periodic constraints) */
        if (row_length == 1) {
          continue;
        }

        const auto U_i = U.get_tensor(i);
        const Number mass = lumped_mass_matrix.local_element(i);
        const Number hd_i = mass * measure_of_omega_inverse;

        indicator_serial.reset(U_i, evc_entropies_.local_element(i));

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const unsigned int j = js[col_idx];

          const auto U_j = U.get_tensor(j);

          const auto c_ij = cij_matrix.get_tensor(i, col_idx);
          const auto beta_ij = betaij_matrix.get_entry(i, col_idx);
          indicator_serial.add(
              U_j, c_ij, beta_ij, evc_entropies_.local_element(j));

          /* Only iterate over the upper triangular portion of d_ij */
          if (j <= i)
            continue;

          const auto norm = c_ij.norm();
          const auto n_ij = c_ij / norm;

          const auto [lambda_max, p_star, n_iterations] =
              riemann_solver_serial.compute(U_i, U_j, n_ij);

          Number d = norm * lambda_max;

          /*
           * In case both dofs are located at the boundary we have to
           * symmetrize.
           */

          if (boundary_map.count(i) != 0 && boundary_map.count(j) != 0) {

            const auto c_ji = cij_matrix.get_transposed_tensor(i, col_idx);
            const auto norm_2 = c_ji.norm();
            const auto n_ji = c_ji / norm_2;

            auto [lambda_max_2, p_star_2, n_iterations_2] =
                riemann_solver_serial.compute(U_j, U_i, n_ji);
            d = std::max(d, norm_2 * lambda_max_2);
          }

          dij_matrix_.write_entry(d, i, col_idx);
        }

        alpha_.local_element(i) = indicator_serial.alpha(hd_i);
        second_variations_.local_element(i) =
            indicator_serial.second_variations();
      } /* parallel non-vectorized loop */

      /* Stored thread locally: */
      RiemannSolver<dim, VA> riemann_solver_simd(*problem_description_);
      Indicator<dim, VA> indicator_simd(*problem_description_);
      bool thread_ready = false;

      /* Parallel SIMD loop: */
      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_internal; i += simd_length) {

        synchronization_dispatch.check(thread_ready, i >= n_export_indices);

        const auto U_i = U.get_vectorized_tensor(i);
        const auto entropy_i = simd_load(evc_entropies_, i);

        indicator_simd.reset(U_i, entropy_i);

        const auto mass = simd_load(lumped_mass_matrix, i);
        const auto hd_i = mass * measure_of_omega_inverse;

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i) + simd_length;
        for (unsigned int col_idx = 1; col_idx < row_length;
             ++col_idx, js += simd_length) {

          bool all_below_diagonal = true;
          for (unsigned int k = 0; k < simd_length; ++k)
            if (js[k] >= i + k) {
              all_below_diagonal = false;
              break;
            }

          const auto U_j = U.get_vectorized_tensor(js);
          const auto entropy_j = simd_load(evc_entropies_, js);

          const auto c_ij = cij_matrix.get_vectorized_tensor(i, col_idx);
          const auto beta_ij = betaij_matrix.get_vectorized_entry(i, col_idx);
          indicator_simd.add(U_j, c_ij, beta_ij, entropy_j);

          /* Only iterate over the upper triangular portion of d_ij */
          if (all_below_diagonal)
            continue;

          const auto norm = c_ij.norm();
          const auto n_ij = c_ij / norm;

          const auto [lambda_max, p_star, n_iterations] =
              riemann_solver_simd.compute(U_i, U_j, n_ij);

          const auto d = norm * lambda_max;

          dij_matrix_.write_vectorized_entry(d, i, col_idx, true);
        }

        simd_store(alpha_, indicator_simd.alpha(hd_i), i);
        simd_store(second_variations_, indicator_simd.second_variations(), i);
      } /* parallel SIMD loop */

      LIKWID_MARKER_STOP("time_step_1");
      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * Step 2: Compute diagonal of d_ij, and maximal time-step size.
     */

    std::atomic<Number> tau_max{std::numeric_limits<Number>::infinity()};

    {
      Scope scope(computing_timer_, "time step [E] 2 - compute d_ii, and tau_max");

      /* Parallel region */
      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_2");

      /* Parallel non-vectorized loop: */
      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_owned; ++i) {

        const unsigned int row_length = sparsity_simd.row_length(i);

        /* Skip constrained degrees of freedom (periodic constraints) */
        if (row_length == 1)
          continue;

        Number d_sum = Number(0.);

        const unsigned int *js = sparsity_simd.columns(i);

        /* skip diagonal: */
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const auto j =
              *(i < n_internal ? js + col_idx * simd_length : js + col_idx);

          // fill lower triangular part of dij_matrix missing from step 1
          if (j < i) {
            const auto d_ji = dij_matrix_.get_transposed_entry(i, col_idx);
            dij_matrix_.write_entry(d_ji, i, col_idx);
          }

          d_sum -= dij_matrix_.get_entry(i, col_idx);
        }

        /* write diagonal element */
        dij_matrix_.write_entry(d_sum, i, 0);

        const Number mass = lumped_mass_matrix.local_element(i);
        const Number tau = cfl_update_ * mass / (Number(-2.) * d_sum);

        Number current_tau_max = tau_max.load();
        while (current_tau_max > tau &&
               !tau_max.compare_exchange_weak(current_tau_max, tau))
          ;
      } /* parallel non-vectorized loop */

      LIKWID_MARKER_STOP("time_step_2");
      RYUJIN_PARALLEL_REGION_END
    }

    {
      Scope scope(computing_timer_, "time step [E] 2 - synchronization barrier");

      alpha_.update_ghost_values_finish();
      second_variations_.update_ghost_values_finish();

      /* MPI Barrier: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      AssertThrow(!std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
                  ExcMessage("I'm sorry, Dave. I'm afraid I can't "
                             "do that. - We crashed."));

#ifdef DEBUG_OUTPUT
      std::cout << "        computed tau_max = " << tau_max << std::endl;
#endif
      tau = (tau == Number(0.) ? tau_max.load() : tau);
#ifdef DEBUG_OUTPUT
      std::cout << "        perform time-step with tau = " << tau << std::endl;
#endif

      if (tau * cfl_update_ > tau_max.load() * cfl_max_) {
#ifdef DEBUG_OUTPUT
        std::cout
            << "        insufficient CFL, refuse update and abort stepping"
            << std::endl;
#endif
        U[0] *= std::numeric_limits<Number>::quiet_NaN();
        return tau_max;
      }
    }

    /*
     * Step 3: Low-order update, also compute limiter bounds, R_i
     *
     *   \bar U_ij = 1/2 (U_i + U_j) - 1/2 (f_j - f_i) c_ij / d_ij^L
     *
     *        R_i = \sum_j - c_ij f_j + d_ij^H (U_j - U_i)
     *
     *   Low-order update: += tau / m_i * 2 d_ij^L (\bar U_ij)
     */

    {
      Scope scope(computing_timer_,
                  "time step [E] 3 - l.-o. update, bounds, and r_i");

      SynchronizationDispatch synchronization_dispatch([&]() {
        if (RYUJIN_LIKELY(limiter_iter_ != 0)) {
          r_.update_ghost_values_start(channel++);
        } else {
          /* If we do not do high-order, synchronize at this point: */
          temp_euler_.update_ghost_values_start(channel++);
        }
      });

      /* Parallel region */
      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_3");

      /* Nota bene: This bounds variable is thread local: */
      Limiter<dim, Number> limiter_serial(*problem_description_);

      /* Parallel non-vectorized loop: */
      RYUJIN_OMP_FOR_NOWAIT
      for (unsigned int i = n_internal; i < n_owned; ++i) {

        /* Skip constrained degrees of freedom (periodic constraints) */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        const auto U_i = U.get_tensor(i);
        const auto f_i = problem_description_->f(U_i);
        auto U_i_new = U_i;
        const auto alpha_i = alpha_.local_element(i);
        const auto variations_i = second_variations_.local_element(i);

        const Number m_i = lumped_mass_matrix.local_element(i);
        const Number m_i_inv = lumped_mass_matrix_inverse.local_element(i);

        rank1_type r_i;

        /* Clear bounds: */
        limiter_serial.reset(variations_i);

        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
          const auto j = js[col_idx];

          const auto U_j = U.get_tensor(j);
          const auto alpha_j = alpha_.local_element(j);
          const auto variations_j = second_variations_.local_element(j);

          const auto d_ij = dij_matrix_.get_entry(i, col_idx);
          const Number d_ij_inv = Number(1.) / d_ij;

          const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                     Indicator<dim, Number>::Indicators::
                                         entropy_viscosity_commutator
                                 ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                 : d_ij * std::max(alpha_i, alpha_j);

          dealii::Tensor<1, problem_dimension, Number> U_ij_bar;
          const auto c_ij = cij_matrix.get_tensor(i, col_idx);
          const auto f_j = problem_description_->f(U_j);

          for (unsigned int k = 0; k < problem_dimension; ++k) {
            const auto temp = (f_j[k] - f_i[k]) * c_ij;

            r_i[k] += -temp + d_ijH * (U_j - U_i)[k];
            U_ij_bar[k] =
                Number(0.5) * (U_i[k] + U_j[k]) - Number(0.5) * temp * d_ij_inv;
          }

          U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

          const auto beta_ij = betaij_matrix.get_entry(i, col_idx);

          limiter_serial.accumulate(U_i,
                                    U_j,
                                    U_ij_bar,
                                    beta_ij,
                                    specific_entropies_.local_element(j),
                                    variations_j,
                                    /* is diagonal */ col_idx == 0);
        }

        if (RYUJIN_UNLIKELY(limiter_iter_ == 0)) {
          /* Fix up boundary: */
          const auto range = boundary_map.equal_range(i);
          for (auto it = range.first; it != range.second; ++it) {
            const auto &[normal, id, position] = it->second;

            /*
             * Remove the normal component of the momentum for slip and
             * "no_slip" boundary conditions. It is a bit counterintuitive,
             * but we have to treat "no_slip" boundary conditions the same
             * way as "slip" boundary conditions in the Euler module. The
             * reason lies in the Strang splitting that we use to separate
             * the hyperbolic part from viscous effects. Only the latter
             * can cause "no slip" and thus the actual enforcement of v = 0
             * happens in the dissipation module.
             */
            if (id == Boundary::slip || id == Boundary::no_slip) {
              auto m = problem_description_->momentum(U_i_new);
              m -= 1. * (m * normal) * normal;
              for (unsigned int k = 0; k < dim; ++k)
                U_i_new[k + 1] = m[k];
            }

            /* On Dirichlet boundaries enforce initial conditions: */
            if (id == Boundary::dirichlet) {
              U_i_new = initial_values_->initial_state(position, t + tau);
            }
          }
        }

        temp_euler_.write_tensor(U_i_new, i);
        r_.write_tensor(r_i, i);

        const Number hd_i = m_i * measure_of_omega_inverse;
        limiter_serial.apply_relaxation(hd_i);
        bounds_.write_tensor(limiter_serial.bounds(), i);
      } /* parallel non-vectorized loop */

      /* Nota bene: This bounds variable is thread local: */
      Limiter<dim, VA> limiter_simd(*problem_description_);
      bool thread_ready = false;

      /* Parallel SIMD loop: */

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_internal; i += simd_length) {

        synchronization_dispatch.check(thread_ready, i >= n_export_indices);

        const auto U_i = U.get_vectorized_tensor(i);
        const auto f_i = problem_description_->f(U_i);
        auto U_i_new = U_i;
        const auto alpha_i = simd_load(alpha_, i);
        const auto variations_i = simd_load(second_variations_, i);

        const auto m_i = simd_load(lumped_mass_matrix, i);
        const auto m_i_inv = simd_load(lumped_mass_matrix_inverse, i);

        ProblemDescription::rank1_type<dim, VA> r_i;

        /* Clear bounds: */
        limiter_simd.reset(variations_i);

        const unsigned int *js = sparsity_simd.columns(i);
        const unsigned int row_length = sparsity_simd.row_length(i);

        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += simd_length) {

          const auto alpha_j = simd_load(alpha_, js);
          const auto variations_j = simd_load(second_variations_, js);

          const auto d_ij = dij_matrix_.get_vectorized_entry(i, col_idx);

          const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                     Indicator<dim, Number>::Indicators::
                                         entropy_viscosity_commutator
                                 ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                 : d_ij * std::max(alpha_i, alpha_j);

          const auto U_j = U.get_vectorized_tensor(js);

          dealii::Tensor<1, problem_dimension, VA> U_ij_bar;
          const auto c_ij = cij_matrix.get_vectorized_tensor(i, col_idx);
          const auto d_ij_inv = Number(1.) / d_ij;

          const auto f_j = problem_description_->f(U_j);
          for (unsigned int k = 0; k < problem_dimension; ++k) {
            const auto temp = (f_j[k] - f_i[k]) * c_ij;

            r_i[k] += -temp + d_ijH * (U_j[k] - U_i[k]);
            U_ij_bar[k] = Number(0.5) * (U_i[k] + U_j[k] - temp * d_ij_inv);
          }

          U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

          const auto beta_ij = betaij_matrix.get_vectorized_entry(i, col_idx);
          const auto entropy_j = simd_load(specific_entropies_, js);

          limiter_simd.accumulate(U_i,
                                  U_j,
                                  U_ij_bar,
                                  beta_ij,
                                  entropy_j,
                                  variations_j,
                                  /* is diagonal */ col_idx == 0);
        }

        temp_euler_.write_vectorized_tensor(U_i_new, i);
        r_.write_vectorized_tensor(r_i, i);

        const auto hd_i = m_i * measure_of_omega_inverse;
        limiter_simd.apply_relaxation(hd_i);
        bounds_.write_vectorized_tensor(limiter_simd.bounds(), i);
      } /* parallel SIMD loop */

      LIKWID_MARKER_STOP("time_step_3");
      RYUJIN_PARALLEL_REGION_END
    }

    {
      Scope scope(computing_timer_, "time step [E] 3 - synchronization");

      if (RYUJIN_LIKELY(limiter_iter_ != 0)) {
        r_.update_ghost_values_finish();
      } else {
        /* If we do not do high-order, synchronize at this point: */
        temp_euler_.update_ghost_values_finish();
      }
    }

    /*
     * Step 4: Compute P_ij, and l_ij (first round):
     *
     *    P_ij = tau / m_i / lambda ( (d_ij^H - d_ij^L) (U_i - U_j) +
     *                                (b_ij R_j - b_ji R_i) )
     */

    if (RYUJIN_LIKELY(limiter_iter_ != 0)) {
      Scope scope(computing_timer_, "time step [E] 4 - compute p_ij, and l_ij");

      SynchronizationDispatch synchronization_dispatch(
          [&]() { lij_matrix_.update_ghost_rows_start(channel++); });

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_4");

      /* Parallel non-vectorized loop: */

      RYUJIN_OMP_FOR_NOWAIT
      for (unsigned int i = n_internal; i < n_owned; ++i) {

        /* Skip constrained degrees of freedom (periodic constraints) */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        const auto bounds =
            bounds_.template get_tensor<std::array<Number, 3>>(i);

        const auto U_i_new = temp_euler_.get_tensor(i);
        const auto U_i = U.get_tensor(i);
        const auto r_i = r_.get_tensor(i);

        const auto alpha_i = alpha_.local_element(i);
        const Number m_i_inv = lumped_mass_matrix_inverse.local_element(i);

        const unsigned int *js = sparsity_simd.columns(i);
        const Number lambda_inv = Number(row_length - 1);

        for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
          const auto j = js[col_idx];

          const auto U_j = U.get_tensor(j);

          const auto r_j = r_.get_tensor(j);

          const auto alpha_j = alpha_.local_element(j);
          const Number m_j_inv = lumped_mass_matrix_inverse.local_element(j);

          const auto d_ij = dij_matrix_.get_entry(i, col_idx);
          const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                     Indicator<dim, Number>::Indicators::
                                         entropy_viscosity_commutator
                                 ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                 : d_ij * std::max(alpha_i, alpha_j);

          const auto m_ij = mass_matrix.get_entry(i, col_idx);
          const auto b_ij =
              (col_idx == 0 ? Number(1.) : Number(0.)) - m_ij * m_j_inv;
          const auto b_ji =
              (col_idx == 0 ? Number(1.) : Number(0.)) - m_ij * m_i_inv;

          const auto p_ij =
              tau * m_i_inv * lambda_inv *
              ((d_ijH - d_ij) * (U_j - U_i) + b_ij * r_j - b_ji * r_i);
          pij_matrix_.write_tensor(p_ij, i, col_idx);

          const auto l_ij = Limiter<dim, Number>::limit(
              *problem_description_, bounds, U_i_new, p_ij);
          lij_matrix_.write_entry(l_ij, i, col_idx);
        }
      } /* parallel non-vectorized loop */

      /* Parallel SIMD loop: */

      bool thread_ready = false;

      RYUJIN_OMP_FOR
      for (unsigned int i = 0; i < n_internal; i += simd_length) {

        synchronization_dispatch.check(thread_ready, i >= n_export_indices);

        const auto bounds =
            bounds_.template get_vectorized_tensor<std::array<VA, 3>>(i);

        const auto m_i_inv = simd_load(lumped_mass_matrix_inverse, i);

        const unsigned int row_length = sparsity_simd.row_length(i);
        const VA lambda_inv = Number(row_length - 1);

        const auto U_i_new = temp_euler_.get_vectorized_tensor(i);
        const auto U_i = U.get_vectorized_tensor(i);
        const auto r_i = r_.get_vectorized_tensor(i);
        const auto alpha_i = simd_load(alpha_, i);

        const unsigned int *js = sparsity_simd.columns(i);

        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += simd_length) {

          const auto m_j_inv = simd_load(lumped_mass_matrix_inverse, js);

          const auto alpha_j = simd_load(alpha_, js);

          const auto d_ij = dij_matrix_.get_vectorized_entry(i, col_idx);

          const auto d_ijH = Indicator<dim, Number>::indicator_ ==
                                     Indicator<dim, Number>::Indicators::
                                         entropy_viscosity_commutator
                                 ? d_ij * (alpha_i + alpha_j) * Number(.5)
                                 : d_ij * std::max(alpha_i, alpha_j);

          const auto m_ij = mass_matrix.get_vectorized_entry(i, col_idx);
          const auto b_ij = (col_idx == 0 ? VA(1.) : VA(0.)) - m_ij * m_j_inv;
          const auto b_ji = (col_idx == 0 ? VA(1.) : VA(0.)) - m_ij * m_i_inv;

          const auto U_j = U.get_vectorized_tensor(js);
          const auto r_j = r_.get_vectorized_tensor(js);

          const auto p_ij =
              tau * m_i_inv * lambda_inv *
              ((d_ijH - d_ij) * (U_j - U_i) + b_ij * r_j - b_ji * r_i);
          pij_matrix_.write_vectorized_tensor(p_ij, i, col_idx, true);

          const auto l_ij = Limiter<dim, VA>::limit(
              *problem_description_, bounds, U_i_new, p_ij);

          lij_matrix_.write_vectorized_entry(l_ij, i, col_idx, true);
        }
      } /* parallel SIMD loop */

      LIKWID_MARKER_STOP("time_step_4");
      RYUJIN_PARALLEL_REGION_END
    }

    if (RYUJIN_LIKELY(limiter_iter_ != 0)) {
      Scope scope(computing_timer_, "time step [E] 4 - synchronization");

      lij_matrix_.update_ghost_rows_finish();
    }

    /*
     * Step 5, 6, ..., 4 + limiter_iter_: Perform high-order update:
     *
     *   Symmetrize l_ij
     *   High-order update: += l_ij * lambda * P_ij
     *   Compute next l_ij
     */

    for (unsigned int pass = 0; pass < limiter_iter_; ++pass) {

      std::string step_no = std::to_string(5 + pass);
      std::string additional_step =
          pass + 1 < limiter_iter_ ? ", next l_ij" : "";
      bool last_round = (pass + 1 == limiter_iter_);

      {
        Scope scope(computing_timer_,
                    "time step [E] " + step_no + " - " +
                        "symmetrize l_ij, h.-o. update" + additional_step);

        SynchronizationDispatch synchronization_dispatch([&]() {
          if (last_round)
            temp_euler_.update_ghost_values_start(channel++);
          else
            lij_matrix_next_.update_ghost_rows_start(channel++);
        });

        RYUJIN_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START(("time_step_" + step_no).c_str());

        /* Stored thread locally: */
        AlignedVector<Number> lij_row_serial;

        /* Parallel non-vectorized loop: */
        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = n_internal; i < n_owned; ++i) {

          /* Skip constrained degrees of freedom (periodic constraints) */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          lij_row_serial.resize_fast(row_length);

          auto U_i_new = temp_euler_.get_tensor(i);

          const Number lambda = Number(1.) / Number(row_length - 1);

          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
            auto p_ij = pij_matrix_.get_tensor(i, col_idx);

            const auto l_ij =
                std::min(lij_matrix_.get_entry(i, col_idx),
                         lij_matrix_.get_transposed_entry(i, col_idx));

            U_i_new += l_ij * lambda * p_ij;

            if (!last_round)
              lij_row_serial[col_idx] = l_ij;
          }

          /* In the last round */
          if (last_round) {
            /* Fix up boundary: */
            const auto range = boundary_map.equal_range(i);
            for (auto it = range.first; it != range.second; ++it) {
              const auto &[normal, id, position] = it->second;

              /* see comment above */
              if (id == Boundary::slip || id == Boundary::no_slip) {
                auto m = problem_description_->momentum(U_i_new);
                m -= 1. * (m * normal) * normal;
                for (unsigned int k = 0; k < dim; ++k)
                  U_i_new[k + 1] = m[k];
              }

              /* see comment above */
              if (id == Boundary::dirichlet) {
                U_i_new = initial_values_->initial_state(position, t + tau);
              }
            }
          }

#ifdef CHECK_BOUNDS
          const auto rho_new = problem_description_->density(U_i_new);
          const auto e_new = problem_description_->internal_energy(U_i_new);
          const auto s_new = problem_description_->specific_entropy(U_i_new);

          AssertThrowSIMD(
              rho_new,
              [](auto val) { return val > Number(0.); },
              dealii::ExcMessage("Negative density."));

          AssertThrowSIMD(
              e_new,
              [](auto val) { return val > Number(0.); },
              dealii::ExcMessage("Negative internal energy."));

          AssertThrowSIMD(
              s_new,
              [](auto val) { return val > Number(0.); },
              dealii::ExcMessage("Negative specific entropy."));
#endif

          temp_euler_.write_tensor(U_i_new, i);

          /* Skip computating l_ij and updating p_ij in the last round */
          if (last_round)
            continue;

          const auto bounds =
              bounds_.template get_tensor<std::array<Number, 3>>(i);

          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {
            const auto old_l_ij = lij_row_serial[col_idx];
            const auto new_p_ij =
                (Number(1.) - old_l_ij) * pij_matrix_.get_tensor(i, col_idx);

            const auto new_l_ij = Limiter<dim, Number>::limit(
                *problem_description_, bounds, U_i_new, new_p_ij);

            /*
             * FIXME: If this if statement causes too much of a performance
             * penalty we could refactor it outside of the main loop.
             */
            if (RYUJIN_LIKELY(limiter_iter_ == 2)) {
              /*
               * Shortcut: We omit updating the p_ij vector and simply
               * write (1 - l_ij^(1)) * l_ij^(2) into the l_ij matrix. This
               * approach only works for two limiting steps.
               */
              lij_matrix_next_.write_entry(
                  (Number(1.) - old_l_ij) * new_l_ij, i, col_idx);
            } else {
              /*
               * @todo: This is expensive. If we ever end up using more
               * than two limiter passes we should implement this by
               * storing a scalar factor instead of writing back into p_ij.
               */
              lij_matrix_next_.write_entry(new_l_ij, i, col_idx);
              pij_matrix_.write_tensor(new_p_ij, i, col_idx);
            }
          }
        } /* parallel non-vectorized loop */

        /* Stored thread locally: */
        AlignedVector<VectorizedArray<Number>> lij_row_simd;
        bool thread_ready = false;

        /* Parallel vectorized loop: */
        RYUJIN_OMP_FOR
        for (unsigned int i = 0; i < n_internal; i += simd_length) {

          synchronization_dispatch.check(thread_ready, i >= n_export_indices);

          auto U_i_new = temp_euler_.get_vectorized_tensor(i);

          const unsigned int row_length = sparsity_simd.row_length(i);
          const Number lambda = Number(1.) / Number(row_length - 1);
          lij_row_simd.resize_fast(row_length);

          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {

            const auto l_ij = std::min(
                lij_matrix_.get_vectorized_entry(i, col_idx),
                lij_matrix_.get_vectorized_transposed_entry(i, col_idx));

            auto p_ij = pij_matrix_.get_vectorized_tensor(i, col_idx);

            U_i_new += l_ij * lambda * p_ij;

            if (!last_round)
              lij_row_simd[col_idx] = l_ij;
          }

#ifdef CHECK_BOUNDS
          const auto rho_new = problem_description_->density(U_i_new);
          const auto e_new = problem_description_->internal_energy(U_i_new);
          const auto s_new = problem_description_->specific_entropy(U_i_new);

          AssertThrowSIMD(
              rho_new,
              [](auto val) { return val > Number(0.); },
              dealii::ExcMessage("Negative density."));

          AssertThrowSIMD(
              e_new,
              [](auto val) { return val > Number(0.); },
              dealii::ExcMessage("Negative internal energy."));

          AssertThrowSIMD(
              s_new,
              [](auto val) { return val > Number(0.); },
              dealii::ExcMessage("Negative specific entropy."));
#endif
          temp_euler_.write_vectorized_tensor(U_i_new, i);

          /* Skip computating l_ij and updating p_ij in the last round */
          if (last_round)
            continue;

          const auto bounds =
              bounds_.template get_vectorized_tensor<std::array<VA, 3>>(i);
          for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {

            const auto old_l_ij = lij_row_simd[col_idx];

            const auto new_p_ij = (VA(1.) - old_l_ij) *
                                  pij_matrix_.get_vectorized_tensor(i, col_idx);

            const auto new_l_ij = Limiter<dim, VA>::limit(
                *problem_description_, bounds, U_i_new, new_p_ij);

            /*
             * FIXME: If this if statement causes too much of a performance
             * penalty we could refactor it outside of the main loop.
             */
            if (RYUJIN_LIKELY(limiter_iter_ == 2)) {
              /*
               * Shortcut: We omit updating the p_ij vector and simply
               * write (1 - l_ij^(1)) * l_ij^(2) into the l_ij matrix. This
               * approach only works for two limiting steps.
               */
              const auto entry =
                  (VectorizedArray<Number>(1.) - old_l_ij) * new_l_ij;
              lij_matrix_next_.write_vectorized_entry(entry, i, col_idx, true);
            } else {
              /*
               * @todo: This is expensive. If we ever end up using more
               * than two limiter passes we should implement this by
               * storing a scalar factor instead of writing back into p_ij.
               */
              lij_matrix_next_.write_vectorized_entry(
                  new_l_ij, i, col_idx, true);
              pij_matrix_.write_vectorized_tensor(new_p_ij, i, col_idx);
            }
          }
        }

        LIKWID_MARKER_STOP(("time_step_" + step_no).c_str());
        RYUJIN_PARALLEL_REGION_END
      }

      {
        Scope scope(computing_timer_,
                    "time step [E] " + step_no + " - synchronization");

        if (last_round)
          temp_euler_.update_ghost_values_finish();
        else {
          lij_matrix_next_.update_ghost_rows_finish();
          std::swap(lij_matrix_, lij_matrix_next_);
        }
      }
    } /* limiter_iter_ */

    /* And finally update the result: */
    U.swap(temp_euler_);

    CALLGRIND_STOP_INSTRUMENTATION

    return tau_max;
  }


  template <int dim, typename Number>
  Number EulerModule<dim, Number>::ssph2_step(vector_type &U,
                                              Number t,
                                              Number tau_0 /*= 0*/)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::ssph2_step()" << std::endl;
#endif

  restart_ssph2_step:
    /* This also copies ghost elements: */
    temp_ssp_ = U;

    /* Step 1: U1 = U_old + tau * L(U_old) */
    Number tau_1 = euler_step(U, t, tau_0);

    AssertThrow(tau_1 * cfl_max_ / cfl_update_ >= tau_0,
                ExcMessage("failed to recover from CFL violation"));
    tau_1 = (tau_0 == 0. ? tau_1 : tau_0);

    /* Step 2: U2 = 1/2 U_old + 1/2 (U1 + tau L(U1)) */
    const Number tau_2 = euler_step(U, t, tau_1);

    AssertThrow(tau_2 * cfl_max_ / cfl_update_ >= tau_0,
                ExcMessage("failed to recover from CFL violation"));

    if (tau_2 * cfl_max_ / cfl_update_ < tau_1) {
      /* Restart and force smaller time step: */
#ifdef DEBUG_OUTPUT
      std::cout << "        insufficient step size, restart" << std::endl;
#endif
      tau_0 = tau_2;
      U.swap(temp_ssp_);
      ++n_restarts_;
      goto restart_ssph2_step;
    }

    U.sadd(Number(1. / 2.), Number(1. / 2.), temp_ssp_);

    return tau_1;
  }


  template <int dim, typename Number>
  Number EulerModule<dim, Number>::ssprk3_step(vector_type &U,
                                               Number t,
                                               Number tau_0 /*= 0*/)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::ssprk3_step()" << std::endl;
#endif

  restart_ssprk3_step:
    /* This also copies ghost elements: */
    temp_ssp_ = U;

    /* Step 1: U1 = U_old + tau * L(U_old) */
    Number tau_1 = euler_step(U, t, tau_0);

    AssertThrow(tau_1 * cfl_max_ / cfl_update_ >= tau_0,
                ExcMessage("failed to recover from CFL violation"));
    tau_1 = (tau_0 == 0. ? tau_1 : tau_0);

    /* Step 2: U2 = 3/4 U_old + 1/4 (U1 + tau L(U1)) */
    const Number tau_2 = euler_step(U, t, tau_1);

    AssertThrow(tau_2 * cfl_max_ / cfl_update_ >= tau_0,
                ExcMessage("failed to recover from CFL violation"));

    if (tau_2 * cfl_max_ / cfl_update_ < tau_1) {
      /* Restart and force smaller time step: */
#ifdef DEBUG_OUTPUT
      std::cout << "        insufficient step size, restart" << std::endl;
#endif
      tau_0 = tau_2;
      U.swap(temp_ssp_);
      ++n_restarts_;
      goto restart_ssprk3_step;
    }

    U.sadd(Number(1. / 4.), Number(3. / 4.), temp_ssp_);

    /* Step 3: U_new = 1/3 U_old + 2/3 (U2 + tau L(U2)) */
    const Number tau_3 = euler_step(U, t, tau_1);

    AssertThrow(tau_3 * cfl_max_ / cfl_update_ >= tau_0,
                ExcMessage("failed to recover from CFL violation"));

    if (tau_3 * cfl_max_ < tau_1 * cfl_update_) {
      /* Restart and force smaller time step: */
#ifdef DEBUG_OUTPUT
      std::cout << "        insufficient step size, restart" << std::endl;
#endif
      tau_0 = tau_3;
      U.swap(temp_ssp_);
      ++n_restarts_;
      goto restart_ssprk3_step;
    }

    U.sadd(Number(2. / 3.), Number(1. / 3.), temp_ssp_);

    return tau_1;
  }


  template <int dim, typename Number>
  Number
  EulerModule<dim, Number>::step(vector_type &U, Number t, Number tau /*= 0*/)
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::step()" << std::endl;
#endif

    switch (time_step_order_) {
    case 1:
      return euler_step(U, t, tau);
    case 2:
      return ssph2_step(U, t, tau);
    case 3:
      return ssprk3_step(U, t, tau);
    default:
      AssertThrow(false,
                  ExcMessage("The chosen order of the time stepping method "
                             "must be in the interval [1, 3]"));
      __builtin_trap();
    }

    __builtin_unreachable();
  }


} /* namespace ryujin */

#endif /* EULER_MODULE_TEMPLATE_H */
