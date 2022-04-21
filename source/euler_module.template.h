//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "euler_module.h"
#include "indicator.h"
#include "introspection.h"
#include "openmp.h"
#include "riemann_solver.h"
#include "scope.h"
#include "simd.h"

#include <atomic>

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
      , id_violation_strategy_(IDViolationStrategy::warn)
      , mpi_communicator_(mpi_communicator)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , problem_description_(&problem_description)
      , initial_values_(&initial_values)
      , cfl_(0.2)
      , n_restarts_(0)
      , n_warnings_(0)
  {
    limiter_iter_ = 2;
    add_parameter(
        "limiter iterations", limiter_iter_, "Number of limiter iterations");

    if constexpr (std::is_same<Number, double>::value)
      limiter_newton_tolerance_ = 1.e-10;
    else
      limiter_newton_tolerance_ = 1.e-4;
    add_parameter("limiter newton tolerance",
                  limiter_newton_tolerance_,
                  "Tolerance for the quadratic newton stopping criterion");

    limiter_newton_max_iter_ = 2;
    add_parameter("limiter newton max iterations",
                  limiter_newton_max_iter_,
                  "Maximal number of quadratic newton iterations performed "
                  "during limiting");

    cfl_with_boundary_dofs_ = false;
    add_parameter("cfl with boundary dofs",
                  cfl_with_boundary_dofs_,
                  "Use also the local wave-speed estimate d_ij of boundary "
                  "dofs when computing the maximal admissible step size");
  }


  template <int dim, typename Number>
  void EulerModule<dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::prepare()" << std::endl;
#endif

    /* Initialize vectors: */

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();

    alpha_.reinit(scalar_partitioner);
    second_variations_.reinit(scalar_partitioner);
    specific_entropies_.reinit(scalar_partitioner);
    evc_entropies_.reinit(scalar_partitioner);

    bounds_.reinit_with_scalar_partitioner(scalar_partitioner);

    const auto &vector_partitioner = offline_data_->vector_partitioner();
    r_.reinit(vector_partitioner);

    /* Initialize matrices: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    dij_matrix_.reinit(sparsity_simd);
    lij_matrix_.reinit(sparsity_simd);
    lij_matrix_next_.reinit(sparsity_simd);
    pij_matrix_.reinit(sparsity_simd);
  }


  namespace
  {
    /**
     * Internally used: returns true if all indices are on the lower
     * triangular.
     */
    template <typename T>
    bool all_below_diagonal(unsigned int i, const unsigned int *js)
    {
      if constexpr (std::is_same<T, typename get_value_type<T>::type>::value) {
        /* Non-vectorized sequential access. */
        const auto j = *js;
        return j < i;

      } else {
        /* Vectorized fast access. index must be divisible by simd_length */

        constexpr auto simd_length = T::size();

        bool all_below_diagonal = true;
        for (unsigned int k = 0; k < simd_length; ++k)
          if (js[k] >= i + k) {
            all_below_diagonal = false;
            break;
          }
        return all_below_diagonal;
      }
    }

    /**
     * Return the stride size:
     */

    template <typename Number>
    unsigned int get_stride_size = 1;

    template <typename Number>
    unsigned int get_stride_size<VectorizedArray<Number>> =
        VectorizedArray<Number>::size();

  } // namespace


  template <int dim, typename Number>
  template <int stages>
  Number EulerModule<dim, Number>::step(
      const vector_type &old_U,
      std::array<std::reference_wrapper<const vector_type>, stages> stage_U,
      const std::array<Number, stages> stage_weights,
      vector_type &new_U,
      Number tau /*= 0.*/) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::single_step()" << std::endl;
#endif

    CALLGRIND_START_INSTRUMENTATION

    using VA = VectorizedArray<Number>;

    /* Index ranges for the iteration over the sparsity pattern : */

    constexpr auto simd_length = VA::size();
    const unsigned int n_export_indices = offline_data_->n_export_indices();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    /* References to precomputed matrices and the stencil: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &lumped_mass_matrix_inverse =
        offline_data_->lumped_mass_matrix_inverse();
    const auto &mass_matrix = offline_data_->mass_matrix();
    const auto &betaij_matrix = offline_data_->betaij_matrix();
    const auto &cij_matrix = offline_data_->cij_matrix();

    const auto &boundary_map = offline_data_->boundary_map();
    const auto &coupling_boundary_pairs =
        offline_data_->coupling_boundary_pairs();

    const Number measure_of_omega_inverse =
        Number(1.) / offline_data_->measure_of_omega();

    /* A monotonically increasing "channel" variable for mpi_tags: */
    unsigned int channel = 10;

    /* A boolean signalling that a restart is necessary: */
    std::atomic<bool> restart_needed = false;

    /*
     * Step 0: Precompute f(U) and the entropies of U
     */
    {
      Scope scope(computing_timer_, "time step [E] 0 - compute entropies");

      SynchronizationDispatch synchronization_dispatch([&]() {
        specific_entropies_.update_ghost_values_start(channel++);
        evc_entropies_.update_ghost_values_start(channel++);
      });

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_0");

      /* Stored thread locally: */
      bool thread_ready = false;

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          synchronization_dispatch.check(
              thread_ready, i >= n_export_indices && i < n_internal);

          const auto U_i = old_U.template get_tensor<T>(i);

          const auto s_i = problem_description_->specific_entropy(U_i);
          store_value<T>(specific_entropies_, s_i, i);

          const auto evc_entropy = problem_description_->harten_entropy(U_i);
          store_value<T>(evc_entropies_, evc_entropy, i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP("time_step_0");
      RYUJIN_PARALLEL_REGION_END
    }

    {
#if defined(SPLIT_SYNCHRONIZATION_TIMERS) || defined(DEBUG_OUTPUT)
      Scope scope(computing_timer_, "time step [E] 0 - synchronization");
#else
      Scope scope(computing_timer_, "time step [E] 0 - compute entropies");
#endif

      specific_entropies_.update_ghost_values_finish();
      evc_entropies_.update_ghost_values_finish();
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
      Scope scope(computing_timer_,
                  "time step [E] 1 - compute d_ij, and alpha_i");

      SynchronizationDispatch synchronization_dispatch([&]() {
        alpha_.update_ghost_values_start(channel++);
        second_variations_.update_ghost_values_start(channel++);
      });

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_1");

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        /* Stored thread locally: */
        RiemannSolver<dim, T> riemann_solver(*problem_description_);
        Indicator<dim, T> indicator(*problem_description_);
        bool thread_ready = false;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          synchronization_dispatch.check(
              thread_ready, i >= n_export_indices && i < n_internal);

          const auto U_i = old_U.template get_tensor<T>(i);
          const auto entropy_i = load_value<T>(evc_entropies_, i);
          const auto entropy_derivative_i =
              problem_description_->harten_entropy_derivative(U_i);

          indicator.reset(U_i, entropy_i, entropy_derivative_i);

          const auto mass = load_value<T>(lumped_mass_matrix, i);
          const auto hd_i = mass * measure_of_omega_inverse;

          /* Skip diagonal. */
          const unsigned int *js = sparsity_simd.columns(i) + stride_size;
          for (unsigned int col_idx = 1; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = old_U.template get_tensor<T>(js);
            const auto entropy_j = load_value<T>(evc_entropies_, js);

            const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);
            const auto beta_ij =
                betaij_matrix.template get_entry<T>(i, col_idx);
            indicator.add(U_j, c_ij, beta_ij, entropy_j);

            /* Only iterate over the upper triangular portion of d_ij */
            if (all_below_diagonal<T>(i, js))
              continue;

            const auto norm = c_ij.norm();
            const auto n_ij = c_ij / norm;

            const auto [lambda_max, p_star] =
                riemann_solver.compute(U_i, U_j, n_ij);

            const auto d = norm * lambda_max;

            dij_matrix_.write_entry(d, i, col_idx, true);
          }

          store_value<T>(alpha_, indicator.alpha(hd_i), i);
          store_value<T>(second_variations_, indicator.second_variations(), i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP("time_step_1");
      RYUJIN_PARALLEL_REGION_END
    }

    /*
     * Step 2: Compute diagonal of d_ij, and maximal time-step size.
     */

    std::atomic<Number> tau_max{std::numeric_limits<Number>::infinity()};

    {
      Scope scope(
          computing_timer_,
          "time step [E] 2 - compute bdry d_ij, diag d_ii, and tau_max");

      /* Parallel region */
      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_2");

      /* Complete d_ij at boundary: */

      RiemannSolver<dim, Number> riemann_solver(*problem_description_);

      RYUJIN_OMP_FOR /* with barrier */
      for (std::size_t k = 0; k < coupling_boundary_pairs.size(); ++k) {
        const auto &entry = coupling_boundary_pairs[k];
        const auto &[i, col_idx, j] = entry;
        const auto U_i = old_U.get_tensor(i);
        const auto U_j = old_U.get_tensor(j);
        const auto c_ji = cij_matrix.get_transposed_tensor(i, col_idx);
        Assert(c_ji.norm() > 1.e-12, ExcInternalError());
        const auto norm = c_ji.norm();
        const auto n_ji = c_ji / norm;
        auto [lambda_max_2, p_star_2] = riemann_solver.compute(U_j, U_i, n_ji);

        auto d = dij_matrix_.get_entry(i, col_idx);
        d = std::max(d, norm * lambda_max_2);
        dij_matrix_.write_entry(d, i, col_idx);
      }

      /* Symmetrize d_ij: */

      RYUJIN_OMP_FOR /* with barrier */
      for (unsigned int i = 0; i < n_owned; ++i) {

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          continue;

        Number d_sum = Number(0.);

        /* skip diagonal: */
        const unsigned int *js = sparsity_simd.columns(i);
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
        const Number tau = cfl_ * mass / (Number(-2.) * d_sum);

        if (boundary_map.count(i) == 0 || cfl_with_boundary_dofs_) {
          Number current_tau_max = tau_max.load();
          while (current_tau_max > tau &&
                 !tau_max.compare_exchange_weak(current_tau_max, tau))
            ;
        }
      }

      LIKWID_MARKER_STOP("time_step_2");
      RYUJIN_PARALLEL_REGION_END
    }

    {
      Scope scope(computing_timer_,
                  "time step [E] 2 - synchronization barrier");

      alpha_.update_ghost_values_finish();
      second_variations_.update_ghost_values_finish();

      /* MPI Barrier: */
      tau_max.store(Utilities::MPI::min(tau_max.load(), mpi_communicator_));

      AssertThrow(
          !std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
          ExcMessage(
              "I'm sorry, Dave. I'm afraid I can't do that.\nWe crashed."));

      tau = (tau == Number(0.) ? tau_max.load() : tau);

#ifdef DEBUG_OUTPUT
      std::cout << "        computed tau_max = " << tau_max << std::endl;
      std::cout << "        perform time-step with tau = " << tau << std::endl;
#endif
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
        if (limiter_iter_ != 0)
          r_.update_ghost_values_start(channel++);
      });

      const Number weight =
          -std::accumulate(stage_weights.begin(), stage_weights.end(), -1.);

      /* Parallel region */
      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_3");

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        /* Stored thread locally: */
        Limiter<dim, T> limiter(*problem_description_);
        bool thread_ready = false;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          synchronization_dispatch.check(
              thread_ready, i >= n_export_indices && i < n_internal);

          const auto U_i = old_U.template get_tensor<T>(i);
          const auto f_i = problem_description_->f(U_i);

          std::array<ProblemDescription::flux_type<dim, T>, stages> f_iHs;
          for (int s = 0; s < stages; ++s) {
            const auto temp = stage_U[s].get().template get_tensor<T>(i);
            f_iHs[s] = problem_description_->f(temp);
          }

          auto U_i_new = U_i;
          const auto alpha_i = load_value<T>(alpha_, i);
          const auto specific_entropy_i = load_value<T>(specific_entropies_, i);
          const auto variations_i = load_value<T>(second_variations_, i);

          const auto m_i = load_value<T>(lumped_mass_matrix, i);
          const auto m_i_inv = load_value<T>(lumped_mass_matrix_inverse, i);

          ProblemDescription::state_type<dim, T> r_i;

          /* Clear bounds: */
          limiter.reset(specific_entropy_i, variations_i);

          const unsigned int *js = sparsity_simd.columns(i);
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = old_U.template get_tensor<T>(js);
            const auto f_j = problem_description_->f(U_j);

            std::array<ProblemDescription::state_type<dim, T>, stages> U_jHs;
            std::array<ProblemDescription::flux_type<dim, T>, stages> f_jHs;
            for (int s = 0; s < stages; ++s) {
              U_jHs[s] = stage_U[s].get().template get_tensor<T>(js);
              f_jHs[s] = problem_description_->f(U_jHs[s]);
            }

            const auto alpha_j = load_value<T>(alpha_, js);
            const auto variations_j = load_value<T>(second_variations_, js);

            const auto d_ij = dij_matrix_.template get_entry<T>(i, col_idx);
            const auto d_ijH = d_ij * (alpha_i + alpha_j) * Number(.5);

            dealii::Tensor<1, problem_dimension, T> U_ij_bar;
            const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);
            const auto d_ij_inv = Number(1.) / d_ij;

            for (unsigned int k = 0; k < problem_dimension; ++k) {
              const auto temp = (f_j[k] - f_i[k]) * c_ij;
              r_i[k] += weight * (-temp) + d_ijH * (U_j[k] - U_i[k]);
              U_ij_bar[k] = Number(0.5) * (U_i[k] + U_j[k] - temp * d_ij_inv);
            }

            for (int s = 0; s < stages; ++s) {
              for (unsigned int k = 0; k < problem_dimension; ++k) {
                const auto temp = (f_jHs[s][k] - f_iHs[s][k]) * c_ij;
                r_i[k] += stage_weights[s] * (-temp);
              }
            }

            U_i_new += tau * m_i_inv * Number(2.) * d_ij * U_ij_bar;

            const auto beta_ij =
                betaij_matrix.template get_entry<T>(i, col_idx);
            const auto specific_entropy_j =
                load_value<T>(specific_entropies_, js);

            limiter.accumulate(U_i,
                               U_j,
                               U_ij_bar,
                               beta_ij,
                               specific_entropy_j,
                               variations_j,
                               /* is diagonal */ col_idx == 0);
          }

          new_U.template write_tensor<T>(U_i_new, i);
          r_.template write_tensor<T>(r_i, i);

          const auto hd_i = m_i * measure_of_omega_inverse;
          limiter.apply_relaxation(hd_i);
          bounds_.template write_tensor<T>(limiter.bounds(), i);
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP("time_step_3");
      RYUJIN_PARALLEL_REGION_END
    }

    {
#if defined(SPLIT_SYNCHRONIZATION_TIMERS) || defined(DEBUG_OUTPUT)
      Scope scope(computing_timer_, "time step [E] 3 - synchronization");
#else
      Scope scope(computing_timer_,
                  "time step [E] 3 - l.-o. update, bounds, and r_i");
#endif

      if (limiter_iter_ != 0)
        r_.update_ghost_values_finish();
    }

    /*
     * Step 4: Compute P_ij, and l_ij (first round):
     *
     *    P_ij = tau / m_i / lambda ( (d_ij^H - d_ij^L) (U_i - U_j) +
     *                                (b_ij R_j - b_ji R_i) )
     */

    if (limiter_iter_ != 0) {
      Scope scope(computing_timer_, "time step [E] 4 - compute p_ij, and l_ij");

      SynchronizationDispatch synchronization_dispatch(
          [&]() { lij_matrix_.update_ghost_rows_start(channel++); });

      const Number weight =
          -std::accumulate(stage_weights.begin(), stage_weights.end(), -1.);

      RYUJIN_PARALLEL_REGION_BEGIN
      LIKWID_MARKER_START("time_step_4");

      auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
        using T = decltype(sentinel);
        unsigned int stride_size = get_stride_size<T>;

        /* Stored thread locally: */
        bool thread_ready = false;

        RYUJIN_OMP_FOR_NOWAIT
        for (unsigned int i = left; i < right; i += stride_size) {

          /* Skip constrained degrees of freedom: */
          const unsigned int row_length = sparsity_simd.row_length(i);
          if (row_length == 1)
            continue;

          synchronization_dispatch.check(
              thread_ready, i >= n_export_indices && i < n_internal);

          const auto bounds =
              bounds_.template get_tensor<T, std::array<T, 3>>(i);

          const auto m_i_inv = load_value<T>(lumped_mass_matrix_inverse, i);

          const auto U_i_new = new_U.template get_tensor<T>(i);
          const auto U_i = old_U.template get_tensor<T>(i);
          const auto f_i = problem_description_->f(U_i);

          std::array<ProblemDescription::flux_type<dim, T>, stages> f_iHs;
          for (int s = 0; s < stages; ++s) {
            const auto temp = stage_U[s].get().template get_tensor<T>(i);
            f_iHs[s] = problem_description_->f(temp);
          }

          const auto r_i = r_.template get_tensor<T>(i);
          const auto alpha_i = load_value<T>(alpha_, i);
          const auto lambda_inv = Number(row_length - 1);
          const auto factor = tau * m_i_inv * lambda_inv;

          const unsigned int *js = sparsity_simd.columns(i);
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = old_U.template get_tensor<T>(js);

            std::array<ProblemDescription::state_type<dim, T>, stages> U_jHs;
            std::array<ProblemDescription::flux_type<dim, T>, stages> f_jHs;
            for (int s = 0; s < stages; ++s) {
              U_jHs[s] = stage_U[s].get().template get_tensor<T>(js);
              f_jHs[s] = problem_description_->f(U_jHs[s]);
            }

            const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);
            const auto r_j = r_.template get_tensor<T>(js);

            const auto alpha_j = load_value<T>(alpha_, js);
            const auto m_j_inv = load_value<T>(lumped_mass_matrix_inverse, js);

            const auto d_ij = dij_matrix_.template get_entry<T>(i, col_idx);
            const auto d_ijH = d_ij * (alpha_i + alpha_j) * Number(.5);

            const auto m_ij = mass_matrix.template get_entry<T>(i, col_idx);
            const auto b_ij = (col_idx == 0 ? T(1.) : T(0.)) - m_ij * m_j_inv;
            const auto b_ji = (col_idx == 0 ? T(1.) : T(0.)) - m_ij * m_i_inv;

            /* Contributions from graph viscosity and mass matrix correction: */

            auto p_ij = (d_ijH - d_ij) * (U_j - U_i) + b_ij * r_j - b_ji * r_i;

            if constexpr (stages != 0) {
              /* Flux contributions: */

              const auto f_j = problem_description_->f(U_j);

              for (unsigned int k = 0; k < problem_dimension; ++k) {
                const auto temp = (f_j[k] + f_i[k]) * c_ij;
                p_ij[k] += (weight - Number(1.)) * (-temp);
              }

              for (int s = 0; s < stages; ++s) {
                for (unsigned int k = 0; k < problem_dimension; ++k) {
                  const auto temp = (f_jHs[s][k] + f_iHs[s][k]) * c_ij;
                  p_ij[k] += stage_weights[s] * (-temp);
                }
              }
            }

            p_ij *= factor;
            pij_matrix_.write_tensor(p_ij, i, col_idx, true);

            const auto &[l_ij, success] =
                Limiter<dim, T>::limit(*problem_description_,
                                       bounds,
                                       U_i_new,
                                       p_ij,
                                       limiter_newton_tolerance_,
                                       limiter_newton_max_iter_);
            lij_matrix_.template write_entry<T>(l_ij, i, col_idx, true);

            /* Unsuccessful with current CFL, force a restart. */
            if (!success)
              restart_needed = true;
          }
        }
      };

      /* Parallel non-vectorized loop: */
      loop(Number(), n_internal, n_owned);
      /* Parallel vectorized SIMD loop: */
      loop(VA(), 0, n_internal);

      LIKWID_MARKER_STOP("time_step_4");
      RYUJIN_PARALLEL_REGION_END
    }

    if (limiter_iter_ != 0) {
#if defined(SPLIT_SYNCHRONIZATION_TIMERS) || defined(DEBUG_OUTPUT)
      Scope scope(computing_timer_, "time step [E] 4 - synchronization");
#else
      Scope scope(computing_timer_, "time step [E] 4 - compute p_ij, and l_ij");
#endif

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
          if (!last_round)
            lij_matrix_next_.update_ghost_rows_start(channel++);
        });

        RYUJIN_PARALLEL_REGION_BEGIN
        LIKWID_MARKER_START(("time_step_" + step_no).c_str());

        auto loop = [&](auto sentinel, unsigned int left, unsigned int right) {
          using T = decltype(sentinel);
          unsigned int stride_size = get_stride_size<T>;

          /* Stored thread locally: */
          AlignedVector<T> lij_row;
          bool thread_ready = false;

          RYUJIN_OMP_FOR_NOWAIT
          for (unsigned int i = left; i < right; i += stride_size) {

            /* Skip constrained degrees of freedom: */
            const unsigned int row_length = sparsity_simd.row_length(i);
            if (row_length == 1)
              continue;

            synchronization_dispatch.check(
                thread_ready, i >= n_export_indices && i < n_internal);

            auto U_i_new = new_U.template get_tensor<T>(i);

            const Number lambda = Number(1.) / Number(row_length - 1);
            lij_row.resize_fast(row_length);

            for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {

              const auto l_ij = std::min(
                  lij_matrix_.template get_entry<T>(i, col_idx),
                  lij_matrix_.template get_transposed_entry<T>(i, col_idx));

              auto p_ij = pij_matrix_.template get_tensor<T>(i, col_idx);

              U_i_new += l_ij * lambda * p_ij;

              if (!last_round)
                lij_row[col_idx] = l_ij;
            }

#ifdef CHECK_BOUNDS
            {
              const auto rho_new = problem_description_->density(U_i_new);
              const auto e_new = problem_description_->internal_energy(U_i_new);
              const auto s_new =
                  problem_description_->specific_entropy(U_i_new);

              using namespace dealii;
              constexpr auto gt = dealii::SIMDComparison::greater_than;
              const auto test =
                  compare_and_apply_mask<gt>(rho_new, T(0.), T(0.), T(-1.)) + //
                  compare_and_apply_mask<gt>(e_new, T(0.), T(0.), T(-1.)) +   //
                  compare_and_apply_mask<gt>(s_new, T(0.), T(0.), T(-1.));
              if (!(test == T(0.))) {
#ifdef DEBUG_OUTPUT
                std::cout << std::fixed << std::setprecision(16);
                std::cout << "Bounds violation: Negative rho, e, s detected!"
                          << std::endl;
                std::cout << "rho: !!! " << rho_new << std::endl;
                std::cout << "int: !!! " << e_new << std::endl;
                std::cout << "ent: !!! " << s_new << std::endl << std::endl;
#endif
                restart_needed = true;
              }
            }
#endif

            new_U.template write_tensor<T>(U_i_new, i);

            /* Skip computating l_ij and updating p_ij in the last round */
            if (last_round)
              continue;

            const auto bounds =
                bounds_.template get_tensor<T, std::array<T, 3>>(i);
            for (unsigned int col_idx = 0; col_idx < row_length; ++col_idx) {

              const auto old_l_ij = lij_row[col_idx];

              const auto new_p_ij =
                  (T(1.) - old_l_ij) *
                  pij_matrix_.template get_tensor<T>(i, col_idx);

              const auto &[new_l_ij, success] =
                  Limiter<dim, T>::limit(*problem_description_,
                                         bounds,
                                         U_i_new,
                                         new_p_ij,
                                         limiter_newton_tolerance_,
                                         limiter_newton_max_iter_);

              /* Unsuccessful with current CFL, force a restart. */
              if (!success)
                restart_needed = true;

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
                const auto entry = (T(1.) - old_l_ij) * new_l_ij;
                lij_matrix_next_.write_entry(entry, i, col_idx, true);
              } else {
                /*
                 * @todo: This is expensive. If we ever end up using more
                 * than two limiter passes we should implement this by
                 * storing a scalar factor instead of writing back into p_ij.
                 */
                lij_matrix_next_.write_entry(new_l_ij, i, col_idx, true);
                pij_matrix_.write_tensor(new_p_ij, i, col_idx);
              }
            }
          }
        };

        /* Parallel non-vectorized loop: */
        loop(Number(), n_internal, n_owned);
        /* Parallel vectorized SIMD loop: */
        loop(VA(), 0, n_internal);

        LIKWID_MARKER_STOP(("time_step_" + step_no).c_str());
        RYUJIN_PARALLEL_REGION_END
      }

      {
#if defined(SPLIT_SYNCHRONIZATION_TIMERS) || defined(DEBUG_OUTPUT)
        Scope scope(computing_timer_,
                    "time step [E] " + step_no + " - synchronization");
#else
        Scope scope(computing_timer_,
                    "time step [E] " + step_no + " - " +
                        "symmetrize l_ij, h.-o. update" + additional_step);
#endif

        if (!last_round) {
          lij_matrix_next_.update_ghost_rows_finish();
          std::swap(lij_matrix_, lij_matrix_next_);
        }
      }
    } /* limiter_iter_ */

    CALLGRIND_STOP_INSTRUMENTATION

    /* Do we have to restart? */

    restart_needed.store(
        Utilities::MPI::logical_or(restart_needed.load(), mpi_communicator_));

    if (restart_needed) {
      switch (id_violation_strategy_) {
      case IDViolationStrategy::warn:
        n_warnings_++;
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0)
          std::cout << "[INFO] Euler module: Insufficient CFL: Invariant "
                       "domain violation detected"
                    << std::endl;
        break;
      case IDViolationStrategy::raise_exception:
        n_restarts_++;
        throw Restart();
      }
    }

    /* Update the result and return tau_max: */
    return tau_max;
  }


  template <int dim, typename Number>
  void EulerModule<dim, Number>::apply_boundary_conditions(vector_type &U,
                                                           Number t) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "EulerModule<dim, Number>::apply_boundary_conditions()"
              << std::endl;
#endif

    Scope scope(computing_timer_,
                "time step [E] " + std::to_string(5 + limiter_iter_) +
                    " - apply boundary conditions");

    const auto &boundary_map = offline_data_->boundary_map();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    for (auto entry : boundary_map) {
      const auto i = entry.first;
      if (i >= n_owned)
        continue;

      const auto &[normal, normal_mass, boundary_mass, id, position] =
          entry.second;

      if (id == Boundary::do_nothing)
        continue;

      auto U_i = U.get_tensor(i);

      if (id == Boundary::slip) {
        /* Remove the normal component of the momentum: */
        auto m = problem_description_->momentum(U_i);
        m -= 1. * (m * normal) * normal;
        for (unsigned int k = 0; k < dim; ++k)
          U_i[k + 1] = m[k];

      } else if (id == Boundary::no_slip) {
        /* Enforce no-slip conditions: */
        for (unsigned int k = 0; k < dim; ++k)
          U_i[k + 1] = Number(0.);

      } else if (id == Boundary::dirichlet) {
        /* On Dirichlet boundaries enforce initial conditions: */
        U_i = initial_values_->initial_state(position, t);

      } else if (id == Boundary::dynamic) {
        /*
         * On dynamic boundary conditions, we distinguish four cases:
         *
         *  - supersonic inflow: prescribe full state
         *  - subsonic inflow:
         *      decompose into Riemann invariants and leave R_2
         *      characteristic untouched.
         *  - supersonic outflow: do nothing
         *  - subsonic outflow:
         *      decompose into Riemann invariants and prescribe incoming
         *      R_1 characteristic.
         */
        const auto m = problem_description_->momentum(U_i);
        const auto rho = problem_description_->density(U_i);
        const auto a = problem_description_->speed_of_sound(U_i);
        const auto vn = m * normal / rho;

        /* Supersonic inflow: */
        if (vn < -a) {
          U_i = initial_values_->initial_state(position, t);
        }

        /* Subsonic inflow: */
        if (vn >= -a && vn <= 0.) {
          const auto U_i_dirichlet =
              initial_values_->initial_state(position, t);
          U_i = problem_description_->prescribe_riemann_characteristic<2>(
              U_i_dirichlet, U_i, normal);
        }

        /* Subsonic outflow: */
        if (vn > 0. && vn <= a) {
          const auto U_i_dirichlet =
              initial_values_->initial_state(position, t);
          U_i = problem_description_->prescribe_riemann_characteristic<1>(
              U_i, U_i_dirichlet, normal);
        }

        /* Supersonic outflow: */
        if (vn > a) {
          continue;
        }
      }

      U.write_tensor(U_i, i);
    }

    U.update_ghost_values();
  }

} /* namespace ryujin */
