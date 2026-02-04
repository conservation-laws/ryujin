//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "hyperbolic_module.h"
#include "instrumentation.h"
#include "loop.h"
#include "mpi_ensemble.h"
#include "scope.h"
#include "simd.h"

#include "sparse_matrix_simd.template.h"

#include <atomic>

namespace ryujin
{
  namespace ShallowWater
  {
    struct Description;
  }

  using namespace dealii;

  template <typename Description, int dim, typename Number>
  HyperbolicModule<Description, dim, Number>::HyperbolicModule(
      const MPIEnsemble &mpi_ensemble,
      std::map<std::string, dealii::Timer> &computing_timer,
      const OfflineData<dim, Number> &offline_data,
      const HyperbolicSystem &hyperbolic_system,
      const InitialValues<Description, dim, Number> &initial_values,
      const std::string &subsection /*= "HyperbolicModule"*/)
      : ParameterAcceptor(subsection)
      , indicator_parameters_(subsection + "/indicator")
      , limiter_parameters_(subsection + "/limiter")
      , riemann_solver_parameters_(subsection + "/riemann solver")
      , mpi_ensemble_(mpi_ensemble)
      , computing_timer_(computing_timer)
      , offline_data_(&offline_data)
      , hyperbolic_system_(&hyperbolic_system)
      , initial_values_(&initial_values)
      , cfl_(0.2)
      , acceptable_tau_max_ratio_(1.e6)
      , id_violation_strategy_(IDViolationStrategy::warn)
      , n_restarts_(0)
      , n_corrections_(0)
      , n_warnings_(0)
  {
  }


  template <typename Description, int dim, typename Number>
  void HyperbolicModule<Description, dim, Number>::prepare()
  {
#ifdef DEBUG_OUTPUT
    std::cout << "HyperbolicModule<Description, dim, Number>::prepare()"
              << std::endl;
#endif

    AssertThrow(limiter_parameters_.iterations() <= 2,
                dealii::ExcMessage(
                    "The number of limiter iterations must be between [0,2]"));

    /* Initialize vectors: */

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();
    alpha_.reinit(scalar_partitioner);
    bounds_.reinit_with_scalar_partitioner(scalar_partitioner);

    r_.reinit(offline_data_->hyperbolic_vector_partitioner());
    using View =
        typename Description::template HyperbolicSystemView<dim, Number>;

    /* Initialize matrices: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    dij_matrix_.reinit(sparsity_simd);
    lij_matrix_.reinit(sparsity_simd);
    lij_matrix_next_.reinit(sparsity_simd);
    pij_matrix_.reinit(sparsity_simd);

    /* Set up initial precomputed vector: */

    initial_precomputed_ =
        initial_values_->interpolate_initial_precomputed_vector();
  }


  /*
   * -------------------------------------------------------------------------
   * Step 0: Reinitialize vector
   * -------------------------------------------------------------------------
   */


  /**
   * Helper function that (re)initializes the state and the precomputed
   * component of a StateVector to proper sizes.
   *
   * @note This method does neither initialize nor resize the parabolic
   * state vector component. The ParabolicModule itself has to ensure
   * proper setup during prepare_state_vector() and solve().
   */
  template <typename Description, int dim, typename Number>
  void HyperbolicModule<Description, dim, Number>::reinit_state_vector(
      StateVector &state_vector) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "HyperbolicModule<dim, Number>::reinit_state_vector()"
              << std::endl;
#endif

    auto &[U, precomputed, V] = state_vector;
    U.reinit(offline_data_->hyperbolic_vector_partitioner());
    precomputed.reinit(offline_data_->precomputed_vector_partitioner());

#ifdef DEBUG
    /* Poison all vectors: */
    using state_type = typename View::state_type;

    constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();

    const unsigned int n_owned = offline_data_->n_locally_owned();
    for (unsigned int i = 0; i < n_owned; ++i) {
      U.write_tensor(state_type{} * nan, i);
      precomputed.write_tensor(
          dealii::Tensor<1, n_precomputed_values, Number>() * nan, i);
    }
#endif
  }


  /*
   * -------------------------------------------------------------------------
   * Step 1: Apply boundary conditions and precompute values
   * -------------------------------------------------------------------------
   */


  template <typename Description, int dim, typename Number>
  void HyperbolicModule<Description, dim, Number>::prepare_state_vector(
      StateVector &state_vector, Number t) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "HyperbolicModule<Description, dim, "
                 "Number>::prepare_state_vector()"
              << std::endl;
#endif

    auto &U = std::get<0>(state_vector);

    const auto &boundary_map = offline_data_->boundary_map();
    using VA = VectorizedArray<Number>;

    Scope scope(computing_timer_,
                "time step [H] 1 - update boundary values, precompute values");

    LIKWID_MARKER_START("time_step_1");

    /* FIXME: not thread parallel... */
    for (const auto &entry : boundary_map) {
      const auto &[i, normal, normal_mass, boundary_mass, id, position] = entry;

      /*
       * Relay the task of applying appropriate boundary conditions to the
       * Problem Description.
       */

      if (id == Boundary::do_nothing)
        continue;

      auto U_i = U.get_tensor(i);

      /* Use a lambda to avoid computing unnecessary state values */
      auto get_dirichlet_data = [position = position, t = t, this]() {
        return initial_values_->initial_state(position, t);
      };

      const auto view = hyperbolic_system_->template view<dim, Number>();
      U_i = view.apply_boundary_conditions(id, U_i, normal, get_dirichlet_data);
      U.write_tensor(U_i, i);
    }

    LIKWID_MARKER_STOP("time_step_1");

    U.update_ghost_values();

    /*
     * Compute and populate precomputed values.
     */

    LIKWID_MARKER_START("time_step_1");

    auto &precomputed = std::get<1>(state_vector);
    hyperbolic_system_->fill_precomputed_values(*offline_data_, state_vector);

    LIKWID_MARKER_STOP("time_step_1");

    precomputed.update_ghost_values();
  }


  /*
   * -------------------------------------------------------------------------
   * Step 2 - 7: Perform an explicit Euler step
   * -------------------------------------------------------------------------
   */


  namespace
  {
    /**
     * Internally used: returns true if all indices are on the lower
     * triangular part of the matrix.
     */
    template <typename T>
    bool all_below_diagonal(unsigned int i, const unsigned int *js)
    {
      if constexpr (std::is_same_v<T, typename get_value_type<T>::type>) {
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
  } // namespace


  template <typename Description, int dim, typename Number>
  template <int stages>
  Number HyperbolicModule<Description, dim, Number>::step(
      const StateVector &old_state_vector,
      std::array<std::reference_wrapper<const StateVector>, stages>
          stage_state_vectors,
      const std::array<Number, stages> stage_weights,
      StateVector &new_state_vector,
      Number tau /*= 0.*/,
      Number tau_max /*= std::numeric_limits<Number>::max()*/) const
  {
#ifdef DEBUG_OUTPUT
    std::cout << "HyperbolicModule<Description, dim, Number>::step()"
              << std::endl;
#endif

    auto &old_U = std::get<0>(old_state_vector);
    auto &old_precomputed = std::get<1>(old_state_vector);
    auto &new_U = std::get<0>(new_state_vector);

    CALLGRIND_START_INSTRUMENTATION;

    /*
     * Workaround: A constexpr boolean storing the fact whether we
     * instantiate the HyperbolicModule for the shallow water equations.
     *
     * Rationale: Currently, the shallow water equations is the only
     * hyperbolic system for which we have to (a) form equilibrated states
     * for the low-order update, and (b) apply an affine shift for
     * computing limiter bounds. It's not so easy to come up with a
     * meaningful abstraction layer for this (in particular because we only
     * have one PDE). Thus, for the time being we simply special case a
     * small amount of code in this routine.
     *
     * FIXME: Refactor into a proper abstraction layer / interface.
     */
    constexpr bool shallow_water =
        std::is_same_v<Description, ShallowWater::Description>;

    using VA = VectorizedArray<Number>;

    /* Index ranges for the iteration over the sparsity pattern : */

    constexpr auto simd_length = VA::size();
    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    /* References to precomputed matrices and the stencil: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();

    const auto &mass_matrix = offline_data_->mass_matrix();
    const auto &mass_matrix_inverse = offline_data_->mass_matrix_inverse();
    const auto &lumped_mass_matrix = offline_data_->lumped_mass_matrix();
    const auto &lumped_mass_matrix_inverse =
        offline_data_->lumped_mass_matrix_inverse();

    const auto &cij_matrix = offline_data_->cij_matrix();
    const auto &incidence_matrix = offline_data_->incidence_matrix();

    const auto &coupling_boundary_pairs =
        offline_data_->coupling_boundary_pairs();

    const Number measure_of_omega_inverse =
        Number(1.) / offline_data_->measure_of_omega();

    /*
     * Lambdas for creating the computing timer and loop strings:
     */

    int step_no = 1;

    const auto scoped_name = [&step_no](const auto &name,
                                        const bool advance = true) {
      advance || step_no--;
      return "time step [H] " + std::to_string(++step_no) + " - " + name;
    };

    const auto loop_name = [&step_no]() {
      return "time_step_" + std::to_string(step_no);
    };

    /* A boolean signalling that a restart is necessary: */
    std::atomic<bool> restart_needed = false;

    /*
     * -------------------------------------------------------------------------
     * Step 2: Compute off-diagonal d_ij, and alpha_i
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
     * -------------------------------------------------------------------------
     */
    {
      Scope scope(computing_timer_, scoped_name("compute d_ij, and alpha_i"));

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        constexpr unsigned int stride_size = get_stride_size<T>;

        using RiemannSolver =
            typename Description::template RiemannSolver<dim, T>;
        RiemannSolver riemann_solver(
            *hyperbolic_system_, riemann_solver_parameters_, old_precomputed);

        using Indicator = typename Description::template Indicator<dim, T>;
        Indicator indicator(
            *hyperbolic_system_, indicator_parameters_, old_precomputed);

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          return;

        const auto U_i = old_U.template get_tensor<T>(i);

        indicator.reset(i, U_i);

        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto U_j = old_U.template get_tensor<T>(js);

          const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);

          indicator.accumulate(js, U_j, c_ij);

          /* Skip diagonal. */
          if (col_idx == 0)
            continue;

          /* Only iterate over the upper triangular portion of d_ij */
          if (all_below_diagonal<T>(i, js))
            continue;

          const auto norm = c_ij.norm();
          const auto n_ij = c_ij / norm;
          const auto lambda_max = riemann_solver.compute(U_i, U_j, i, js, n_ij);
          const auto d_ij = norm * lambda_max;

          dij_matrix_.write_entry(d_ij, i, col_idx, true);
        }

        const auto mass = get_entry<T>(lumped_mass_matrix, i);
        const auto hd_i = mass * measure_of_omega_inverse;
        write_entry<T>(alpha_, indicator.alpha(hd_i), i);
      };

      cpu_simd_loop<Number>(loop_name(), body, 0, n_internal, n_owned);

      alpha_.update_ghost_values();
    }

    /*
     * -------------------------------------------------------------------------
     * Step 3: Compute diagonal of d_ij, and maximal time-step size.
     * -------------------------------------------------------------------------
     */

    {
      Scope scope(computing_timer_,
                  scoped_name("compute bdry d_ij, diag d_ii, and tau_max"));

      /*
       * Complete d_ij at boundary:
       *
       * Here, for continuous finite elements the assumption c_ij = -c_ji
       * no longer holds true. This implies that d_ij != d_ji. We thus need
       * to compute the lower-triangular part of d_ij, where i and j are
       * boundary degrees of freedom as well.
       */

      /*
       * Note: we need this dance of iterating over an integer and then
       * accessing the element to make Apple's OpenMP implementation
       * happy.
       */
      const auto body_boundary = [&](const auto &, const unsigned int k) {
        const auto &[i, col_idx, j] = coupling_boundary_pairs[k];

        using RiemannSolver =
            typename Description::template RiemannSolver<dim, Number>;
        RiemannSolver riemann_solver(
            *hyperbolic_system_, riemann_solver_parameters_, old_precomputed);

        /*
         * Only work on index pairs "i < j" that point to the upper
         * triangular portion of the d_ij matrix. For all of these index
         * pairs we compute the corresponding d_ji entry and fix up the
         * d_ij entry (from step 2) by taking the maximum. Note that we
         * actually do not store anything in the d_ji entry itself because
         * we symmetrize the matrix later on anyway.
         */
        if (j < i)
          return;

        const auto U_i = old_U.get_tensor(i);
        const auto U_j = old_U.get_tensor(j);

        const auto c_ji = cij_matrix.get_transposed_tensor(i, col_idx);
        Assert(c_ji.norm() > 1.e-12, ExcInternalError());
        const auto norm_ji = c_ji.norm();
        const auto n_ji = c_ji / norm_ji;

        const auto d_ij = dij_matrix_.get_entry(i, col_idx);

        const auto lambda_max = riemann_solver.compute(U_j, U_i, j, &i, n_ji);
        const auto d_ji = norm_ji * lambda_max;

        dij_matrix_.write_entry(std::max(d_ij, d_ji), i, col_idx);
      };

      cpu_simd_loop<Number>(loop_name(),
                            body_boundary,
                            0,
                            /*no vectorization*/ 0,
                            coupling_boundary_pairs.size());

      std::atomic<Number> local_tau_max = tau_max;

      /* Symmetrize d_ij: */
      const auto body = [&](auto, unsigned int i) {
        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          return;

        Number d_sum = Number(0.);

        /* skip diagonal: */
        const unsigned int *js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const auto j =
              *(i < n_internal ? js + col_idx * simd_length : js + col_idx);

          // fill lower triangular part of dij_matrix missing from step 1
          if (j < i) {
            const auto d_ji = dij_matrix_.get_transposed_entry(i, col_idx);

#ifdef DEBUG_SYMMETRY_CHECK
            /* Verify that d_ji == std::max(d_ij, d_ji): */

            const auto U_i = old_U.get_tensor(i);
            const auto U_j = old_U.get_tensor(j);

            const auto c_ij = cij_matrix.get_tensor(i, col_idx);
            Assert(c_ij.norm() > 1.e-12, ExcInternalError());
            const auto norm_ij = c_ij.norm();
            const auto n_ij = c_ij / norm_ij;

            const auto lambda_max =
                riemann_solver.compute(U_i, U_j, i, &j, n_ij);
            const auto d_ij = norm_ij * lambda_max;

            Assert(d_ij <= d_ji + 1.0e-12,
                   dealii::ExcMessage("d_ij not symmetrized correctly on "
                                      "boundary degrees of freedom."));
#endif

            dij_matrix_.write_entry(d_ji, i, col_idx);
          }

          d_sum -= dij_matrix_.get_entry(i, col_idx);
        }

        /*
         * Make sure that we do not accidentally divide by zero. (Yes, this
         * can happen for some (admittedly, rather esoteric) scalar
         * conservation equations...).
         */
        d_sum =
            std::min(d_sum, Number(-1.e6) * std::numeric_limits<Number>::min());

        /* write diagonal element */
        dij_matrix_.write_entry(d_sum, i, 0);

        const Number mass = lumped_mass_matrix.local_element(i);
        const Number local_tau = cfl_ * mass / (Number(-2.) * d_sum);

        Number current_value = local_tau_max.load();
        while (current_value > local_tau &&
               !local_tau_max.compare_exchange_weak(current_value, local_tau))
          ;
      };

      cpu_simd_loop<Number>(
          loop_name(), body, 0, /*no vectorization*/ 0, n_owned);

      tau_max = local_tau_max.load();
    }

    {
      Scope scope(computing_timer_,
                  "time step [X] _ - synchronization barriers");

      /*
       * MPI Barrier: Synchronize the maximal time-step size. This has to
       * happen either over the global, or the local subrange communicator:
       */
      tau_max = Utilities::MPI::min(
          tau_max, mpi_ensemble_.synchronization_communicator());

      AssertThrow(
          !std::isnan(tau_max) && !std::isinf(tau_max) && tau_max > 0.,
          ExcMessage(
              "I'm sorry, Dave. I'm afraid I can't do that.\nWe crashed."));

      tau = (tau == Number(0.) ? tau_max : tau);

#ifdef DEBUG_OUTPUT
      std::cout << "        computed tau_max = " << tau_max
                << " (CFL = " << cfl_ << ")" << std::endl;
      std::cout << "        step with tau    = " << tau << std::endl;
#endif

      /* We need to signal a restart if the enforced tau is too wacky: */
      restart_needed = (tau > acceptable_tau_max_ratio_ * tau_max);

      /* Don't bother with computing the update step, signal a restart: */
      if (restart_needed &&
          id_violation_strategy_ == IDViolationStrategy::raise_exception) {
        n_restarts_++;
        /* Suggest a restart with tau_max: */
#ifdef DEBUG_OUTPUT
        std::cout << "        signalling restart (suggested_tau_max = "
                  << tau_max << ")" << std::endl;
#endif
        throw Restart{tau_max};
      }
    }

#ifdef DEBUG
    /*  Exchange d_ij so that we can check for symmetry: */
    dij_matrix_.update_ghost_rows();
#endif

    /*
     * -------------------------------------------------------------------------
     * Step 4: Low-order update, also compute limiter bounds, R_i
     * -------------------------------------------------------------------------
     */

    {
      Scope scope(computing_timer_,
                  scoped_name("l.-o. update, compute bounds, r_i, and p_ij"));

      const Number weight =
          -std::accumulate(stage_weights.begin(), stage_weights.end(), -1.);

      auto body = [&](auto sentinel,
                      auto have_discontinuous_ansatz,
                      const unsigned int i) {
        using T = decltype(sentinel);
        using View =
            typename Description::template HyperbolicSystemView<dim, T>;
        using Limiter = typename Description::template Limiter<dim, T>;
        using flux_contribution_type = typename View::flux_contribution_type;
        using state_type = typename View::state_type;

        constexpr unsigned int stride_size = get_stride_size<T>;

        const auto view = hyperbolic_system_->template view<dim, T>();

        Limiter limiter(
            *hyperbolic_system_, limiter_parameters_, old_precomputed);

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          return;

        const auto U_i = old_U.template get_tensor<T>(i);
        auto U_i_new = U_i;

        const auto alpha_i = get_entry<T>(alpha_, i);
        const auto m_i = get_entry<T>(lumped_mass_matrix, i);
        const auto m_i_inv = get_entry<T>(lumped_mass_matrix_inverse, i);

        const auto flux_i = view.flux_contribution(
            old_precomputed, initial_precomputed_, i, U_i);

        std::array<flux_contribution_type, stages> flux_iHs;
        [[maybe_unused]] state_type S_iH;

        for (int s = 0; s < stages; ++s) {
          const auto &[U_s, prec_s, V_s] = stage_state_vectors[s].get();

          const auto U_iHs = U_s.template get_tensor<T>(i);
          flux_iHs[s] =
              view.flux_contribution(prec_s, initial_precomputed_, i, U_iHs);

          if constexpr (View::have_source_terms) {
            S_iH += stage_weights[s] * view.nodal_source(prec_s, i, U_iHs, tau);
          }
        }

        [[maybe_unused]] state_type S_i;
        state_type F_iH;

        if constexpr (View::have_source_terms) {
          S_i = view.nodal_source(old_precomputed, i, U_i, tau);
          S_iH += weight * S_i;
          U_i_new += tau * /* m_i_inv * m_i */ S_i;
          F_iH += m_i * S_iH;
        }

        limiter.reset(i, U_i, flux_i);

        [[maybe_unused]] state_type affine_shift;

        /*
         * Workaround: For shallow water we need to accumulate an
         * additional contribution to the affine shift over the stencil
         * before we can compute limiter bounds.
         */

        const unsigned int *js = sparsity_simd.columns(i);
        if constexpr (shallow_water) {
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = old_U.template get_tensor<T>(js);
            const auto flux_j = view.flux_contribution(
                old_precomputed, initial_precomputed_, js, U_j);

            const auto d_ij = dij_matrix_.template get_entry<T>(i, col_idx);
            const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);

            const auto B_ij = view.affine_shift(flux_i, flux_j, c_ij, d_ij);
            affine_shift += B_ij;
          }

          affine_shift *= tau * m_i_inv;
        }

        if constexpr (View::have_source_terms) {
          affine_shift += tau * /* m_i_inv * m_i */ S_i;
        }

        js = sparsity_simd.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto U_j = old_U.template get_tensor<T>(js);

          const auto alpha_j = get_entry<T>(alpha_, js);

          const auto d_ij = dij_matrix_.template get_entry<T>(i, col_idx);
          auto factor = (alpha_i + alpha_j) * Number(.5);

          if constexpr (have_discontinuous_ansatz) {
            const auto incidence_ij =
                incidence_matrix.template get_entry<T>(i, col_idx);
            factor = std::max(factor, incidence_ij);
          }

          const auto d_ijH = d_ij * factor;

#ifdef DEBUG_SYMMETRY_CHECK
          /*
           * Verify that all local chunks of the d_ij matrix have been
           * computed consistently over all MPI ranks. For that we import
           * all ghost rows from neighboring MPI ranks and simply check
           * that the (local) values of d_ij and d_ji match.
           */
          const auto d_ji =
              dij_matrix_.template get_transposed_entry<T>(i, col_idx);
          Assert(std::max(std::abs(d_ij - d_ji), T(1.0e-12)) == T(1.0e-12),
                 dealii::ExcMessage(
                     "d_ij not symmetrized correctly over MPI ranks"));
#endif

          const auto c_ij = cij_matrix.template get_tensor<T>(i, col_idx);
          constexpr auto eps = std::numeric_limits<Number>::epsilon();
          const auto scale =
              dealii::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                  std::abs(d_ij), T(eps * eps), T(0.), T(1.) / d_ij);
          const auto scaled_c_ij = c_ij * scale;

          const auto flux_j = view.flux_contribution(
              old_precomputed, initial_precomputed_, js, U_j);

          const auto m_ij = mass_matrix.template get_entry<T>(i, col_idx);

          /*
           * Compute low-order flux and limiter bounds:
           */

          const auto flux_ij = view.flux_divergence(flux_i, flux_j, c_ij);
          U_i_new += tau * m_i_inv * flux_ij;
          auto P_ij = -flux_ij;

          if constexpr (shallow_water) {
            /*
             * Workaround: Shallow water (and related) are special:
             */

            const auto &[U_star_ij, U_star_ji] =
                view.equilibrated_states(flux_i, flux_j);

            U_i_new += tau * m_i_inv * d_ij * (U_star_ji - U_star_ij);
            F_iH += d_ijH * (U_star_ji - U_star_ij);
            P_ij += (d_ijH - d_ij) * (U_star_ji - U_star_ij);

            limiter.accumulate(
                U_j, U_star_ij, U_star_ji, scaled_c_ij, affine_shift);

          } else {

            U_i_new += tau * m_i_inv * d_ij * (U_j - U_i);
            F_iH += d_ijH * (U_j - U_i);
            P_ij += (d_ijH - d_ij) * (U_j - U_i);

            limiter.accumulate(js, U_j, flux_j, scaled_c_ij, affine_shift);
          }

          if constexpr (View::have_source_terms) {
            F_iH -= m_ij * S_iH;
            P_ij -= m_ij * /*sic!*/ S_i;
          }

          /*
           * Compute high-order fluxes and source terms:
           */

          if constexpr (View::have_high_order_flux) {
            const auto high_order_flux_ij =
                view.high_order_flux_divergence(flux_i, flux_j, c_ij);
            F_iH += weight * high_order_flux_ij;
            P_ij += weight * high_order_flux_ij;
          } else {
            F_iH += weight * flux_ij;
            P_ij += weight * flux_ij;
          }

          if constexpr (View::have_source_terms) {
            const auto S_j = view.nodal_source(old_precomputed, js, U_j, tau);
            F_iH += weight * m_ij * S_j;
            P_ij += weight * m_ij * S_j;
          }

          for (int s = 0; s < stages; ++s) {
            const auto &[U_s, prec_s, V_s] = stage_state_vectors[s].get();

            const auto U_jHs = U_s.template get_tensor<T>(js);
            const auto flux_jHs =
                view.flux_contribution(prec_s, initial_precomputed_, js, U_jHs);

            if constexpr (View::have_high_order_flux) {
              const auto high_order_flux_ij =
                  view.high_order_flux_divergence(flux_iHs[s], flux_jHs, c_ij);
              F_iH += stage_weights[s] * high_order_flux_ij;
              P_ij += stage_weights[s] * high_order_flux_ij;
            } else {
              const auto flux_ij =
                  view.flux_divergence(flux_iHs[s], flux_jHs, c_ij);
              F_iH += stage_weights[s] * flux_ij;
              P_ij += stage_weights[s] * flux_ij;
            }

            if constexpr (View::have_source_terms) {
              const auto S_js = view.nodal_source(prec_s, js, U_jHs, tau);
              F_iH += stage_weights[s] * m_ij * S_js;
              P_ij += stage_weights[s] * m_ij * S_js;
            }
          }

          pij_matrix_.write_entry(P_ij, i, col_idx, true);
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        if (!view.is_admissible(U_i_new)) {
          restart_needed = true;
        }
#endif

        new_U.template write_tensor<T>(U_i_new, i);
        r_.template write_tensor<T>(F_iH, i);

        const auto hd_i = m_i * measure_of_omega_inverse;
        const auto relaxed_bounds = limiter.bounds(hd_i);
        bounds_.template write_tensor<T>(relaxed_bounds, i);
      };

      /*
       * Chain through a compile time integral constant std::true_type for
       * a discontinuous ansatz and std::false_type otherwise. We use the
       * (constexpr) integral constant later on to avoid branching when
       * computing d_ijH.
       */
      if (offline_data_->discretization().have_discontinuous_ansatz()) {
        cpu_simd_loop<Number>(
            loop_name(),
            [&](auto sentinel, const unsigned int i) {
              body(sentinel, std::true_type{}, i);
            },
            0,
            n_internal,
            n_owned);
      } else {
        cpu_simd_loop<Number>(
            loop_name(),
            [&](auto sentinel, const unsigned int i) {
              body(sentinel, std::false_type{}, i);
            },
            0,
            n_internal,
            n_owned);
      }

      r_.update_ghost_values();
      if (offline_data_->discretization().have_discontinuous_ansatz()) {
        /*
         * In case we extend bounds over the stencil, we have to ensure
         * that ghost ranges are properly communicated over all MPI
         * ranks.
         */
        bounds_.update_ghost_values();
      }
    }

    /*
     * -------------------------------------------------------------------------
     * Step 5: Compute second part of P_ij, and l_ij (first round):
     * -------------------------------------------------------------------------
     */

    if (limiter_parameters_.iterations() != 0) {
      Scope scope(computing_timer_, scoped_name("compute p_ij, and l_ij"));

      auto body = [&](auto sentinel,
                      auto have_discontinuous_ansatz,
                      const unsigned int i) {
        using T = decltype(sentinel);
        using View =
            typename Description::template HyperbolicSystemView<dim, T>;
        using Limiter = typename Description::template Limiter<dim, T>;

        constexpr unsigned int stride_size = get_stride_size<T>;

        Limiter limiter(
            *hyperbolic_system_, limiter_parameters_, old_precomputed);

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          return;

        auto bounds =
            bounds_.template get_tensor<T, std::array<T, n_bounds>>(i);

        /*
         * In case of a discontinuous finite element ansatz we need to
         * extend bounds over the stencil. We do this by looping over the
         * stencil once and taking the minimum/maximum:
         */
        if constexpr (have_discontinuous_ansatz) {
          /* Skip diagonal. */
          const unsigned int *js = sparsity_simd.columns(i) + stride_size;
          for (unsigned int col_idx = 1; col_idx < row_length;
               ++col_idx, js += stride_size) {
            bounds = limiter.combine_bounds(
                bounds,
                bounds_.template get_tensor<T, std::array<T, n_bounds>>(js));
          }
          bounds_.template write_tensor<T>(bounds, i);
        }

        [[maybe_unused]] T m_i;
        if constexpr (have_discontinuous_ansatz)
          m_i = get_entry<T>(lumped_mass_matrix, i);
        const auto m_i_inv = get_entry<T>(lumped_mass_matrix_inverse, i);

        const auto U_i_new = new_U.template get_tensor<T>(i);

        const auto F_iH = r_.template get_tensor<T>(i);

        const auto lambda_inv = Number(row_length - 1);
        const auto factor = tau * m_i_inv * lambda_inv;

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd.columns(i) + stride_size;
        for (unsigned int col_idx = 1; col_idx < row_length;
             ++col_idx, js += stride_size) {

          auto P_ij = pij_matrix_.template get_tensor<T>(i, col_idx);
          const auto F_jH = r_.template get_tensor<T>(js);

          /*
           * Mass matrix correction:
           */

          const auto kronecker_ij = col_idx == 0 ? T(1.) : T(0.);

          if constexpr (have_discontinuous_ansatz) {
            /* Use full consistent mass matrix inverse: */

            const auto m_j = get_entry<T>(lumped_mass_matrix, js);
            const auto m_ij_inv =
                mass_matrix_inverse.template get_entry<T>(i, col_idx);
            const auto b_ij = m_i * m_ij_inv - kronecker_ij;
            const auto b_ji = m_j * m_ij_inv - kronecker_ij;

            P_ij += b_ij * F_jH - b_ji * F_iH;

          } else {
            /* Use Neumann series expansion: */

            const auto m_j_inv = get_entry<T>(lumped_mass_matrix_inverse, js);
            const auto m_ij = mass_matrix.template get_entry<T>(i, col_idx);
            const auto b_ij = kronecker_ij - m_ij * m_j_inv;
            const auto b_ji = kronecker_ij - m_ij * m_i_inv;

            P_ij += b_ij * F_jH - b_ji * F_iH;
          }

          P_ij *= factor;
          pij_matrix_.write_entry(P_ij, i, col_idx);

          /*
           * Compute limiter coefficients:
           */

          const auto &[l_ij, success] = limiter.limit(bounds, U_i_new, P_ij);
          lij_matrix_.template write_entry<T>(l_ij, i, col_idx, true);

          /*
           * If the success is set to false then the low-order update
           * resulted in a state outside of the limiter bounds. This can
           * happen if we compute with an aggressive CFL number. We
           * signal this condition by setting the restart_needed boolean
           * to true and defer further action to the chosen
           * IDViolationStrategy and the policy set in the
           * TimeIntegrator.
           */
          if (!success)
            restart_needed = true;
        }
      };

      /*
       * Chain through a compile time integral constant std::true_type for
       * a discontinuous ansatz and std::false_type otherwise. We use the
       * (constexpr) integral constant later on to avoid branching when
       * computing d_ijH.
       */
      if (offline_data_->discretization().have_discontinuous_ansatz()) {
        cpu_simd_loop<Number>(
            loop_name(),
            [&](auto sentinel, const unsigned int i) {
              body(sentinel, std::true_type{}, i);
            },
            0,
            n_internal,
            n_owned);
      } else {
        cpu_simd_loop<Number>(
            loop_name(),
            [&](auto sentinel, const unsigned int i) {
              body(sentinel, std::false_type{}, i);
            },
            0,
            n_internal,
            n_owned);
      }

      lij_matrix_.update_ghost_rows();
    }

    /*
     * -------------------------------------------------------------------------
     * Step 6, 7: Perform high-order update:
     *
     *   Symmetrize l_ij
     *   High-order update: += l_ij * lambda * P_ij
     *   Compute next l_ij
     * -------------------------------------------------------------------------
     */

    const auto n_iterations = limiter_parameters_.iterations();
    for (unsigned int pass = 0; pass < n_iterations; ++pass) {
      bool last_round = (pass + 1 == n_iterations);

      std::string additional_step = (last_round ? "" : ", next l_ij");
      Scope scope(
          computing_timer_,
          scoped_name("symmetrize l_ij, h.-o. update" + additional_step));

      if ((n_iterations == 2) && last_round) {
        std::swap(lij_matrix_, lij_matrix_next_);
      }

      auto body = [&](auto sentinel, const unsigned int i) {
        using T = decltype(sentinel);
        using View =
            typename Description::template HyperbolicSystemView<dim, T>;
        using Limiter = typename Description::template Limiter<dim, T>;

        Limiter limiter(
            *hyperbolic_system_, limiter_parameters_, old_precomputed);

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd.row_length(i);
        if (row_length == 1)
          return;

        auto U_i_new = new_U.template get_tensor<T>(i);

        const Number lambda = Number(1.) / Number(row_length - 1);

        /* Stored thread locally: */
        boost::container::small_vector<T, 54> lij_row(row_length);

        /* Skip diagonal. */
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {

          const auto l_ij = std::min(
              lij_matrix_.template get_entry<T>(i, col_idx),
              lij_matrix_.template get_transposed_entry<T>(i, col_idx));

          const auto p_ij = pij_matrix_.template get_tensor<T>(i, col_idx);

          U_i_new += l_ij * lambda * p_ij;

          if (!last_round)
            lij_row[col_idx] = l_ij;
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        const auto view = hyperbolic_system_->template view<dim, T>();
        if (!view.is_admissible(U_i_new)) {
          restart_needed = true;
        }
#endif

        new_U.template write_tensor<T>(U_i_new, i);

        /* Skip computating l_ij and updating p_ij in the last round */
        if (last_round)
          return;

        const auto bounds =
            bounds_.template get_tensor<T, std::array<T, n_bounds>>(i);
        /* Skip diagonal. */
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {

          const auto old_l_ij = lij_row[col_idx];

          const auto new_p_ij = (T(1.) - old_l_ij) *
                                pij_matrix_.template get_tensor<T>(i, col_idx);

          const auto &[new_l_ij, success] =
              limiter.limit(bounds, U_i_new, new_p_ij);

          /*
           * This is the second pass of the limiter. Under rare
           * circumstances the previous high-order update might be
           * slightly out of bounds due to roundoff errors. This happens
           * for example in flat regions or in stagnation points at a
           * (slip boundary) point. The limiter should ensure that we do
           * not further manipulate the state in this case. We thus only
           * signal a restart condition if the `EXPENSIVE_BOUNDS_CHECK` debug
           * macro is defined.
           */
#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
          if (!success)
            restart_needed = true;
#endif

          /*
           * Shortcut: We omit updating the p_ij and q_ij matrices and
           * simply write (1 - l_ij^(1)) * l_ij^(2) into the l_ij matrix.
           *
           * This approach only works for at most two limiting steps.
           */
          const auto entry = (T(1.) - old_l_ij) * new_l_ij;
          lij_matrix_next_.write_entry(entry, i, col_idx, true);
        }
      };

      cpu_simd_loop<Number>(loop_name(), body, 0, n_internal, n_owned);

      if (!last_round) {
        lij_matrix_next_.update_ghost_rows();
      }
    } /* limiter_iter_ */

    CALLGRIND_STOP_INSTRUMENTATION;

    /*
     * Pass through the parabolic state vector
     */
    const auto &old_V = std::get<2>(old_state_vector);
    auto &new_V = std::get<2>(new_state_vector);
    new_V = old_V;

    /*
     * Do we have to restart?
     */

    {
      Scope scope(computing_timer_,
                  "time step [X] _ - synchronization barriers");

      /*
       * Synchronize whether we have to restart the time step. Even though
       * the restart condition itself only affects the local ensemble we
       * nevertheless need to synchronize the boolean in case we perform
       * synchronized global time steps. (Otherwise different ensembles
       * might end up with a different time step.)
       */
      restart_needed.store(Utilities::MPI::logical_or(
          restart_needed.load(), mpi_ensemble_.synchronization_communicator()));
    }

    if (restart_needed) {
      switch (id_violation_strategy_) {
      case IDViolationStrategy::warn:
        n_warnings_++;
#ifdef DEBUG_OUTPUT
        std::cout << "        raised warning, CFL/IDP violation encountered "
                  << std::endl;
#endif
        break;
      case IDViolationStrategy::raise_exception:
        n_restarts_++;
        /* Suggest a restart with tau_max: */
#ifdef DEBUG_OUTPUT
        std::cout << "        signalling restart (suggested_tau_max = "
                  << tau_max << ")" << std::endl;
#endif
        throw Restart{tau_max};
      }
    }

    /* Poison all values that are left invalid after the update step: */
    Vectors::debug_poison_invalid_values(new_state_vector, *offline_data_);

    /* Return the time step size tau: */
    return tau;
  }

} /* namespace ryujin */
