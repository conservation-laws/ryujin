//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include "gpu.h"
#include "hyperbolic_module.h"
#include "loop.h"
#include "mpi_ensemble.h"
#include "scope.h"
#include "simd.h"

#include <numeric>
#include <utility>

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
      , indicator_(hyperbolic_system, subsection + "/indicator")
      , limiter_(hyperbolic_system, subsection + "/limiter")
      , wave_speed_estimator_(hyperbolic_system,
                              subsection + "/wave speed estimator")
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

    const auto limiter_view = limiter_.template view<dim, Number>();
    AssertThrow(limiter_view.iterations() <= 2,
                dealii::ExcMessage(
                    "The number of limiter iterations must be between [0,2]"));

    /* Initialize vectors: */

    const auto &scalar_partitioner = offline_data_->scalar_partitioner();

    /* The alpha vector is also read on the host memory space: */
    alpha_.reinit_with_scalar_partitioner(scalar_partitioner,
                                          TransferPolicy::implicit_transfers);

    bounds_.reinit_with_scalar_partitioner(scalar_partitioner);
    r_.reinit_with_vector_partitioner(
        offline_data_->hyperbolic_vector_partitioner());

    /* Initialize the compact buffer used for updating boundary values: */

    boundary_states_.reinit(offline_data_->boundary_indices().size() *
                                problem_dimension,
                            TransferPolicy::implicit_transfers);

    /* Initialize matrices: */

    const auto &sparsity_simd = offline_data_->sparsity_pattern_simd();
    dij_matrix_.reinit(sparsity_simd);
    lij_matrix_.reinit(sparsity_simd);
    lij_matrix_next_.reinit(sparsity_simd);
    pij_matrix_.reinit(sparsity_simd);

    /* Set up initial precomputed vector: */

    initial_precomputed_ =
        initial_values_->interpolate_initial_precomputed_vector();

    /*
     * Move all temporary data structures to the correct memory space:
     */

    if constexpr (have_separate_memory_spaces) {
      using MemorySpace = selected_memory_space_t;

      bounds_.template move_to_memory_space<MemorySpace>();
      r_.template move_to_memory_space<MemorySpace>();

      dij_matrix_.template move_to_memory_space<MemorySpace>();
      lij_matrix_.template move_to_memory_space<MemorySpace>();
      lij_matrix_next_.template move_to_memory_space<MemorySpace>();
      pij_matrix_.template move_to_memory_space<MemorySpace>();

      /* The initial_precomputed vector is also read on the host memory space:*/
      initial_precomputed_.template copy_to_memory_space<MemorySpace>();
    }
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
    U.reinit_with_vector_partitioner(
        offline_data_->hyperbolic_vector_partitioner());
    precomputed.reinit_with_vector_partitioner(
        offline_data_->precomputed_vector_partitioner());

#ifdef DEBUG
    /* Poison all vectors: */
    using state_type = typename View::state_type;

    constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();

    const unsigned int n_owned = offline_data_->n_locally_owned();
    const auto U_view = U.view();
    const auto precomputed_view = precomputed.view();
    for (unsigned int i = 0; i < n_owned; ++i) {
      U_view.write_tensor(state_type{} * nan, i);
      precomputed_view.write_tensor(
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

    auto &[U, precomputed, parabolic] = state_vector;

    using MemorySpace = selected_memory_space_t;

    /* Ensure all vectors are resident on the correct memory space. */
    if constexpr (have_separate_memory_spaces) {
      Scope scope(computing_timer_, "time step [X] _ - memory space transfers");
      U.template move_to_memory_space<MemorySpace>();
      precomputed.template move_to_memory_space<MemorySpace>();
    }

    Scope scope(computing_timer_,
                "time step [H] 1 - update boundary values, precompute values");

    /*
     * Update boundary values and distribute the result over all MPI ranks.
     */

    apply_boundary_conditions<MemorySpace>(U, t);

    U.template update_ghost_values_on_memory_space<MemorySpace>();

    /*
     * Compute and populate precomputed values.
     */

    if constexpr (have_separate_memory_spaces)
      hyperbolic_system_->template fill_precomputed_values<MemorySpace>(
          *offline_data_, state_vector);
    else
      hyperbolic_system_->fill_precomputed_values(*offline_data_, state_vector);

    precomputed.template view<MemorySpace>().update_ghost_values();
  }


  template <typename Description, int dim, typename Number>
  template <typename MemorySpace>
  void HyperbolicModule<Description, dim, Number>::apply_boundary_conditions(
      HyperbolicVector &U, const Number t) const
  {
    constexpr auto n_comp = problem_dimension;

    const auto &boundary_indices = offline_data_->boundary_indices();
    const auto n_boundary_indices =
        static_cast<unsigned int>(boundary_indices.size());

    /*
     * Gather all boundary states into the compact, mirrored buffer:
     */
    {
      const auto U_view = std::as_const(U).template view<MemorySpace>();
      const auto *indices = boundary_indices.template view<MemorySpace>();
      auto *states = boundary_states_.template view<MemorySpace>();

      const auto body = [=](auto /*sentinel*/, unsigned int k) {
        const auto U_i = U_view.read_tensor(indices[k]);
        for (unsigned int d = 0; d < n_comp; ++d)
          states[k * n_comp + d] = U_i[d];
      };

      loop<MemorySpace, Number>("hyperbolic_module_gather_boundary_states",
                                body,
                                0,
                                /*no vectorization*/ 0,
                                n_boundary_indices);
    }

    /*
     * Apply boundary conditions on the host memory space. Requesting a
     * writable host view copies the buffer back from the default memory
     * space.
     */
    {
      auto *states =
          boundary_states_.template view<dealii::MemorySpace::Host>();

      const auto &boundary_map = offline_data_->boundary_map();
      const auto &boundary_slots = offline_data_->boundary_slots();
      const auto view = hyperbolic_system_->template view<dim, Number>();

      /* FIXME: not thread parallel... */
      for (std::size_t e = 0; e < boundary_map.size(); ++e) {
        const auto &[i, normal, normal_mass, boundary_mass, id, position] =
            boundary_map[e];

        /*
         * Relay the task of applying appropriate boundary conditions to the
         * Problem Description.
         */

        if (id == Boundary::do_nothing)
          continue;

        /*
         * Note: The boundary map can contain more than one entry for the
         * same degree of freedom. All such entries share the same position
         * in the buffer so that boundary conditions compose in the same way
         * as they would when operating on the state vector directly.
         */
        const auto k = boundary_slots[e];

        state_type U_i;
        for (unsigned int d = 0; d < n_comp; ++d)
          U_i[d] = states[k * n_comp + d];

        /* Use a lambda to avoid computing unnecessary state values */
        auto get_dirichlet = [position = position, t = t, this]() {
          return initial_values_->initial_state(position, t);
        };

        U_i = view.apply_boundary_conditions(id, U_i, normal, get_dirichlet);

        for (unsigned int d = 0; d < n_comp; ++d)
          states[k * n_comp + d] = U_i[d];
      }
    }

    /* Write back the updated boundary states: */
    {
      const auto U_view = U.template view<MemorySpace>();
      const auto *indices = boundary_indices.template view<MemorySpace>();
      const auto *states =
          std::as_const(boundary_states_).template view<MemorySpace>();

      const auto body = [=](auto /*sentinel*/, unsigned int k) {
        state_type U_i;
        for (unsigned int d = 0; d < n_comp; ++d)
          U_i[d] = states[k * n_comp + d];
        U_view.write_tensor(U_i, indices[k]);
      };

      loop<MemorySpace, Number>("hyperbolic_module_scatter_boundary_states",
                                body,
                                0,
                                /*no vectorization*/ 0,
                                n_boundary_indices);
    }
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
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE bool
    all_below_diagonal(unsigned int i, const unsigned int *js)
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

    auto &[old_U, old_precomputed, old_parabolic] = old_state_vector;
    auto &new_U = std::get<0>(new_state_vector);

    using MemorySpace = selected_memory_space_t;

    /* Ensure all vectors are resident on the correct memory space. */
    if constexpr (have_separate_memory_spaces) {
      Scope scope(computing_timer_, "time step [X] _ - memory space transfers");
      old_U.template copy_to_memory_space<MemorySpace>();
      old_precomputed.template copy_to_memory_space<MemorySpace>();
      for (int s = 0; s < stages; ++s) {
        const auto &[U_s, prec_s, V_s] = stage_state_vectors[s].get();
        U_s.template copy_to_memory_space<MemorySpace>();
        prec_s.template copy_to_memory_space<MemorySpace>();
      }
      new_U.template move_to_memory_space<MemorySpace>();
    }

    /*
     * Taking a view<>() might imply implicit memory space transfers. Let's
     * account for them in our computing timers.
     */
    computing_timer_["(re)initialize data structures"].start();

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

    /* Index ranges for the iteration over the sparsity pattern : */

    const unsigned int n_internal = offline_data_->n_locally_internal();
    const unsigned int n_owned = offline_data_->n_locally_owned();

    /* Sparsity pattern, matrices, boundary information: */

    const auto sparsity_simd_view =
        offline_data_->sparsity_pattern_simd().template view<MemorySpace>();

    const auto mass_matrix_view =
        offline_data_->mass_matrix().template view<MemorySpace>();
    const auto lumped_mass_matrix_view =
        offline_data_->lumped_mass_matrix().template view<MemorySpace>();
    const auto lumped_mass_matrix_inverse_view =
        offline_data_->lumped_mass_matrix_inverse()
            .template view<MemorySpace>();

    const auto cij_matrix_view =
        offline_data_->cij_matrix().template view<MemorySpace>();

    /*
     * The mass_matrix_inverse and incidence_matrix objects are only
     * initialized (and accessed) for a discontinuous ansatz:
     */
    const bool have_discontinuous_ansatz =
        offline_data_->discretization().have_discontinuous_ansatz();
    using MatrixReadView =
        decltype(offline_data_->mass_matrix().template view<MemorySpace>());
    const auto mass_matrix_inverse_view =
        have_discontinuous_ansatz
            ? offline_data_->mass_matrix_inverse().template view<MemorySpace>()
            : MatrixReadView{};
    const auto incidence_matrix_view =
        have_discontinuous_ansatz
            ? offline_data_->incidence_matrix().template view<MemorySpace>()
            : MatrixReadView{};

    const auto *coupling_boundary_pairs =
        offline_data_->coupling_boundary_pairs().template view<MemorySpace>();
    const auto n_coupling_boundary_pairs =
        offline_data_->coupling_boundary_pairs().size();

    const Number measure_of_omega_inverse =
        Number(1.) / offline_data_->measure_of_omega();

    /* Temporary matrices: */

    const auto dij_matrix_view = dij_matrix_.template view<MemorySpace>();
    const auto lij_matrix_view = lij_matrix_.template view<MemorySpace>();
    const auto lij_matrix_next_view =
        lij_matrix_next_.template view<MemorySpace>();
    const auto pij_matrix_view = pij_matrix_.template view<MemorySpace>();

    /* Vectors: */

    const auto initial_precomputed_view =
        initial_precomputed_.template view<MemorySpace>();

    const auto old_U_view = old_U.template view<MemorySpace>();
    const auto old_precomputed_view =
        old_precomputed.template view<MemorySpace>();

    /*
     * FIXME GPU: std::array::operator[] is not device capable. For the
     * time being use a Kokkos::Array for wrapping:
     */

    using HyperbolicVectorView = std::remove_const_t<decltype(old_U_view)>;
    using PrecomputedVectorView =
        std::remove_const_t<decltype(old_precomputed_view)>;

    Kokkos::Array<HyperbolicVectorView, stages> stage_U_view;
    Kokkos::Array<PrecomputedVectorView, stages> stage_precomputed_view;
    Kokkos::Array<Number, stages> stage_weight;
    for (int s = 0; s < stages; ++s) {
      const auto &[U_s, prec_s, V_s] = stage_state_vectors[s].get();
      stage_U_view[s] = U_s.template view<MemorySpace>();
      stage_precomputed_view[s] = prec_s.template view<MemorySpace>();
      stage_weight[s] = stage_weights[s];
    }

    const auto new_U_view = new_U.template view<MemorySpace>();

    const auto alpha_view = alpha_.template view<MemorySpace>();
    const auto bounds_view = bounds_.template view<MemorySpace>();
    const auto r_view = r_.template view<MemorySpace>();

    computing_timer_["(re)initialize data structures"].stop();

    /*
     * Create a local copy of cfl_ so that we do not capture "this" in the
     * compute kernels
     */
    const Number cfl = cfl_;

    const auto hyperbolic_system_views =
        make_select_view<dim, Number, MemorySpace>(*hyperbolic_system_);

    const auto indicator_views =
        make_select_view<dim, Number, MemorySpace>(indicator_);

    const auto limiter_views =
        make_select_view<dim, Number, MemorySpace>(limiter_);
    const auto n_limiter_iterations =
        limiter_.template view<dim, Number>().iterations();

    const auto wave_speed_estimator_views =
        make_select_view<dim, Number, MemorySpace>(wave_speed_estimator_);

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

    /* A flag signalling that a restart is necessary. */
    Mirrored<int> restart_needed("hyperbolic_module_restart_needed",
                                 TransferPolicy::implicit_transfers);
    *restart_needed.view() = 0;

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

      const auto body = [=](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);

        const unsigned int stride_size = sparsity_simd_view.stride_of_row(i);

        const auto wave_speed_estimator_view =
            wave_speed_estimator_views.template view<T>();

        auto indicator_view = indicator_views.template view<T>();

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (row_length == 1)
          return;

        const auto U_i = old_U_view.template read_tensor<T>(i);

        indicator_view.reset(old_precomputed_view, i, U_i);

        const unsigned int *js = sparsity_simd_view.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto U_j = old_U_view.template read_tensor<T>(js);

          const auto c_ij = cij_matrix_view.template read_tensor<T>(i, col_idx);

          indicator_view.accumulate(old_precomputed_view, js, U_j, c_ij);

          /* Skip diagonal. */
          if (col_idx == 0)
            continue;

          /* Only iterate over the upper triangular portion of d_ij */
          if (all_below_diagonal<T>(i, js))
            continue;

          const auto norm = c_ij.norm();
          const auto n_ij = c_ij / norm;
          const auto lambda_max = wave_speed_estimator_view.compute(
              old_precomputed_view, U_i, U_j, i, js, n_ij);
          const auto d_ij = norm * lambda_max;

          dij_matrix_view.write_entry(d_ij, i, col_idx, true);
        }

        const auto mass = lumped_mass_matrix_view.template read_entry<T>(i);
        const auto hd_i = mass * measure_of_omega_inverse;
        alpha_view.template write_entry<T>(indicator_view.alpha(hd_i), i);
      };

      loop<MemorySpace, Number>(loop_name(), body, 0, n_internal, n_owned);

      alpha_view.update_ghost_values();
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
      const auto body_boundary = [=](auto, const unsigned int k) {
        const auto &[i, col_idx, j] = coupling_boundary_pairs[k];

        const auto wave_speed_estimator_view =
            wave_speed_estimator_views.template view<Number>();

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

        const auto U_i = old_U_view.read_tensor(i);
        const auto U_j = old_U_view.read_tensor(j);

        const auto c_ji = cij_matrix_view.read_transposed_tensor(i, col_idx);
        Assert(c_ji.norm() > 1.e-12, ExcInternalError());
        const auto norm_ji = c_ji.norm();
        const auto n_ji = c_ji / norm_ji;

        const auto d_ij = dij_matrix_view.read_entry(i, col_idx);

        const auto lambda_max = wave_speed_estimator_view.compute(
            old_precomputed_view, U_j, U_i, j, &i, n_ji);
        const auto d_ji = norm_ji * lambda_max;

        dij_matrix_view.write_entry(std::max(d_ij, d_ji), i, col_idx);
      };

      loop<MemorySpace, Number>(loop_name(),
                                body_boundary,
                                0,
                                /*no vectorization*/ 0,
                                n_coupling_boundary_pairs);

      /* Symmetrize d_ij and compute the maximal time-step size: */
      const auto body = [=](auto, unsigned int i, Number &result) {

#ifdef DEBUG_SYMMETRY_CHECK
        const auto wave_speed_estimator_view =
            wave_speed_estimator_views.template view<Number>();
#endif

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (row_length == 1)
          return;

        Number d_sum = Number(0.);

        /* skip diagonal: */
        const unsigned int stride_size = sparsity_simd_view.stride_of_row(i);
        const unsigned int *js = sparsity_simd_view.columns(i);
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
          const auto j = *(js + col_idx * stride_size);

          // fill lower triangular part of dij_matrix missing from step 1
          if (j < i) {
            const auto d_ji = dij_matrix_view.read_transposed_entry(i, col_idx);

#ifdef DEBUG_SYMMETRY_CHECK
            /* Verify that d_ji == std::max(d_ij, d_ji): */

            const auto U_i = old_U_view.read_tensor(i);
            const auto U_j = old_U_view.read_tensor(j);

            const auto c_ij = cij_matrix_view.read_tensor(i, col_idx);
            Assert(c_ij.norm() > 1.e-12, ExcInternalError());
            const auto norm_ij = c_ij.norm();
            const auto n_ij = c_ij / norm_ij;

            const auto lambda_max = wave_speed_estimator_view.compute(
                old_precomputed_view, U_i, U_j, i, &j, n_ij);
            const auto d_ij = norm_ij * lambda_max;

            Assert(d_ij <= d_ji + 1.0e-12,
                   dealii::ExcMessage("d_ij not symmetrized correctly on "
                                      "boundary degrees of freedom."));
#endif

            dij_matrix_view.write_entry(d_ji, i, col_idx);
          }

          d_sum -= dij_matrix_view.read_entry(i, col_idx);
        }

        /*
         * Make sure that we do not accidentally divide by zero. (Yes, this
         * can happen for some (admittedly, rather esoteric) scalar
         * conservation equations...).
         */
        d_sum =
            std::min(d_sum, Number(-1.e6) * std::numeric_limits<Number>::min());

        /* write diagonal element */
        dij_matrix_view.write_entry(d_sum, i, 0);

        const Number mass = lumped_mass_matrix_view.read_entry(i);
        const Number local_tau = cfl * mass / (Number(-2.) * d_sum);

        result = std::min(result, local_tau);
      };

      tau_max = reduction_loop<MemorySpace, Kokkos::Min<Number>>(
          loop_name(), body, tau_max, 0, n_owned);
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
      std::cout << "        computed tau_max = " << tau_max << " (CFL = " << cfl
                << ")" << std::endl;
      std::cout << "        step with tau    = " << tau << std::endl;
#endif

      /* We need to signal a restart if the enforced tau is too wacky: */
      *restart_needed.view() = (tau > acceptable_tau_max_ratio_ * tau_max);

      /* Don't bother with computing the update step, signal a restart: */
      if (*restart_needed.view() &&
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

    /* moves the "boolean" to device memory space: */
    int *restart_needed_view = restart_needed.view<MemorySpace>();

#ifdef DEBUG
    /*  Exchange d_ij so that we can check for symmetry: */
    dij_matrix_view.update_ghost_rows();
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

      const auto body = [=](auto sentinel,
                            auto have_discontinuous_ansatz,
                            const unsigned int i) {
        using T = decltype(sentinel);

        const auto view = hyperbolic_system_views.template view<T>();

        using View = decltype(view);
        using flux_contribution_type = typename View::flux_contribution_type;
        using state_type = typename View::state_type;

        const unsigned int stride_size = sparsity_simd_view.stride_of_row(i);

        auto limiter_view = limiter_views.template view<T>();

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (row_length == 1)
          return;

        const auto U_i = old_U_view.template read_tensor<T>(i);
        auto U_i_new = U_i;

        const auto alpha_i = alpha_view.template read_entry<T>(i);
        const auto m_i = lumped_mass_matrix_view.template read_entry<T>(i);
        const auto m_i_inv =
            lumped_mass_matrix_inverse_view.template read_entry<T>(i);

        const auto flux_i = view.flux_contribution(
            old_precomputed_view, initial_precomputed_view, i, U_i);

        Kokkos::Array<flux_contribution_type, stages> flux_iHs;
        [[maybe_unused]] state_type S_iH;

        for (int s = 0; s < stages; ++s) {
          const auto U_iHs = stage_U_view[s].template read_tensor<T>(i);
          flux_iHs[s] = view.flux_contribution(
              stage_precomputed_view[s], initial_precomputed_view, i, U_iHs);

          if constexpr (View::have_source_terms) {
            S_iH += stage_weight[s] *
                    view.nodal_source(stage_precomputed_view[s], i, U_iHs, tau);
          }
        }

        [[maybe_unused]] state_type S_i;
        state_type F_iH;

        if constexpr (View::have_source_terms) {
          S_i = view.nodal_source(old_precomputed_view, i, U_i, tau);
          S_iH += weight * S_i;
          U_i_new += tau * /* m_i_inv * m_i */ S_i;
          F_iH += m_i * S_iH;
        }

        limiter_view.reset(old_precomputed_view, i, U_i, flux_i);

        [[maybe_unused]] state_type affine_shift;

        /*
         * Workaround: For shallow water we need to accumulate an
         * additional contribution to the affine shift over the stencil
         * before we can compute limiter bounds.
         */

        const unsigned int *js = sparsity_simd_view.columns(i);
        if constexpr (shallow_water) {
          for (unsigned int col_idx = 0; col_idx < row_length;
               ++col_idx, js += stride_size) {

            const auto U_j = old_U_view.template read_tensor<T>(js);
            const auto flux_j = view.flux_contribution(
                old_precomputed_view, initial_precomputed_view, js, U_j);

            const auto d_ij =
                dij_matrix_view.template read_entry<T>(i, col_idx);
            const auto c_ij =
                cij_matrix_view.template read_tensor<T>(i, col_idx);

            const auto B_ij = view.affine_shift(flux_i, flux_j, c_ij, d_ij);
            affine_shift += B_ij;
          }

          affine_shift *= tau * m_i_inv;
        }

        if constexpr (View::have_source_terms) {
          affine_shift += tau * /* m_i_inv * m_i */ S_i;
        }

        js = sparsity_simd_view.columns(i);
        for (unsigned int col_idx = 0; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto U_j = old_U_view.template read_tensor<T>(js);

          const auto alpha_j = alpha_view.template read_entry<T>(js);

          const auto d_ij = dij_matrix_view.template read_entry<T>(i, col_idx);
          auto factor = (alpha_i + alpha_j) * Number(.5);

          if constexpr (have_discontinuous_ansatz) {
            const auto incidence_ij =
                incidence_matrix_view.template read_entry<T>(i, col_idx);
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
              dij_matrix_view.template read_transposed_entry<T>(i, col_idx);
          Assert(std::max(std::abs(d_ij - d_ji), T(1.0e-12)) == T(1.0e-12),
                 dealii::ExcMessage(
                     "d_ij not symmetrized correctly over MPI ranks"));
#endif

          const auto c_ij = cij_matrix_view.template read_tensor<T>(i, col_idx);
          constexpr auto eps = std::numeric_limits<Number>::epsilon();

          const auto scale =
              ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                  std::abs(d_ij), T(eps * eps), T(0.), T(1.) / d_ij);

          const auto scaled_c_ij = c_ij * scale;

          const auto flux_j = view.flux_contribution(
              old_precomputed_view, initial_precomputed_view, js, U_j);

          const auto m_ij = mass_matrix_view.template read_entry<T>(i, col_idx);

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

            limiter_view.accumulate(old_precomputed_view,
                                    U_j,
                                    U_star_ij,
                                    U_star_ji,
                                    scaled_c_ij,
                                    affine_shift);

          } else {

            U_i_new += tau * m_i_inv * d_ij * (U_j - U_i);
            F_iH += d_ijH * (U_j - U_i);
            P_ij += (d_ijH - d_ij) * (U_j - U_i);

            limiter_view.accumulate(old_precomputed_view,
                                    js,
                                    U_j,
                                    flux_j,
                                    scaled_c_ij,
                                    affine_shift);
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
            const auto S_j =
                view.nodal_source(old_precomputed_view, js, U_j, tau);
            F_iH += weight * m_ij * S_j;
            P_ij += weight * m_ij * S_j;
          }

          for (int s = 0; s < stages; ++s) {
            const auto U_jHs = stage_U_view[s].template read_tensor<T>(js);
            const auto flux_jHs = view.flux_contribution(
                stage_precomputed_view[s], initial_precomputed_view, js, U_jHs);

            if constexpr (View::have_high_order_flux) {
              const auto high_order_flux_ij =
                  view.high_order_flux_divergence(flux_iHs[s], flux_jHs, c_ij);
              F_iH += stage_weight[s] * high_order_flux_ij;
              P_ij += stage_weight[s] * high_order_flux_ij;
            } else {
              const auto flux_ij =
                  view.flux_divergence(flux_iHs[s], flux_jHs, c_ij);
              F_iH += stage_weight[s] * flux_ij;
              P_ij += stage_weight[s] * flux_ij;
            }

            if constexpr (View::have_source_terms) {
              const auto S_js =
                  view.nodal_source(stage_precomputed_view[s], js, U_jHs, tau);
              F_iH += stage_weight[s] * m_ij * S_js;
              P_ij += stage_weight[s] * m_ij * S_js;
            }
          }

          pij_matrix_view.template write_tensor<T>(P_ij, i, col_idx, true);
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        if (!view.is_admissible(U_i_new)) {
          Kokkos::atomic_store(restart_needed_view, 1);
        }
#endif

        new_U_view.template write_tensor<T>(U_i_new, i);
        r_view.template write_tensor<T>(F_iH, i);

        const auto hd_i = m_i * measure_of_omega_inverse;
        const auto relaxed_bounds = limiter_view.bounds(hd_i);
        bounds_view.template write_tensor<T>(relaxed_bounds, i);
      };

      /*
       * Chain through a compile time integral constant std::true_type for
       * a discontinuous ansatz and std::false_type otherwise. We use the
       * (constexpr) integral constant later on to avoid branching when
       * computing d_ijH.
       */
      if (have_discontinuous_ansatz) {
        loop<MemorySpace, Number>(
            loop_name(), body, 0, n_internal, n_owned, std::true_type{});
      } else {
        loop<MemorySpace, Number>(
            loop_name(), body, 0, n_internal, n_owned, std::false_type{});
      }

      r_view.update_ghost_values();
      if (have_discontinuous_ansatz) {
        /*
         * In case we extend bounds over the stencil, we have to ensure
         * that ghost ranges are properly communicated over all MPI
         * ranks.
         */
        bounds_view.update_ghost_values();
      }
    }

    /*
     * -------------------------------------------------------------------------
     * Step 5: Compute second part of P_ij, and l_ij (first round):
     * -------------------------------------------------------------------------
     */

    if (n_limiter_iterations != 0) {
      Scope scope(computing_timer_, scoped_name("compute p_ij, and l_ij"));

      const auto body = [=](auto sentinel,
                            auto have_discontinuous_ansatz,
                            const unsigned int i) {
        using T = decltype(sentinel);

        const unsigned int stride_size = sparsity_simd_view.stride_of_row(i);

        auto limiter_view = limiter_views.template view<T>();

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (row_length == 1)
          return;

        auto local_bounds =
            bounds_view.template read_tensor<T, std::array<T, n_bounds>>(i);

        /*
         * In case of a discontinuous finite element ansatz we need to
         * extend bounds over the stencil. We do this by looping over the
         * stencil once and taking the minimum/maximum:
         */
        if constexpr (have_discontinuous_ansatz) {
          /* Skip diagonal. */
          const unsigned int *js = sparsity_simd_view.columns(i) + stride_size;
          for (unsigned int col_idx = 1; col_idx < row_length;
               ++col_idx, js += stride_size) {
            local_bounds = limiter_view.combine_bounds(
                local_bounds,
                bounds_view.template read_tensor<T, std::array<T, n_bounds>>(
                    js));
          }
          bounds_view.template write_tensor<T>(local_bounds, i);
        }

        [[maybe_unused]] T m_i;
        if constexpr (have_discontinuous_ansatz)
          m_i = lumped_mass_matrix_view.template read_entry<T>(i);

        const auto m_i_inv =
            lumped_mass_matrix_inverse_view.template read_entry<T>(i);

        const auto U_i_new = new_U_view.template read_tensor<T>(i);

        const auto F_iH = r_view.template read_tensor<T>(i);

        const auto lambda_inv = Number(row_length - 1);
        const auto factor = tau * m_i_inv * lambda_inv;

        /*
         * Note: We "software-pipeline" the read access into the p_ij
         * matrix by one column index. This ensures that the P_ij entry of
         * the next column index is loaded *before* we store the updated
         * entry for the current column index. Otherwise we run into
         * aliasing issues that force the (cuda) compiler to serialize the
         * final store of P_ij and the read of the next entry.
         */
        auto P_ij = pij_matrix_view.template read_tensor<T>(i, 1);

        /* Skip diagonal. */
        const unsigned int *js = sparsity_simd_view.columns(i) + stride_size;
        for (unsigned int col_idx = 1; col_idx < row_length;
             ++col_idx, js += stride_size) {

          const auto P_ij_next = pij_matrix_view.template read_tensor<T>(
              i, col_idx + 1 < row_length ? col_idx + 1 : col_idx);
          const auto F_jH = r_view.template read_tensor<T>(js);

          /*
           * Mass matrix correction:
           */

          const auto kronecker_ij = col_idx == 0 ? T(1.) : T(0.);

          if constexpr (have_discontinuous_ansatz) {
            /* Use full consistent mass matrix inverse: */

            const auto m_j = lumped_mass_matrix_view.template read_entry<T>(js);
            const auto m_ij_inv =
                mass_matrix_inverse_view.template read_entry<T>(i, col_idx);
            const auto b_ij = m_i * m_ij_inv - kronecker_ij;
            const auto b_ji = m_j * m_ij_inv - kronecker_ij;

            P_ij += b_ij * F_jH - b_ji * F_iH;

          } else {
            /* Use Neumann series expansion: */

            const auto m_j_inv =
                lumped_mass_matrix_inverse_view.template read_entry<T>(js);
            const auto m_ij =
                mass_matrix_view.template read_entry<T>(i, col_idx);
            const auto b_ij = kronecker_ij - m_ij * m_j_inv;
            const auto b_ji = kronecker_ij - m_ij * m_i_inv;

            P_ij += b_ij * F_jH - b_ji * F_iH;
          }

          P_ij *= factor;
          pij_matrix_view.template write_tensor<T>(P_ij, i, col_idx);

          /*
           * Compute limiter coefficients:
           */

          const auto &[l_ij, success] =
              limiter_view.limit(local_bounds, U_i_new, P_ij);
          lij_matrix_view.template write_entry<T>(l_ij, i, col_idx, true);

          /*
           * If the success is set to false then the low-order update
           * resulted in a state outside of the limiter bounds. This can
           * happen if we compute with an aggressive CFL number. We
           * signal this condition by setting the restart_needed flag and
           * defer further action to the chosen IDViolationStrategy and the
           * policy set in the TimeIntegrator.
           */
          if (!success)
            Kokkos::atomic_store(restart_needed_view, 1);

          P_ij = P_ij_next;
        }
      };

      /*
       * Chain through a compile time integral constant std::true_type for
       * a discontinuous ansatz and std::false_type otherwise. We use the
       * (constexpr) integral constant later on to avoid branching when
       * computing d_ijH.
       */
      if (have_discontinuous_ansatz) {
        loop<MemorySpace, Number>(
            loop_name(), body, 0, n_internal, n_owned, std::true_type{});
      } else {
        loop<MemorySpace, Number>(
            loop_name(), body, 0, n_internal, n_owned, std::false_type{});
      }

      lij_matrix_view.update_ghost_rows();
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

    for (unsigned int pass = 0; pass < n_limiter_iterations; ++pass) {
      bool last_round = (pass + 1 == n_limiter_iterations);

      std::string additional_step = (last_round ? "" : ", next l_ij");
      Scope scope(
          computing_timer_,
          scoped_name("symmetrize l_ij, h.-o. update" + additional_step));

      const auto lij_view = (n_limiter_iterations == 2 && last_round)
                                ? lij_matrix_next_view
                                : lij_matrix_view;

      const auto body = [=](auto sentinel, const unsigned int i) {
        using T = decltype(sentinel);

        auto limiter_view = limiter_views.template view<T>();

        /* Skip constrained degrees of freedom: */
        const unsigned int row_length = sparsity_simd_view.row_length(i);
        if (row_length == 1)
          return;

        auto U_i_new = new_U_view.template read_tensor<T>(i);

        const Number lambda = Number(1.) / Number(row_length - 1);

        /* Skip diagonal. */
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {

          const auto l_ij =
              std::min(lij_view.template read_entry<T>(i, col_idx),
                       lij_view.template read_transposed_entry<T>(i, col_idx));

          const auto p_ij = pij_matrix_view.template read_tensor<T>(i, col_idx);

          U_i_new += l_ij * lambda * p_ij;
        }

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
        const auto view = hyperbolic_system_views.template view<T>();
        if (!view.is_admissible(U_i_new)) {
          Kokkos::atomic_store(restart_needed_view, 1);
        }
#endif

        new_U_view.template write_tensor<T>(U_i_new, i);

        /* Skip computating l_ij and updating p_ij in the last round */
        if (last_round)
          return;

        const auto local_bounds =
            bounds_view.template read_tensor<T, std::array<T, n_bounds>>(i);
        /* Skip diagonal. */
        for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {

          const auto old_l_ij =
              std::min(lij_view.template read_entry<T>(i, col_idx),
                       lij_view.template read_transposed_entry<T>(i, col_idx));

          const auto new_p_ij =
              (T(1.) - old_l_ij) *
              pij_matrix_view.template read_tensor<T>(i, col_idx);

          const auto &[new_l_ij, success] =
              limiter_view.limit(local_bounds, U_i_new, new_p_ij);

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
            Kokkos::atomic_store(restart_needed_view, 1);
#endif

          /*
           * Shortcut: We omit updating the p_ij and q_ij matrices and
           * simply write (1 - l_ij^(1)) * l_ij^(2) into the l_ij matrix.
           *
           * This approach only works for at most two limiting steps.
           */
          const auto entry = (T(1.) - old_l_ij) * new_l_ij;
          lij_matrix_next_view.write_entry(entry, i, col_idx, true);
        }
      };

      loop<MemorySpace, Number>(loop_name(), body, 0, n_internal, n_owned);

      if (!last_round) {
        lij_matrix_next_view.update_ghost_rows();
      }
    } /* limiter_iter_ */

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
       * nevertheless need to synchronize the flag in case we perform
       * synchronized global time steps. (Otherwise different ensembles
       * might end up with a different time step.)
       *
       * The host view reads the flag back from the selected memory space.
       */
      int &restart_flag = *restart_needed.view();
      restart_flag = Utilities::MPI::logical_or(
          restart_flag != 0, mpi_ensemble_.synchronization_communicator());
    }

    if (*restart_needed.view()) {
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
