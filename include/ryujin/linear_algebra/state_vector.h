//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ryujin/base/computing_timer.h>
#include <ryujin/linear_algebra/multicomponent_vector.h>

#include <algorithm>
#include <concepts>
#include <limits>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  /* Forward declaration */
  namespace Interface
  {
    template <int dim, typename Number>
    class OfflineData;
  } // namespace Interface
#endif

  /**
   * A scalar vector representing a single component given by a deal.II
   * data type that is compatible with deal.II functions and methods and
   * lives in the host memory space.
   */
  template <typename Number>
  using ScalarHostVector = dealii::LinearAlgebra::distributed::Vector<Number>;


  /**
   * A scalar vector representing a single component, i.e., a
   * MultiComponentVector with n_comp() == 1.
   */
  template <typename Number>
  using ScalarVector = MultiComponentVector<Number>;


  /**
   * A compound state vector consisting of four parts:
   *
   *                 static (interleaved)    dynamic (scalar vectors)
   *   state         state                   dynamic_state
   *   precomputed   precomputed             dynamic_precomputed
   *
   * The parts @p state and @p precomputed belong to the hyperbolic system,
   * @p dynamic_state and @p dynamic_precomputed to the parabolic system.
   * The parts @p state and @p dynamic_state take part in sadd(), in the
   * solution transfer, and in a checkpoint. The precomputed parts are
   * recomputed.
   */
  template <typename Number>
  struct StateVector {
    MultiComponentVector<Number> state;
    MultiComponentVector<Number> precomputed;
    std::vector<ScalarVector<Number>> dynamic_state;
    std::vector<ScalarVector<Number>> dynamic_precomputed;
  };


  /**
   * An enum identifying the individual parts of a StateVector. A dynamic
   * part selects all scalar vectors of it.
   */
  enum class StateVectorPart {
    state,
    precomputed,
    dynamic_state,
    dynamic_precomputed,
  };


  /**
   * Ensure that the selected @p parts of the given @p state_vector are
   * resident on @p MemorySpace for read access.
   */
  template <typename MemorySpace,
            typename Number,
            std::same_as<StateVectorPart>... Parts>
  void copy_to_memory_space(const StateVector<Number> &state_vector
                            [[maybe_unused]],
                            Parts... parts [[maybe_unused]])
  {
    static_assert(sizeof...(Parts) > 0, "No state vector part selected");

    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");

      const auto &[U, Q, U_dynamic, Q_dynamic] = state_vector;

      const auto copy = [&](const StateVectorPart part) {
        switch (part) {
        case StateVectorPart::state:
          U.template copy_to_memory_space<MemorySpace>();
          break;
        case StateVectorPart::precomputed:
          Q.template copy_to_memory_space<MemorySpace>();
          break;
        case StateVectorPart::dynamic_state:
          for (const auto &U_k : U_dynamic)
            U_k.template copy_to_memory_space<MemorySpace>();
          break;
        case StateVectorPart::dynamic_precomputed:
          for (const auto &Q_k : Q_dynamic)
            Q_k.template copy_to_memory_space<MemorySpace>();
          break;
        }
      };

      (copy(parts), ...);
    }
  }


  /**
   * A variant of the above function that ensures that a single
   * (scalar, or multi-component) @p vector is resident on @p MemorySpace
   * for read access.
   */
  template <typename MemorySpace, typename Number>
  void copy_to_memory_space(const MultiComponentVector<Number> &vector
                            [[maybe_unused]])
  {
    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");
      vector.template copy_to_memory_space<MemorySpace>();
    }
  }


  /**
   * Ensure that the selected @p parts of the given @p state_vector are
   * resident on @p MemorySpace for write access.
   */
  template <typename MemorySpace,
            typename Number,
            std::same_as<StateVectorPart>... Parts>
  void move_to_memory_space(StateVector<Number> &state_vector [[maybe_unused]],
                            Parts... parts [[maybe_unused]])
  {
    static_assert(sizeof...(Parts) > 0, "No state vector part selected");

    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");

      auto &[U, Q, U_dynamic, Q_dynamic] = state_vector;

      const auto move = [&](const StateVectorPart part) {
        switch (part) {
        case StateVectorPart::state:
          U.template move_to_memory_space<MemorySpace>();
          break;
        case StateVectorPart::precomputed:
          Q.template move_to_memory_space<MemorySpace>();
          break;
        case StateVectorPart::dynamic_state:
          for (auto &U_k : U_dynamic)
            U_k.template move_to_memory_space<MemorySpace>();
          break;
        case StateVectorPart::dynamic_precomputed:
          for (auto &Q_k : Q_dynamic)
            Q_k.template move_to_memory_space<MemorySpace>();
          break;
        }
      };

      (move(parts), ...);
    }
  }


  /**
   * A variant of the above function that ensures that a single
   * (scalar, or multi-component) @p vector is resident on @p MemorySpace
   * for write access.
   */
  template <typename MemorySpace, typename Number>
  void move_to_memory_space(MultiComponentVector<Number> &vector
                            [[maybe_unused]])
  {
    if constexpr (have_separate_memory_spaces) {
      ComputingTimer::Scope scope("time step [X] _ - memory space transfers");
      vector.template move_to_memory_space<MemorySpace>();
    }
  }


  /**
   * Scaled addition $U\leftarrow s\,U+b\,V$ of the parts @p state and
   * @p dynamic_state of the given state vectors @p dst and @p src. The
   * operation is performed on @p MemorySpace. The precomputed parts are
   * left untouched.
   */
  template <typename MemorySpace, typename Number>
  void sadd(StateVector<Number> &dst,
            const Number s,
            const Number b,
            const StateVector<Number> &src)
  {
    copy_to_memory_space<MemorySpace>(
        src, StateVectorPart::state, StateVectorPart::dynamic_state);
    move_to_memory_space<MemorySpace>(
        dst, StateVectorPart::state, StateVectorPart::dynamic_state);

    auto &[dst_U, dst_Q, dst_U_dynamic, dst_Q_dynamic] = dst;
    const auto &[src_U, src_Q, src_U_dynamic, src_Q_dynamic] = src;

    auto &dst_U_vector = dst_U.template deal_ii_vector<MemorySpace>();
    dst_U_vector.zero_out_ghost_values();
    dst_U_vector.sadd(s, b, src_U.template deal_ii_vector<MemorySpace>());

    AssertDimension(dst_U_dynamic.size(), src_U_dynamic.size());

    for (std::size_t k = 0; k < dst_U_dynamic.size(); ++k) {
      auto &dst_k = dst_U_dynamic[k].template deal_ii_vector<MemorySpace>();
      const auto &src_k =
          src_U_dynamic[k].template deal_ii_vector<MemorySpace>();
      dst_k.zero_out_ghost_values();
      dst_k.sadd(s, b, src_k);
    }
  }


  /**
   * A small helper function that sets all values of the hyperbolic
   * vector that are invalid after a hyperbolic substep to a NaN value.
   * This includes:
   *  - the entire precomputed state vector
   *  - constrained degrees of freedom of the hyperbolic state vector
   *  - the ghost range of the hyperbolic state vector
   */
  template <typename Number, typename OfflineData>
  void debug_poison_invalid_values(StateVector<Number> &state_vector
                                   [[maybe_unused]],
                                   const OfflineData &offline_data
                                   [[maybe_unused]])
  {
#ifdef DEBUG

    move_to_memory_space<dealii::MemorySpace::Host>(
        state_vector, StateVectorPart::state, StateVectorPart::precomputed);

    auto &[U, Q, U_dynamic, Q_dynamic] = state_vector;

    constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();

    const unsigned int n_owned = offline_data.n_locally_owned();
    const unsigned int n_relevant = offline_data.n_locally_relevant();
    const auto &partitioner = offline_data.scalar_partitioner();

    /*
     * The number of components is a run time property: work on the
     * underlying deal.II vectors with n_comp() as stride.
     */
    const unsigned int prob_dim = U.n_comp();
    const unsigned int prec_dim = Q.n_comp();
    Number *const U_data = U.deal_ii_vector().begin();
    Number *const prec_data = Q.deal_ii_vector().begin();

    for (unsigned int i = 0; i < n_owned; ++i) {
      std::fill_n(prec_data + i * prec_dim, prec_dim, nan);

      if (!offline_data.affine_constraints().is_constrained(
              partitioner->local_to_global(i)))
        continue;
      std::fill_n(U_data + i * prob_dim, prob_dim, nan);
    }

    for (unsigned int i = n_owned; i < n_relevant; ++i) {
      std::fill_n(prec_data + i * prec_dim, prec_dim, nan);

      std::fill_n(U_data + i * prob_dim, prob_dim, nan);
    }
#endif
  }
} // namespace ryujin
