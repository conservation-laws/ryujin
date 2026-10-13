//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "computing_timer.h"
#include "multicomponent_vector.h"

#include <algorithm>
#include <concepts>
#include <limits>
#include <tuple>
#include <vector>

namespace ryujin
{
#ifndef DOXYGEN
  /* Forward declaration */
  template <int dim, typename Number>
  class OfflineData;
#endif

  /**
   * A namespace for various vector type aliases.
   *
   * @ingroup LinearAlgebra
   */
  namespace Vectors
  {
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
     * A compound state vector formed by a std::tuple consisting of the
     * hyperbolic state vector @p U, precomputed values, and a "parabolic
     * state" stored as a std::vector of scalar vectors, one per parabolic
     * component. All of these vectors have in common that they are
     * associated with a hyperbolic, or parabolic state and precomputed
     * data (derived from the hyperbolic state) for point in time.
     */
    template <typename Number>
    using StateVector =
        std::tuple<MultiComponentVector<Number> /*U*/,
                   MultiComponentVector<Number> /*precomputed values*/,
                   std::vector<ScalarVector<Number>> /*parabolic state*/>;


    /**
     * An enum identifying the individual parts of a StateVector: the
     * hyperbolic state vector @p U, the precomputed values, and the
     * parabolic state (all scalar vectors of it).
     */
    enum class StateVectorPart {
      hyperbolic,
      precomputed,
      parabolic,
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

        const auto &[U, precomputed, parabolic] = state_vector;

        const auto copy = [&](const StateVectorPart part) {
          switch (part) {
          case StateVectorPart::hyperbolic:
            U.template copy_to_memory_space<MemorySpace>();
            break;
          case StateVectorPart::precomputed:
            precomputed.template copy_to_memory_space<MemorySpace>();
            break;
          case StateVectorPart::parabolic:
            for (const auto &V : parabolic)
              V.template copy_to_memory_space<MemorySpace>();
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
    template <typename MemorySpace, typename Number, int simd_length>
    void
    copy_to_memory_space(const MultiComponentVector<Number, simd_length> &vector
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
    void move_to_memory_space(StateVector<Number> &state_vector
                              [[maybe_unused]],
                              Parts... parts [[maybe_unused]])
    {
      static_assert(sizeof...(Parts) > 0, "No state vector part selected");

      if constexpr (have_separate_memory_spaces) {
        ComputingTimer::Scope scope("time step [X] _ - memory space transfers");

        auto &[U, precomputed, parabolic] = state_vector;

        const auto move = [&](const StateVectorPart part) {
          switch (part) {
          case StateVectorPart::hyperbolic:
            U.template move_to_memory_space<MemorySpace>();
            break;
          case StateVectorPart::precomputed:
            precomputed.template move_to_memory_space<MemorySpace>();
            break;
          case StateVectorPart::parabolic:
            for (auto &V : parabolic)
              V.template move_to_memory_space<MemorySpace>();
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
    template <typename MemorySpace, typename Number, int simd_length>
    void move_to_memory_space(MultiComponentVector<Number, simd_length> &vector
                              [[maybe_unused]])
    {
      if constexpr (have_separate_memory_spaces) {
        ComputingTimer::Scope scope("time step [X] _ - memory space transfers");
        vector.template move_to_memory_space<MemorySpace>();
      }
    }


    /**
     * Scaled addition $U\leftarrow s\,U+b\,V$ of the hyperbolic state
     * and all components of the parabolic state of the given state vectors
     * @p dst and @p src. The operation is performed on @p MemorySpace.
     * The precomputed values are left untouched.
     */
    template <typename MemorySpace, typename Number>
    void sadd(StateVector<Number> &dst,
              const Number s,
              const Number b,
              const StateVector<Number> &src)
    {
      copy_to_memory_space<MemorySpace>(
          src, StateVectorPart::hyperbolic, StateVectorPart::parabolic);
      move_to_memory_space<MemorySpace>(
          dst, StateVectorPart::hyperbolic, StateVectorPart::parabolic);

      /*
       * If the host and default memory spaces coincide all storage is
       * held by the host vector:
       */
      using VectorSpace = std::conditional_t<have_separate_memory_spaces,
                                             MemorySpace,
                                             dealii::MemorySpace::Host>;

      auto &dst_U = std::get<0>(dst).template deal_ii_vector<VectorSpace>();
      const auto &src_U =
          std::get<0>(src).template deal_ii_vector<VectorSpace>();
      dst_U.zero_out_ghost_values();
      dst_U.sadd(s, b, src_U);

      auto &dst_V = std::get<2>(dst);
      const auto &src_V = std::get<2>(src);
      AssertDimension(dst_V.size(), src_V.size());

      for (std::size_t k = 0; k < dst_V.size(); ++k) {
        auto &dst_V_k = dst_V[k].template deal_ii_vector<VectorSpace>();
        const auto &src_V_k = src_V[k].template deal_ii_vector<VectorSpace>();
        dst_V_k.zero_out_ghost_values();
        dst_V_k.sadd(s, b, src_V_k);
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
          state_vector,
          StateVectorPart::hyperbolic,
          StateVectorPart::precomputed);

      auto &[U, prec, V] = state_vector;

      constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();

      const unsigned int n_owned = offline_data.n_locally_owned();
      const unsigned int n_relevant = offline_data.n_locally_relevant();
      const auto &partitioner = offline_data.scalar_partitioner();

      /*
       * The number of components is a run time property: work on the
       * underlying deal.II vectors with n_comp() as stride.
       */
      const unsigned int prob_dim = U.n_comp();
      const unsigned int prec_dim = prec.n_comp();
      Number *const U_data = U.deal_ii_vector().begin();
      Number *const prec_data = prec.deal_ii_vector().begin();

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

  } // namespace Vectors
} // namespace ryujin
