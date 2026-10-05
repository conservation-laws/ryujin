//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "computing_timer.h"
#include "multicomponent_vector.h"

#include <concepts>
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
     * A scalar vector representing a single component.
     */
    template <typename Number>
    using ScalarVector = MultiComponentVector<Number, 1>;


    /**
     * A compound state vector formed by a std::tuple consisting of the
     * hyperbolic state vector @p U, precomputed values, and a "parabolic
     * state" stored as a std::vector of scalar vectors, one per parabolic
     * component. All of these vectors have in common that they are
     * associated with a hyperbolic, or parabolic state and precomputed
     * data (derived from the hyperbolic state) for point in time.
     */
    template <typename Number, unsigned int problem_dim, unsigned int prec_dim>
    using StateVector = std::tuple<
        MultiComponentVector<Number, problem_dim> /*U*/,
        MultiComponentVector<Number, prec_dim> /*precomputed values*/,
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
              int prob_dim,
              int prec_dim,
              std::same_as<StateVectorPart>... Parts>
    void copy_to_memory_space(
        const StateVector<Number, prob_dim, prec_dim> &state_vector
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
    template <typename MemorySpace,
              typename Number,
              int n_comp,
              int simd_length>
    void copy_to_memory_space(
        const MultiComponentVector<Number, n_comp, simd_length> &vector
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
              int prob_dim,
              int prec_dim,
              std::same_as<StateVectorPart>... Parts>
    void
    move_to_memory_space(StateVector<Number, prob_dim, prec_dim> &state_vector
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
     * A small helper function that sets all values of the hyperbolic
     * vector that are invalid after a hyperbolic substep to a NaN value.
     * This includes:
     *  - the entire precomputed state vector
     *  - constrained degrees of freedom of the hyperbolic state vector
     *  - the ghost range of the hyperbolic state vector
     */
    template <typename Number, int prob_dim, int prec_dim, typename OfflineData>
    void debug_poison_invalid_values(
        StateVector<Number, prob_dim, prec_dim> &state_vector [[maybe_unused]],
        const OfflineData &offline_data [[maybe_unused]])
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

      const auto U_view = U.view();
      const auto prec_view = prec.view();

      for (unsigned int i = 0; i < n_owned; ++i) {
        prec_view.write_tensor(dealii::Tensor<1, prec_dim, Number>() * nan, i);

        if (!offline_data.affine_constraints().is_constrained(
                partitioner->local_to_global(i)))
          continue;
        U_view.write_tensor(dealii::Tensor<1, prob_dim, Number>() * nan, i);
      }

      for (unsigned int i = n_owned; i < n_relevant; ++i) {
        prec_view.write_tensor(dealii::Tensor<1, prec_dim, Number>() * nan, i);

        U_view.write_tensor(dealii::Tensor<1, prob_dim, Number>() * nan, i);
      }
#endif
    }

  } // namespace Vectors
} // namespace ryujin
