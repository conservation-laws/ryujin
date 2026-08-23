//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "multicomponent_vector.h"

#include <deal.II/lac/la_parallel_block_vector.h>

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
     * A block vector representing a multiple components given by a deal.II
     * data type that is compatible with deal.II functions and methods and
     * lives in the host memory space.
     */
    template <typename Number>
    using BlockHostVector =
        dealii::LinearAlgebra::distributed::BlockVector<Number>;


    /**
     * A scalar vector representing a single component.
     */
    template <typename Number>
    using ScalarVector = MultiComponentVector<Number, 1>;


    /**
     * A compound state vector formed by a std::tuple consisting of the
     * hyperbolic state vector @p U, precomputed values, and an "parabolic
     * state" vector stored as a BlockVector. All of these vectors have in
     * common that they are associated with a hyperbolic, or parabolic state
     * and precomputed data (derived from the hyperbolic state) for point in
     * time.
     */
    template <typename Number, unsigned int problem_dim, unsigned int prec_dim>
    using StateVector = std::tuple<
        MultiComponentVector<Number, problem_dim> /*U*/,
        MultiComponentVector<Number, prec_dim> /*precomputed values*/,
        BlockHostVector<Number> /*parabolic state vector*/>;


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
