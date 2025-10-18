//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
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
   * @ingroup Mesh
   */
  namespace Vectors
  {
    /**
     * Shorthand for dealii::LinearAlgebra::distributed::Vector<Number>.
     */
    template <typename Number>
    using ScalarVector = dealii::LinearAlgebra::distributed::Vector<Number>;


    /**
     * Shorthand for dealii::LinearAlgebra::distributed::BlockVector<Number>.
     */
    template <typename Number>
    using BlockVector = dealii::LinearAlgebra::distributed::BlockVector<Number>;


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
        BlockVector<Number> /*parabolic state vector*/>;


    template <
        typename Description,
        int dim,
        typename Number,
        typename View =
            typename Description::template HyperbolicSystemView<dim, Number>,
        int problem_dimension = View::problem_dimension,
        int prec_dimension = View::n_precomputed_values>
    void debug_poison_constrained_dofs(
        StateVector<Number, problem_dimension, prec_dimension> &state_vector
        [[maybe_unused]],
        const OfflineData<dim, Number> &offline_data [[maybe_unused]])
    {
#ifdef DEBUG
      auto &[U, precomputed, V] = state_vector;

      const unsigned int n_owned = offline_data.n_locally_owned();
      const auto &partitioner = offline_data.scalar_partitioner();

      for (unsigned int i = 0; i < n_owned; ++i) {
        if (!offline_data.affine_constraints().is_constrained(
                partitioner->local_to_global(i)))
          continue;
        constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();
        U.write_tensor(dealii::Tensor<1, problem_dimension, Number>() * nan, i);
      }
#endif
    }


    template <
        typename Description,
        int dim,
        typename Number,
        typename View =
            typename Description::template HyperbolicSystemView<dim, Number>,
        int problem_dimension = View::problem_dimension,
        int prec_dimension = View::n_precomputed_values>
    void debug_poison_precomputed_values(
        StateVector<Number, problem_dimension, prec_dimension> &state_vector
        [[maybe_unused]],
        const OfflineData<dim, Number> &offline_data [[maybe_unused]])
    {
#ifdef DEBUG
      auto &[U, precomputed, V] = state_vector;

      constexpr auto nan = std::numeric_limits<Number>::signaling_NaN();
      const unsigned int n_owned = offline_data.n_locally_owned();

      for (unsigned int i = 0; i < n_owned; ++i) {
        precomputed.write_tensor(
            dealii::Tensor<1, prec_dimension, Number>() * nan, i);
      }
#endif
    }
  } // namespace Vectors
} // namespace ryujin
