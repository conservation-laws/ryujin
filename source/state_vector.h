//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

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


    /**
     * Helper function that (re)initializes all components of a StateVector
     * to proper sizes.
     */
    template <
        typename Description,
        int dim,
        typename Number,
        typename View =
            typename Description::template HyperbolicSystemView<dim, Number>,
        int problem_dim = View::problem_dimension,
        int prec_dim = View::n_precomputed_values>
    void reinit_state_vector(
        StateVector<Number, problem_dim, prec_dim> &state_vector,
        const OfflineData<dim, Number> &offline_data)
    {
      auto &[U, precomputed, V] = state_vector;
      U.reinit(offline_data.hyperbolic_vector_partitioner());
      precomputed.reinit(offline_data.precomputed_vector_partitioner());

      const auto block_size = offline_data.n_auxiliary_state_vectors();
      V.reinit(block_size);
      for (unsigned int i = 0; i < block_size; ++i) {
        V.block(i).reinit(offline_data.scalar_partitioner());
      }
    }
  } // namespace Vectors

} // namespace ryujin
