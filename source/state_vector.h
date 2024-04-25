//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include "multicomponent_vector.h"

#include <deal.II/lac/la_parallel_block_vector.h>

namespace ryujin
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
  using StateVector =
      std::tuple<MultiComponentVector<Number, problem_dim> /*U*/,
                 MultiComponentVector<Number, prec_dim> /*precomputed values*/,
                 BlockVector<Number> /*parabolic state vector*/>;

} // namespace ryujin
