//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2023 by the ryujin authors
//

#pragma once

#include "state_vector.h"

#include <deal.II/lac/la_parallel_block_vector.h>

namespace ryujin
{
  /**
   * A preconditioner implementing a diagonal scaling used for the
   * non-multigrid CG iteration.
   *
   * @ingroup ParabolicModule
   */
  template <typename Number>
  class DiagonalPreconditioner
  {
  public:
    /**
     * @copydoc ryujin::ScalarVector
     */
    using ScalarVector = typename Vectors::ScalarVector<Number>;

    /**
     * @copydoc ryujin::BlockVector
     */
    using BlockVector = typename Vectors::BlockVector<Number>;

    /**
     * Constructor
     */
    DiagonalPreconditioner() = default;

    /**
     * Reinit with a scalar partitioner
     */
    void reinit(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    &scalar_partitioner)
    {
      diagonal_.reinit(scalar_partitioner);
    }

    /**
     * Get access to the internal vector to be externally filled.
     */
    ScalarVector &scaling_vector()
    {
      return diagonal_;
    }

    /**
     * Apply on a scalar vector.
     */
    void vmult(ScalarVector &dst, const ScalarVector &src) const
    {
      const auto n_owned = diagonal_.get_partitioner()->locally_owned_size();
      AssertDimension(n_owned, src.get_partitioner()->locally_owned_size());
      AssertDimension(n_owned, dst.get_partitioner()->locally_owned_size());

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < n_owned; ++i)
        dst.local_element(i) =
            diagonal_.local_element(i) * src.local_element(i);
    }

    /**
     * Apply on a block vector.
     */
    void vmult(BlockVector &dst, const BlockVector &src) const
    {
      const auto n_blocks = src.n_blocks();
      AssertDimension(n_blocks, dst.n_blocks());

      const auto n_owned = diagonal_.get_partitioner()->locally_owned_size();

      for (unsigned int d = 0; d < n_blocks; ++d) {
        AssertDimension(n_owned,
                        src.block(d).get_partitioner()->locally_owned_size());
        AssertDimension(n_owned,
                        dst.block(d).get_partitioner()->locally_owned_size());

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < n_owned; ++i)
          dst.block(d).local_element(i) =
              diagonal_.local_element(i) * src.block(d).local_element(i);
      }
    }

  private:
    ScalarVector diagonal_;
  };

} /* namespace ryujin */
