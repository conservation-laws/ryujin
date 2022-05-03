//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "offline_data.h"
#include "openmp.h"
#include "simd.h"

#include <deal.II/lac/la_parallel_block_vector.h>

#if DEAL_II_VERSION_GTE(9, 3, 0)
#define LOCAL_SIZE locally_owned_size
#else
#define LOCAL_SIZE local_size
#endif

namespace ryujin
{
  /**
   * A preconditioner implementing a diagonal scaling used for the
   * non-multigrid CG iteration.
   *
   * @ingroup DissipationModule
   */
  template <int dim, typename Number>
  class DiagonalPreconditioner
  {
  public:
    /**
     * @copydoc OfflineData::scalar_type
     */
    using scalar_type = typename OfflineData<dim, Number>::scalar_type;

    /**
     * A distributed block vector used for temporary storage of the
     * velocity field.
     */
    using block_vector_type =
        dealii::LinearAlgebra::distributed::BlockVector<Number>;

    /**
     * Constructor
     */
    DiagonalPreconditioner() = default;

    /**
     * Reinit with a scalar partitioner
     */
    void reinit(std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    scalar_partitioner)
    {
      diagonal_.reinit(scalar_partitioner);
    }

    /**
     * Get access to the internal vector to be externally filled.
     */
    scalar_type &scaling_vector()
    {
      return diagonal_;
    }

    /**
     * Apply on a scalar vector.
     */
    void vmult(scalar_type &dst, const scalar_type &src) const
    {
      const auto n_owned = diagonal_.get_partitioner()->LOCAL_SIZE();
      AssertDimension(n_owned, src.get_partitioner()->LOCAL_SIZE());
      AssertDimension(n_owned, dst.get_partitioner()->LOCAL_SIZE());

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < n_owned; ++i)
        dst.local_element(i) =
            diagonal_.local_element(i) * src.local_element(i);
    }

    /**
     * Apply on a block vector.
     */
    void vmult(block_vector_type &dst, const block_vector_type &src) const
    {
      const auto n_blocks = src.n_blocks();
      AssertDimension(n_blocks, dst.n_blocks());

      const auto n_owned = diagonal_.get_partitioner()->LOCAL_SIZE();

      for (unsigned int d = 0; d < n_blocks; ++d) {
        AssertDimension(n_owned, src.block(d).get_partitioner()->LOCAL_SIZE());
        AssertDimension(n_owned, dst.block(d).get_partitioner()->LOCAL_SIZE());

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < n_owned; ++i)
          dst.block(d).local_element(i) =
              diagonal_.local_element(i) * src.block(d).local_element(i);
      }
    }

  private:
    scalar_type diagonal_;
  };

} /* namespace ryujin */

#undef LOCAL_SIZE
