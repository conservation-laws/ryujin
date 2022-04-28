//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2021 by the ryujin authors
//

#pragma once

#include "introspection.h"
#include "openmp.h"
#include "scope.h"
#include "simd.h"

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <atomic>

#if DEAL_II_VERSION_GTE(9, 3, 0)
#define LOCAL_SIZE locally_owned_size
#else
#define LOCAL_SIZE local_size
#endif

namespace ryujin
{
  /**
   * Apply the action of the dissipation operator described by the
   * ParabolicSystem class.
   *
   * @ingroup DissipationModule
   */
  template <int dim, typename Number>
  class DissipationOperator : public dealii::Subscriptor
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
     * Default constructor.
     */
    DissipationOperator() = default;

    /**
     * Reinit with a MatrixFree object, a scalar partitioner used to
     * reinitialize the diagonal_action_ vector and a @p boundary_action
     * lambda that is called after the diagonal and element actions have
     * been applied.
     */
    template <typename Lambda>
    void reinit(const dealii::MatrixFree<dim, Number> &matrix_free,
                std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
                    scalar_partitioner,
                Lambda boundary_action)
    {
      matrix_free_ = &matrix_free;
      diagonal_action_.reinit(scalar_partitioner);
      boundary_action_ = boundary_action;
    }

    /**
     * Get access to the internal vector to be externally filled.
     */
    scalar_type &diagonal_action()
    {
      return diagonal_action_;
    }

    /**
     * Transpose operation. This is a symmetric operator thus this routine
     * simply calls vmult()
     */
    void Tvmult(block_vector_type &dst, const block_vector_type &src) const
    {
      vmult(dst, src);
    }

    /**
     * Apply the action of the operator.
     */
    void vmult(block_vector_type &dst, const block_vector_type &src) const
    {
      /* Apply diagonal action: */

      const auto n_blocks = src.n_blocks();
      AssertDimension(n_blocks, dst.n_blocks());

      const auto n_owned = diagonal_action_.get_partitioner()->LOCAL_SIZE();

      for (unsigned int d = 0; d < n_blocks; ++d) {
        AssertDimension(n_owned, src.block(d).get_partitioner()->LOCAL_SIZE());
        AssertDimension(n_owned, dst.block(d).get_partitioner()->LOCAL_SIZE());

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < n_owned; ++i)
          dst.block(d).local_element(i) =
              diagonal_action_.local_element(i) * src.block(d).local_element(i);
      }

      /* Apply element action: */

      //       const auto integrator = [this](const auto &data,
      //                                      auto &dst,
      //                                      const auto &src,
      //                                      const auto range) {
      //         constexpr auto order_fe =
      //         Discretization<dim>::order_finite_element; constexpr auto
      //         order_quad = Discretization<dim>::order_quadrature;

      //         dealii::FEEvaluation<dim, order_fe, order_quad, dim, Number>
      //         velocity(
      //             data);

      //         for (unsigned int cell = range.first; cell < range.second;
      //         ++cell) {
      //           velocity.reinit(cell);
      //           velocity.read_dof_values(src);
      //           apply_local_operator(velocity);
      //           velocity.distribute_local_to_global(dst);
      //         }
      //       };

      //       matrix_free_->template cell_loop<block_vector_type,
      //       block_vector_type>(
      //           integrator, dst, src, /* zero destination */ false);

      /* Fix up constrained degrees of freedom: */
      boundary_action_(dst, src);
    }

  private:
    const dealii::MatrixFree<dim, Number> *matrix_free_;
    scalar_type diagonal_action_;
    std::function<void(block_vector_type &dst, const block_vector_type &src)>
        boundary_action_;
  };

} // namespace ryujin

#undef LOCAL_SIZE
