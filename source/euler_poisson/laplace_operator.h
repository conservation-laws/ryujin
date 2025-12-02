//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <observer_pointer.h>
#include <offline_data.h>
#include <openmp.h>
#include <simd.h>

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

namespace ryujin
{
  namespace EulerPoisson
  {
    template <int dim, typename Number, typename Number2>
    class LaplaceMatrix : public dealii::EnableObserverPointer
    {
    public:
      // FIXME: refactor
      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      using ScalarVector = Vectors::ScalarVector<Number>;

      LaplaceMatrix() = default;

      void initialize(
          const OfflineData<dim, Number2> &offline_data,
          const dealii::MatrixFree<dim, Number> &matrix_free,
          const unsigned int level = dealii::numbers::invalid_unsigned_int)
      {
        offline_data_ = &offline_data;
        matrix_free_ = &matrix_free;
        level_ = level;
      }

      dealii::types::global_dof_index m() const
      {
        Assert(false, dealii::ExcNotImplemented());
        return 0;
      }

      Number el(const unsigned int, const unsigned int) const
      {
        Assert(false, dealii::ExcNotImplemented());
        return Number();
      }

      void vmult(ScalarVector &dst, const ScalarVector &src) const
      {
        Assert(dst.get_partitioner() == src.get_partitioner(),
               dealii::ExcMessage("src and dst have 2 different partitioners"));

        using namespace dealii;

        const auto body = [](const auto &data,
                             auto &dst,
                             const auto &src,
                             const auto range) {
          FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number> fee(
              data, /*CG*/ 0, /*lumped quadrature*/ 1);

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            fee.reinit(cell);
            fee.gather_evaluate(src, dealii::EvaluationFlags::gradients);
            for (unsigned int q = 0; q < fee.n_q_points; ++q)
              fee.submit_gradient(fee.get_gradient(q), q);
            fee.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
          }
        };

        matrix_free_->template cell_loop<ScalarVector, ScalarVector>(
            body, dst, src, /*zero destination*/ true);
      }

      void Tvmult(ScalarVector &dst, const ScalarVector &src) const
      {
        vmult(dst, src);
      }

      void compute_diagonal(
          dealii::DiagonalMatrix<ScalarVector> &diagonal_matrix) const
      {
        using namespace dealii;

        ScalarVector &diagonal_vector = diagonal_matrix.get_vector();
        matrix_free_->initialize_dof_vector(diagonal_vector, /*CG*/ 0);

        const auto body = [](const auto &data,
                             auto &dst,
                             const auto &,
                             const auto range) {
          FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
              fee_read(data, /*CG*/ 0, /*lumped quadrature*/ 1);
          FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
              fee_write(data, /*CG*/ 0, /*lumped quadrature*/ 1);

          for (unsigned int cell = range.first; cell < range.second; ++cell) {
            fee_read.reinit(cell);
            fee_write.reinit(cell);

            for (unsigned int i = 0; i < fee_read.dofs_per_cell; ++i) {
              /* Set up shape function for degree i: */
              for (unsigned int j = 0; j < fee_read.dofs_per_cell; ++j)
                fee_read.begin_dof_values()[j] =
                    dealii::VectorizedArray<Number>();
              fee_read.begin_dof_values()[i] =
                  dealii::make_vectorized_array<Number>(1.);

              fee_read.evaluate(dealii::EvaluationFlags::gradients);
              for (unsigned int q = 0; q < fee_write.n_q_points; ++q)
                fee_write.submit_gradient(fee_read.get_gradient(q), q);

              fee_write.begin_dof_values()[i] = fee_read.begin_dof_values()[i];
            }

            fee_write.distribute_local_to_global(dst);
          }
        };

        unsigned int dummy = 0;
        matrix_free_->template cell_loop<ScalarVector, unsigned int>(
            body,
            diagonal_vector,
            dummy,
            /*zero destination*/ true);

        /* invert diagonal matrix: */

        const auto n_owned_cg =
            diagonal_vector.get_partitioner()->locally_owned_size();

        RYUJIN_PARALLEL_REGION_BEGIN

        RYUJIN_OMP_FOR
        for (unsigned int i = 0; i < n_owned_cg; ++i) {
          constexpr Number eps = std::numeric_limits<Number>::epsilon();
          diagonal_vector.local_element(i) =
              diagonal_vector.local_element(i) > eps
                  ? 1. / diagonal_vector.local_element(i)
                  : Number(0.);
        }

        RYUJIN_PARALLEL_REGION_END
      }

    private:
      const OfflineData<dim, Number2> *offline_data_;
      const dealii::MatrixFree<dim, Number> *matrix_free_;
      const ScalarVector *potential_;
      unsigned int level_;
    };


    template <int dim, typename Number>
    class MGTransfer : public dealii::MGTransferMatrixFree<dim, Number>
    {
    public:
      void build(const dealii::DoFHandler<dim> &dof_handler,
                 const dealii::MGConstrainedDoFs &mg_constrained_dofs,
                 const dealii::MGLevelObject<dealii::MatrixFree<dim, Number>>
                     &matrix_free)
      {
        dealii::MGTransferMatrixFree<dim, Number>::initialize_constraints(
            mg_constrained_dofs);
        dealii::MGTransferMatrixFree<dim, Number>::build(dof_handler);
        level_matrix_free_ = &matrix_free;
      }

      template <typename Number2>
      void copy_to_mg(
          const dealii::DoFHandler<dim> &dof_handler,
          dealii::MGLevelObject<
              dealii::LinearAlgebra::distributed::Vector<Number>> &dst,
          const dealii::LinearAlgebra::distributed::Vector<Number2> &src) const
      {
        if (dst[dst.min_level()].size() == 0)
          for (unsigned int l = dst.min_level(); l <= dst.max_level(); ++l)
            (*level_matrix_free_)[l].initialize_dof_vector(dst[l]);
        dealii::MGTransferMatrixFree<dim, Number>::copy_to_mg(
            dof_handler, dst, src);
      }

    private:
      const dealii::MGLevelObject<dealii::MatrixFree<dim, Number>>
          *level_matrix_free_;
    };

  } // namespace EulerPoisson
} /* namespace ryujin */

#undef locally_owned_size
