//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/loop.h>
#include <ryujin/base/simd.h>
#include <ryujin/linear_algebra/state_vector.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

namespace ryujin
{
  namespace ScalarDiffusion
  {
    /**
     * The matrix-free operator of the backward Euler step of the power-law
     * diffusion model,
     * \f{align}
     *   (A u)_i = m_i u_i + \sigma \int_D k_h \nabla u_h\cdot\nabla\varphi_i
     *   \,\mathrm dx,
     * \f}
     * with the lumped mass matrix \f$m_i\f$, a scaling factor
     * \f$\sigma\ge0\f$, and the finite element interpolant \f$k_h\f$ of a
     * nodal diffusivity. The operator is symmetric positive definite.
     *
     * @ingroup ScalarDiffusionEquations
     */
    template <int dim, typename Number>
    class DiffusionOperator : public dealii::EnableObserverPointer
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      static constexpr unsigned int order_fe = 1;
      static constexpr unsigned int order_quad = 2;

      using VA = dealii::VectorizedArray<Number>;

      using vector_type = ScalarHostVector<Number>;

      //@}
      /**
       * @name Setup and application
       */
      //@{

      DiffusionOperator() = default;

      /**
       * Store references to the matrix free object and the lumped mass
       * matrix, and evaluate the nodal diffusivity @p diffusivity at all
       * quadrature points.
       */
      void reinit(const dealii::MatrixFree<dim, Number> &matrix_free,
                  const vector_type &lumped_mass_matrix,
                  const vector_type &diffusivity);

      /**
       * Set the scaling factor \f$\sigma\f$.
       */
      void set_factor(const Number factor)
      {
        factor_ = factor;
      }

      dealii::types::global_dof_index m() const
      {
        return matrix_free_->get_vector_partitioner()->size();
      }

      Number el(const unsigned int, const unsigned int) const
      {
        Assert(false, dealii::ExcNotImplemented());
        return Number();
      }

      void vmult(vector_type &dst, const vector_type &src) const;

      void Tvmult(vector_type &dst, const vector_type &src) const
      {
        vmult(dst, src);
      }

      /**
       * Compute the inverse of the diagonal of the operator.
       */
      void
      compute_diagonal(dealii::DiagonalMatrix<vector_type> &diagonal) const;

      /**
       * Compute the stiffness part \f$\int_D k_h \nabla u_h\cdot
       * \nabla\varphi_i\,\mathrm dx\f$ of the operator.
       */
      void apply_stiffness(vector_type &dst, const vector_type &src) const;

      //@}

    private:
      /**
       * @name Internal methods and data
       */
      //@{

      /**
       * The cell-local action of the stiffness part of the operator on the
       * evaluator @p phi.
       */
      template <typename Evaluator>
      void apply_local_operator(Evaluator &phi) const;

      const dealii::MatrixFree<dim, Number> *matrix_free_;
      const vector_type *lumped_mass_matrix_;
      Number factor_;

      dealii::AlignedVector<VA> diffusivity_;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    void DiffusionOperator<dim, Number>::reinit(
        const dealii::MatrixFree<dim, Number> &matrix_free,
        const vector_type &lumped_mass_matrix,
        const vector_type &diffusivity)
    {
      using namespace dealii;

      matrix_free_ = &matrix_free;
      lumped_mass_matrix_ = &lumped_mass_matrix;

      diffusivity.update_ghost_values();

      FEEvaluation<dim, order_fe, order_quad, 1, Number> k(matrix_free);
      constexpr unsigned int n_q_points = decltype(k)::static_n_q_points;

      const unsigned int n_cell_batches = matrix_free.n_cell_batches();
      diffusivity_.resize_fast(n_cell_batches * n_q_points);

      for (unsigned int cell = 0; cell < n_cell_batches; ++cell) {
        k.reinit(cell);
        k.read_dof_values(diffusivity);
        k.evaluate(EvaluationFlags::values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          diffusivity_[cell * n_q_points + q] = k.get_value(q);
      }
    }


    template <int dim, typename Number>
    template <typename Evaluator>
    void
    DiffusionOperator<dim, Number>::apply_local_operator(Evaluator &phi) const
    {
      using namespace dealii;

      phi.evaluate(EvaluationFlags::gradients);

      const unsigned int cell = phi.get_current_cell_index();
      const unsigned int n_q_points = phi.n_q_points;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const auto &k = diffusivity_[cell * n_q_points + q];
        phi.submit_gradient(k * phi.get_gradient(q), q);
      }

      phi.integrate(EvaluationFlags::gradients);
    }


    template <int dim, typename Number>
    void DiffusionOperator<dim, Number>::apply_stiffness(
        vector_type &dst, const vector_type &src) const
    {
      using namespace dealii;

      const auto integrator = [this](const auto &data,
                                     auto &dst,
                                     const auto &src,
                                     const auto range) {
        FEEvaluation<dim, order_fe, order_quad, 1, Number> phi(data);
        for (unsigned int cell = range.first; cell < range.second; ++cell) {
          phi.reinit(cell);
          phi.read_dof_values(src);
          apply_local_operator(phi);
          phi.distribute_local_to_global(dst);
        }
      };

      matrix_free_->template cell_loop<vector_type, vector_type>(
          integrator, dst, src, /* zero destination */ true);
    }


    template <int dim, typename Number>
    void DiffusionOperator<dim, Number>::vmult(vector_type &dst,
                                               const vector_type &src) const
    {
      apply_stiffness(dst, src);

      const unsigned int n_owned = dst.get_partitioner()->locally_owned_size();

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        const auto m_i = read_entry<T>(*lumped_mass_matrix_, i);
        const auto src_i = read_entry<T>(src, i);
        const auto dst_i = read_entry<T>(dst, i);
        write_entry<T>(dst, m_i * src_i + factor_ * dst_i, i);
      };

      cpu_simd_loop<Number>("", body, 0, n_owned, n_owned);
    }


    template <int dim, typename Number>
    void DiffusionOperator<dim, Number>::compute_diagonal(
        dealii::DiagonalMatrix<vector_type> &diagonal) const
    {
      using namespace dealii;

      auto &vector = diagonal.get_vector();
      matrix_free_->initialize_dof_vector(vector);

      MatrixFreeTools::compute_diagonal(
          *matrix_free_,
          vector,
          &DiffusionOperator::template apply_local_operator<
              FEEvaluation<dim, -1, 0, 1, Number>>,
          this);

      const unsigned int n_owned =
          vector.get_partitioner()->locally_owned_size();

      const auto body = [&](auto sentinel, unsigned int i) {
        using T = decltype(sentinel);
        const auto m_i = read_entry<T>(*lumped_mass_matrix_, i);
        const auto a_ii = m_i + factor_ * read_entry<T>(vector, i);
        /* Constrained degrees of freedom have a vanishing diagonal: */
        constexpr auto gt = SIMDComparison::greater_than;
        constexpr Number min = std::numeric_limits<Number>::min();
        write_entry<T>(
            vector,
            compare_and_apply_mask<gt>(a_ii, T(min), Number(1.) / a_ii, T(0.)),
            i);
      };

      cpu_simd_loop<Number>("", body, 0, n_owned, n_owned);
    }
  } // namespace ScalarDiffusion
} // namespace ryujin
