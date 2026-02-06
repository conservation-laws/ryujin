//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <deal.II/base/config.h>
#include <loop.h>
#include <observer_pointer.h>
#include <offline_data.h>
#include <simd.h>

#include <deal.II/base/vectorization.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

namespace ryujin
{
  template <int dim, typename Number, typename Number2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number> apply_B_n(
      const dealii::Tensor<1, (dim == 2 ? 1 : dim), Number> &magnetic_field,
      const Number2 theta_tau,
      const dealii::Tensor<1, dim, Number> &velocity)
  {
    if constexpr (dim == 1) {
      return velocity;

    } else if constexpr (dim == 2) {
      return velocity -
             theta_tau * magnetic_field[0] * cross_product_2d(velocity);

    } else {
      return velocity - theta_tau * cross_product_3d(velocity, magnetic_field);
    }
  }


  template <int dim, typename Number, typename Number2>
  DEAL_II_ALWAYS_INLINE inline dealii::Tensor<1, dim, Number> apply_B_n_inverse(
      const dealii::Tensor<1, (dim == 2 ? 1 : dim), Number> &magnetic_field,
      const Number2 &theta_tau,
      const dealii::Tensor<1, dim, Number> &velocity)
  {
    const auto denominator =
        Number(1.) + theta_tau * theta_tau * magnetic_field.norm_square();

    if constexpr (dim == 1) {
      return velocity;

    } else if constexpr (dim == 2) {
      const auto numerator =
          velocity + theta_tau * magnetic_field[0] * cross_product_2d(velocity);
      return numerator / denominator;

    } else {
      const auto numerator =
          velocity + theta_tau * cross_product_3d(velocity, magnetic_field) +
          theta_tau * theta_tau * (velocity * magnetic_field) * magnetic_field;
      return numerator / denominator;
    }
  }


#ifndef DOXYGEN
  template <typename T, typename... Args>
  void create(std::unique_ptr<T> &ptr, Args &&...args)
  {
    ptr = std::make_unique<T>(args...);
  }
#endif


  /**
   * A matrix-free operator that implements the action of the Laplace
   * operator.
   *
   * @ingroup ParabolicModule
   */
  template <int dim, typename Number>
  class LaplaceOperator : public dealii::EnableObserverPointer
  {
  public:
    // FIXME: refactor
    static constexpr unsigned int order_fe = 1;
    static constexpr unsigned int order_quad = 2;

    using ScalarVector = Vectors::ScalarVector<Number>;

    LaplaceOperator() = default;

    void initialize(const dealii::MatrixFree<dim, Number> &matrix_free)
    {
      matrix_free_ = &matrix_free;
    }

    dealii::types::global_dof_index m() const
    {
      return matrix_free_->get_vector_partitioner(0)->size();
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
            data, /*CG*/ 0, /*full quadrature*/ 0);

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

      const auto body_matrix_free =
          [](const auto &data, auto &dst, const auto &, const auto range) {
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

                fee_write.begin_dof_values()[i] =
                    fee_read.begin_dof_values()[i];
              }

              fee_write.distribute_local_to_global(dst);
            }
          };

      unsigned int dummy = 0;
      matrix_free_->template cell_loop<ScalarVector, unsigned int>(
          body_matrix_free,
          diagonal_vector,
          dummy,
          /*zero destination*/ true);

      /* invert diagonal matrix: */

      const auto n_owned_cg =
          diagonal_vector.get_partitioner()->locally_owned_size();

      const auto body_invert = [&](auto sentinel, const unsigned int i) {
        constexpr Number eps = std::numeric_limits<Number>::epsilon();
        using T = decltype(sentinel);
        const auto m_i = get_entry<T>(diagonal_vector, i);
        constexpr auto gt = dealii::SIMDComparison::greater_than;
        const auto m_i_inv = dealii::compare_and_apply_mask<gt>(
            std::abs(m_i), T(eps), Number(1.) / m_i, T(1.));
        write_entry<T>(diagonal_vector, m_i_inv, i);
      };
      cpu_simd_loop<Number>("", body_invert, 0, n_owned_cg, n_owned_cg);
    }

  private:
    const dealii::MatrixFree<dim, Number> *matrix_free_;
  };


  /**
   * A matrix-free operator that implements the action of the Laplace
   * operator.
   *
   * @ingroup EulerPoissonEquations
   */
  template <int dim, typename Number>
  class UpdateOperator : public dealii::EnableObserverPointer
  {
  public:
    // FIXME: refactor
    static constexpr unsigned int order_fe = 1;
    static constexpr unsigned int order_quad = 2;

    using ScalarVector = Vectors::ScalarVector<Number>;
    using BlockVector = Vectors::BlockVector<Number>;

    UpdateOperator() = default;

    void initialize(const dealii::MatrixFree<dim, Number> &matrix_free,
                    const ScalarVector &density,
                    const BlockVector &magnetic_field)
    {
      matrix_free_ = &matrix_free;
      density_ = &density;
      magnetic_field_ = &magnetic_field;

      theta_tau_ = Number(0.);
      alpha_ = Number(0.);
    }

    dealii::types::global_dof_index m() const
    {
      return matrix_free_->get_vector_partitioner(0)->size();
    }

    Number el(const unsigned int, const unsigned int) const
    {
      Assert(false, dealii::ExcNotImplemented());
      return Number();
    }

    void set_theta_tau(const Number theta_tau) const
    {
      theta_tau_ = theta_tau;
    }

    void set_alpha(const Number alpha) const
    {
      alpha_ = alpha;
    }

    void vmult(ScalarVector &dst, const ScalarVector &src) const
    {
      Assert(dst.get_partitioner() == src.get_partitioner(),
             dealii::ExcMessage("src and dst have 2 different partitioners"));

      using namespace dealii;

      const auto body_laplace = [](const auto &data,
                                   auto &dst,
                                   const auto &src,
                                   const auto range) {
        FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number> fee(
            data, /*CG*/ 0, /*full quadrature*/ 0);

        for (unsigned int cell = range.first; cell < range.second; ++cell) {
          fee.reinit(cell);
          fee.gather_evaluate(src, dealii::EvaluationFlags::gradients);
          for (unsigned int q = 0; q < fee.n_q_points; ++q)
            fee.submit_gradient(fee.get_gradient(q), q);
          fee.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
        }
      };

      matrix_free_->template cell_loop<ScalarVector, ScalarVector>(
          body_laplace, dst, src, /*zero destination*/ true);

      const auto body_velocity = [this](const auto &data,
                                        auto &dst,
                                        const auto &src,
                                        const auto range) {
        FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number> fee(
            data, /*CG*/ 0, /*lumped quadrature*/ 1);
        FEEvaluation<dim, order_fe, order_quad, /*components*/ 1, Number>
            fee_density(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);
        FEEvaluation<dim,
                     order_fe,
                     order_quad,
                     /*components*/ (dim == 2 ? 1 : dim),
                     Number>
            fee_magnetic(data, /*hyperbolic*/ 1, /*lumped quadrature*/ 1);

        for (unsigned int cell = range.first; cell < range.second; ++cell) {
          fee.reinit(cell);
          fee_density.reinit(cell);
          fee_magnetic.reinit(cell);

          fee.gather_evaluate(src, dealii::EvaluationFlags::gradients);
          fee_density.gather_evaluate(*density_,
                                      dealii::EvaluationFlags::values);
          fee_magnetic.gather_evaluate(*magnetic_field_,
                                       dealii::EvaluationFlags::values);

          for (unsigned int q = 0; q < fee.n_q_points; ++q) {
            const auto grad_phi = fee.get_gradient(q);
            auto density = fee_density.get_value(q);
            dealii::Tensor<1, (dim == 2 ? 1 : dim), decltype(density)>
                magnetic_field;
            if constexpr (dim == 2) {
              magnetic_field[0] = fee_magnetic.get_value(q);
            } else if constexpr (dim == 3) {
              magnetic_field = fee_magnetic.get_value(q);
            }

            const auto B_n_inverse_grad_phi =
                apply_B_n_inverse(magnetic_field, theta_tau_, grad_phi);
            const auto result = theta_tau_ * theta_tau_ * alpha_ * density *
                                B_n_inverse_grad_phi;
            fee.submit_gradient(result, q);
          }
          fee.integrate_scatter(dealii::EvaluationFlags::gradients, dst);
        }
      };

      matrix_free_->template cell_loop<ScalarVector, ScalarVector>(
          body_velocity, dst, src, /*zero destination*/ false);
    }

    void Tvmult(ScalarVector &dst, const ScalarVector &src) const
    {
      vmult(dst, src);
    }

  private:
    const dealii::MatrixFree<dim, Number> *matrix_free_;
    const ScalarVector *density_;
    const BlockVector *magnetic_field_;

    mutable Number theta_tau_;
    mutable Number alpha_;
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


  template <int dim, typename Number>
  class MGSmoother : public dealii::EnableObserverPointer
  {
  public:
    // FIXME: refactor
    static constexpr unsigned int order_fe = 1;
    static constexpr unsigned int order_quad = 2;

    using ScalarVector = Vectors::ScalarVector<Number>;
    using ScalarVectorFloat = Vectors::ScalarVector<float>;

    using Preconditioner =
        dealii::PreconditionChebyshev<LaplaceOperator<dim, float>,
                                      ScalarVectorFloat>;

    MGSmoother() = default;

    struct MultigridParameters {
      unsigned int gmg_max_iter;
      double gmg_smoother_range;
      double gmg_smoother_max_eig;
      unsigned int gmg_smoother_degree;
      unsigned int gmg_smoother_n_cg_iter;
      unsigned int gmg_min_level;
      double gmg_coarse_tolerance;
    };

    void initialize(const OfflineData<dim, Number> &offline_data,
                    const std::set<dealii::types::boundary_id> boundary_ids,
                    const MultigridParameters parameters)
    {
      using namespace dealii;

      /*
       * Set up multigrid operators and data structures:
       */

      const auto &discretization = offline_data.discretization();
      const auto &triangulation = discretization.triangulation();
      const unsigned int n_levels = triangulation.n_global_levels();
      const unsigned int min_level =
          std::min(parameters.gmg_min_level, n_levels - 1);
      MGLevelObject<IndexSet> relevant_sets(0, n_levels - 1);

      const auto &dof_handler = offline_data.dof_handler_cg();
      for (unsigned int level = 0; level < n_levels; ++level) {
#if DEAL_II_VERSION_GTE(9, 6, 0)
        relevant_sets[level] =
            dealii::DoFTools::extract_locally_relevant_level_dofs( //
                dof_handler,
                level);
#else
        dealii::DoFTools::extract_locally_relevant_level_dofs(
            dof_handler, level, relevant_sets[level]);
#endif
      }

      // First index CG, second index hyperbolic ansatz
      std::vector<const dealii::DoFHandler<dim> *> dof_handlers = {
          &dof_handler, &offline_data.dof_handler()};

      mg_constrained_dofs_.initialize(dof_handler, relevant_sets);
      /* FIXME: handle periodic boundary conditions and hanging nodes... */
      if (!boundary_ids.empty())
        mg_constrained_dofs_.make_zero_boundary_constraints( //
            dof_handler,
            boundary_ids);

      // First index full quadrature, second index lumped quadrature
      std::vector<dealii::Quadrature<1>> quadratures = {
          discretization.quadrature_1d()[0],
          discretization.nodal_quadrature_1d()[0]};

      typename MatrixFree<dim, float>::AdditionalData additional_data_level;
      additional_data_level.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;
      level_matrix_free_.resize(min_level, n_levels - 1);

      for (unsigned int level = min_level; level < n_levels; ++level) {
        additional_data_level.mg_level = level;

#if DEAL_II_VERSION_GTE(9, 6, 0)
        AffineConstraints<float> level_constraints(relevant_sets[level],
                                                   relevant_sets[level]);
#else
        AffineConstraints<float> level_constraints(relevant_sets[level]);
#endif

        if (!boundary_ids.empty()) {
          level_constraints.add_lines(
              mg_constrained_dofs_.get_boundary_indices(level));
#if DEAL_II_VERSION_GTE(9, 6, 0)
          level_constraints.merge(
              mg_constrained_dofs_.get_level_constraints(level));
#endif
        }
        level_constraints.close();

        AffineConstraints<float> dummy;
        dummy.close();
        std::vector<const dealii::AffineConstraints<float> *>
            level_constraints_list = {&level_constraints, &dummy};

        level_matrix_free_[level].reinit(discretization.mapping(),
                                         dof_handlers,
                                         level_constraints_list,
                                         quadratures,
                                         additional_data_level);
      }

      mg_transfer_.build(dof_handler, mg_constrained_dofs_, level_matrix_free_);

      level_laplace_matrices_.resize(level_matrix_free_.min_level(),
                                     level_matrix_free_.max_level());

      MGLevelObject<typename Preconditioner::AdditionalData> smoother_data(
          level_matrix_free_.min_level(), level_matrix_free_.max_level());

      for (unsigned int level = level_matrix_free_.min_level();
           level <= level_matrix_free_.max_level();
           ++level) {

        level_laplace_matrices_[level].initialize(level_matrix_free_[level]);
        smoother_data[level].preconditioner =
            std::make_shared<dealii::DiagonalMatrix<ScalarVectorFloat>>();
        level_laplace_matrices_[level].compute_diagonal(
            *smoother_data[level].preconditioner);

#if DEAL_II_VERSION_GTE(9, 6, 0)
        if (boundary_ids.empty()) {
          smoother_data[level].eigenvalue_algorithm =
              dealii::internal::EigenvalueAlgorithm::power_iteration;
        }
#endif

        if (level == level_matrix_free_.min_level()) {
          smoother_data[level].degree = numbers::invalid_unsigned_int;
          smoother_data[level].eig_cg_n_iterations = 500;
          smoother_data[level].smoothing_range = 1e-3;
        } else {
          smoother_data[level].degree = parameters.gmg_smoother_degree;
          smoother_data[level].eig_cg_n_iterations =
              parameters.gmg_smoother_n_cg_iter;
          smoother_data[level].smoothing_range = parameters.gmg_smoother_range;
          if (parameters.gmg_smoother_n_cg_iter == 0)
            smoother_data[level].max_eigenvalue =
                parameters.gmg_smoother_max_eig;
        }
      }

      relaxation_.initialize(level_laplace_matrices_, smoother_data);

      /*
       * Set up coarse solver:
       */

      create(coarse_solver_control_, 10000, parameters.gmg_coarse_tolerance);
      create(coarse_solver_data_);
      create(coarse_solver_, *coarse_solver_control_, *coarse_solver_data_);
      create(coarse_preconditioner_);
      level_laplace_matrices_[level_laplace_matrices_.min_level()]
          .compute_diagonal(*coarse_preconditioner_);
      create(coarse_grid_solver_,
             *coarse_solver_,
             level_laplace_matrices_[level_laplace_matrices_.min_level()],
             *coarse_preconditioner_);

      /*
       * Set up preconditioner:
       */

      create(mg_matrix_, level_laplace_matrices_);
      create(mg_,
             *mg_matrix_,
             *coarse_grid_solver_,
             mg_transfer_,
             relaxation_,
             relaxation_,
             level_laplace_matrices_.min_level(),
             level_laplace_matrices_.max_level());
      create(preconditioner_, dof_handler, *mg_, mg_transfer_);
    }

    void vmult(ScalarVector &dst, const ScalarVector &src) const
    {
      Assert(dst.get_partitioner() == src.get_partitioner(),
             dealii::ExcMessage("src and dst have 2 different partitioners"));
      preconditioner_->vmult(dst, src);
    }

    void Tvmult(ScalarVector &dst, const ScalarVector &src) const
    {
      vmult(dst, src);
    }

  private:
    //@}
    /**
     * @name Internal data
     */
    //@{

    dealii::MGConstrainedDoFs mg_constrained_dofs_;
    dealii::MGLevelObject<dealii::MatrixFree<dim, float>> level_matrix_free_;
    MGTransfer<dim, float> mg_transfer_;
    dealii::MGLevelObject<LaplaceOperator<dim, float>> level_laplace_matrices_;

    dealii::mg::SmootherRelaxation<Preconditioner, ScalarVectorFloat>
        relaxation_;

    std::unique_ptr<dealii::SolverControl> coarse_solver_control_;
    using CAD = typename dealii::SolverCG<ScalarVectorFloat>::AdditionalData;
    std::unique_ptr<CAD> coarse_solver_data_;
    std::unique_ptr<dealii::SolverCG<ScalarVectorFloat>> coarse_solver_;
    std::unique_ptr<dealii::DiagonalMatrix<ScalarVectorFloat>>
        coarse_preconditioner_;
    using MGCGIS = dealii::MGCoarseGridIterativeSolver<
        ScalarVectorFloat,
        dealii::SolverCG<ScalarVectorFloat>,
        LaplaceOperator<dim, float>,
        dealii::DiagonalMatrix<ScalarVectorFloat>>;
    std::unique_ptr<MGCGIS> coarse_grid_solver_;

    std::unique_ptr<dealii::mg::Matrix<ScalarVectorFloat>> mg_matrix_;
    std::unique_ptr<dealii::Multigrid<ScalarVectorFloat>> mg_;
    std::unique_ptr<
        dealii::PreconditionMG<dim, ScalarVectorFloat, MGTransfer<dim, float>>>
        preconditioner_;
    //@}
  };

} /* namespace ryujin */
