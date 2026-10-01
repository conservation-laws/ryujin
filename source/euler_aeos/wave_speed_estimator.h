//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"
#include "nasg_riemann_solver.h"

#include <gpu.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace EulerAEOS
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class WaveSpeedEstimatorView;

    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite ClaytonGuermondPopov-2022.
     *
     * @ingroup EulerEquations
     */
    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : protected NASGRiemannSolver<ScalarNumber>
    {
    public:
      /**
       * @name Constructor and setup
       */
      //@{

      /**
       * Constructor.
       */
      WaveSpeedEstimator(const HyperbolicSystem &hyperbolic_system,
                         const std::string &subsection = "/WaveSpeedEstimator")
          : NASGRiemannSolver<ScalarNumber>(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
        /* Transfer all equation of state parameters to the Riemann solver. */

        const auto update_parameters = [this] {
          const auto view =
              hyperbolic_system_->template view<1, ScalarNumber>();
          this->set_equation_of_state(view.eos_covolume_constant(),
                                      view.eos_interpolation_pinfty(),
                                      view.compute_strict_bounds());
        };

        this->parse_parameters_call_back.connect(update_parameters);
        update_parameters();
      }

      /**
       * Return a view on the WaveSpeedEstimator for a given dimension @p dim
       * and choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars). The
       * optional @p MemorySpace template parameter selects whether the
       * view is intended for the host or device memory space.
       */
      template <int dim,
                typename Number,
                typename MemorySpace = dealii::MemorySpace::Host>
      auto view() const
      {
        return WaveSpeedEstimatorView<dim, Number, MemorySpace>{
            hyperbolic_system_->template view<dim, Number>(), *this};
      }

    private:
      //@}
      /**
       * @name Internal fields, methods, and friends
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

      template <int, typename, typename>
      friend class WaveSpeedEstimatorView;

      //@}
    };


    /**
     * A view of the WaveSpeedEstimator that makes the interface available
     * for a given dimension @p dim and choice of number type @p Number
     * (which can be a scalar float, or double, as well as a VectorizedArray
     * holding packed scalars).
     *
     * @ingroup EulerEquations
     */
    template <int dim, typename Number, typename MemorySpace>
    class WaveSpeedEstimatorView
    {
    public:
      static_assert(
          std::is_same_v<MemorySpace, dealii::MemorySpace::Host> ||
              std::is_same_v<MemorySpace, dealii::MemorySpace::Default>,
          "Unexpected memory space");

      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

      using ScalarNumber = typename View::ScalarNumber;

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      /**
       * The view on the Riemann solver used for computing the wavespeed
       * estimate.
       */
      using RiemannSolverView =
          NASGRiemannSolverView<Number,
                                NASGRiemannSolverOptions{},
                                MemorySpace>;

      /**
       * Number of components in a primitive state, we store \f$[\rho, v,
       * p, gamma, a]\f$, thus, 5.
       */
      static constexpr unsigned int riemann_data_size =
          RiemannSolverView::riemann_data_size;

      /**
       * The array type to store the expanded primitive state for the
       * Riemann solver \f$[\rho, v, p, gamma, a]\f$
       */
      using primitive_type = typename RiemannSolverView::primitive_type;

      using precomputed_type = typename View::precomputed_type;

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      //@}
      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystemView and a WaveSpeedEstimator
       * object as arguments
       */
      WaveSpeedEstimatorView(const View &view,
                             const WaveSpeedEstimator<ScalarNumber> &wse)
          : view_(view)
          , riemann_solver_view_(
                static_cast<const NASGRiemannSolver<ScalarNumber> &>(wse)
                    .template view<Number, MemorySpace>())
      {
      }

      /**
       * For two given 1D primitive states riemann_data_i and
       * riemann_data_j, compute an estimate for an upper bound of the
       * maximum wavespeed lambda.
       */
      DEAL_II_HOST_DEVICE Number
      compute(const primitive_type &riemann_data_i,
              const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of the maximum wavespeed
       * lambda.
       */
      DEAL_II_HOST_DEVICE Number
      compute(const PrecomputedVectorView &pv,
              const state_type &U_i,
              const state_type &U_j,
              const unsigned int i,
              const unsigned int *js,
              const dealii::Tensor<1, dim, Number> &n_ij) const;
      //@}

    protected:
      /**
       * @name Internal methods
       */
      //@{

      /**
       * For a given (2+dim dimensional) state vector <code>U</code>, and a
       * (normalized) "direction" n_ij, first compute the corresponding
       * projected state in the corresponding 1D Riemann problem, and then
       * compute and return the Riemann data [rho, u, p, a] (used in the
       * approximative Riemann solver).
       */
      DEAL_II_HOST_DEVICE primitive_type
      riemann_data_from_state(const state_type &U,
                              const Number &p,
                              const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const RiemannSolverView riemann_solver_view_;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      return riemann_solver_view_.compute(riemann_data_i, riemann_data_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const PrecomputedVectorView &pv,
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto &[p_i, unused_i, s_i, eta_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      const auto &[p_j, unused_j, s_j, eta_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      const auto riemann_data_i = riemann_data_from_state(U_i, p_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, p_j, n_ij);

      return compute(riemann_data_i, riemann_data_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::riemann_data_from_state(
        const state_type &U,
        const Number &p,
        const dealii::Tensor<1, dim, Number> &n_ij) const -> primitive_type
    {
      const auto rho = view_.density(U);
      const auto rho_inverse = ScalarNumber(1.0) / rho;

      const auto m = view_.momentum(U);
      const auto proj_m = n_ij * m;

      const auto gamma = view_.surrogate_gamma(U, p);

      const auto covolume_b = view_.eos_covolume_constant();
      const auto pinf = view_.eos_interpolation_pinfty();
      const auto x = Number(1.) - covolume_b * rho;
      const auto a = std::sqrt(gamma * (p + pinf) / (rho * x));

#ifdef DEBUG_EXPENSIVE_BOUNDS_CHECK
      AssertThrowSIMD(
          Number(p + pinf),
          [](auto val) { return val >= ScalarNumber(0.); },
          dealii::ExcMessage("Internal error: p + pinf < 0."));

      AssertThrowSIMD(
          x,
          [](auto val) { return val > ScalarNumber(0.); },
          dealii::ExcMessage("Internal error: 1. - b * rho <= 0."));

      AssertThrowSIMD(
          gamma,
          [](auto val) { return val >= ScalarNumber(1.); },
          dealii::ExcMessage("Internal error: gamma < 1."));
#endif

      return {{rho, proj_m * rho_inverse, p, gamma, a}};
    }
  } // namespace EulerAEOS
} // namespace ryujin
