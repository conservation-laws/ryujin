//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "../euler_aeos/nasg_riemann_solver.h"
#include "hyperbolic_system.h"

#include <gpu.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>


namespace ryujin
{
  namespace Euler
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class WaveSpeedEstimatorView;

    /**
     * Compile time options of the Riemann solver used in the
     * WaveSpeedEstimator: a polytropic gas equation of state with a
     * single, constant gamma.
     */
    inline constexpr EulerAEOS::NASGRiemannSolverOptions
        polytropic_gas_riemann_solver_options{.covolume = false,
                                              .pinf = false,
                                              .safe_division = false,
                                              .variable_gamma = false};

    /**
     * A specialized version of the NASG Riemann Solver for polytropic gas
     */
    template <typename ScalarNumber>
    using PGRiemannSolver =
        EulerAEOS::NASGRiemannSolver<ScalarNumber,
                                     polytropic_gas_riemann_solver_options>;

    /**
     * A fast approximative solver for the 1D Riemann problem. The solver
     * ensures that the estimate \f$\lambda_{\text{max}}\f$ that is returned
     * for the maximal wavespeed is a strict upper bound.
     *
     * The solver is based on @cite GuermondPopov2016b and
     * @cite ClaytonGuermondPopov-2022 and uses the
     * EulerAEOS::NASGRiemannSolver specialized for a polytropic gas
     * equation of state.
     *
     * @ingroup EulerEquations
     */
    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : protected PGRiemannSolver<ScalarNumber>
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
          : PGRiemannSolver<ScalarNumber>(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
        /* Transfer all equation of state parameters to the Riemann solver. */

        const auto update_parameters = [this] {
          const auto view =
              hyperbolic_system_->template view<1, ScalarNumber>();
          this->set_gamma(view.gamma());
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
            hyperbolic_system_->template view<dim, Number, MemorySpace>(),
            *this};
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

      using View = HyperbolicSystemView<dim, Number, MemorySpace>;

      using ScalarNumber = typename View::ScalarNumber;

      using state_type = typename View::state_type;

      /**
       * The view on the Riemann solver used for computing the wavespeed
       * estimate.
       */
      using RiemannSolverView = EulerAEOS::NASGRiemannSolverView<
          Number,
          polytropic_gas_riemann_solver_options,
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
                static_cast<const PGRiemannSolver<ScalarNumber> &>(wse)
                    .template view<Number, MemorySpace>())
      {
      }

      /**
       * For two given 1D primitive states riemann_data_i and riemann_data_j,
       * compute an estimation of an upper bound for the maximum wavespeed
       * lambda.
       */
      DEAL_II_HOST_DEVICE Number
      compute(const primitive_type &riemann_data_i,
              const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimation of an upper bound for lambda.
       *
       * Returns a tuple consisting of lambda max and the number of Newton
       * iterations used in the solver to find it.
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
       * compute and return the Riemann data [rho, u, p, gamma, a] (used in the
       * approximative Riemann solver).
       */
      DEAL_II_HOST_DEVICE primitive_type
      riemann_data_from_state(const state_type &U,
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
        const PrecomputedVectorView & /*pv*/,
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int /*i*/,
        const unsigned int * /*js*/,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto riemann_data_i = riemann_data_from_state(U_i, n_ij);
      const auto riemann_data_j = riemann_data_from_state(U_j, n_ij);

      return compute(riemann_data_i, riemann_data_j);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE auto
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::riemann_data_from_state(
        const state_type &U, const dealii::Tensor<1, dim, Number> &n_ij) const
        -> primitive_type
    {
      const auto rho = view_.density(U);
      const auto rho_inverse = Number(1.0) / rho;

      const auto m = view_.momentum(U);
      const auto proj_m = n_ij * m;
      const auto perp = m - proj_m * n_ij;

      const auto E = view_.total_energy(U) -
                     Number(0.5) * perp.norm_square() * rho_inverse;

      /*
       * Compute the pressure and speed of sound of the projected
       * one-dimensional state [rho, proj_m, E]:
       */
      const auto gamma = view_.gamma();
      const auto internal_energy =
          E - ScalarNumber(0.5) * (proj_m * proj_m) * rho_inverse;
      const auto p = (gamma - ScalarNumber(1.)) * internal_energy;
      const auto a = std::sqrt(gamma * p * rho_inverse);

      return {{rho, proj_m * rho_inverse, p, Number(gamma), a}};
    }
  } // namespace Euler
} // namespace ryujin
