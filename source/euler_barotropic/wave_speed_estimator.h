//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <gpu.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// #define DEBUG_WAVE_SPEED_ESTIMATOR

namespace ryujin
{
  namespace EulerBarotropic
  {
    template <int dim,
              typename Number = double,
              typename MemorySpace = dealii::MemorySpace::Host>
    class WaveSpeedEstimatorView;

    /**
     * Specialized approximative solver for the 1D Riemann problem of the
     * barotropic Euler equations. The solver ensures that the estimate
     * \f$\lambda_{\text{max}}\f$ that is returned by compute() is a
     * guaranteed upper bound of the maximal wavespeed.
     *
     * @ingroup EulerEquations
     */
    template <typename ScalarNumber = double>
    class WaveSpeedEstimator : public dealii::ParameterAcceptor
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
          : ParameterAcceptor(subsection)
          , hyperbolic_system_(&hyperbolic_system)
      {
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

      static constexpr auto problem_dimension = View::problem_dimension;

      using state_type = typename View::state_type;

      /**
       * Number of components in a primitive state, we store \f$[v, a]\f$.
       */
      static constexpr unsigned int riemann_data_size = 2;

      /**
       * The array type to store the primitive state for the Riemann solver
       * \f$[v, a]\f$
       */
      using primitive_type = typename std::array<Number, riemann_data_size>;

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
                             const WaveSpeedEstimator<ScalarNumber> & /*wse*/)
          : view_(view)
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

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;

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
      const auto &[u_i, a_i] = riemann_data_i;
      const auto &[u_j, a_j] = riemann_data_j;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number lambda_max =
          std::max(std::abs(u_i) + a_i, std::abs(u_j) + a_j);
      return lambda_max;
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
      const auto &[e_i, p_i, a_i] =
          pv.template read_tensor<Number, precomputed_type>(i);

      const auto &[e_j, p_j, a_j] =
          pv.template read_tensor<Number, precomputed_type>(js);

      const auto rho_i = view_.density(U_i);
      const auto rho_i_inverse = Number(1.0) / rho_i;
      const auto m_i = view_.momentum(U_i);
      const auto u_i = rho_i_inverse * n_ij * m_i;

      const auto rho_j = view_.density(U_j);
      const auto rho_j_inverse = Number(1.0) / rho_j;
      const auto m_j = view_.momentum(U_j);
      const auto u_j = rho_j_inverse * n_ij * m_j;

      return compute(primitive_type{u_i, a_i}, primitive_type{u_j, a_j});
    }
  } // namespace EulerBarotropic
} // namespace ryujin
