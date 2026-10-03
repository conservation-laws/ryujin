//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// #define DEBUG_WAVE_SPEED_ESTIMATOR

namespace ryujin
{
  namespace EulerBarotropic
  {
    template <int dim, typename Number = double>
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
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * Alias for the view on the wave speed estimator for a given dimension @p
       * dim and choice of number type @p Number.
       */
      template <int dim, typename Number = double>
      using View = WaveSpeedEstimatorView<dim, Number>;

      //@}
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

      //@}
      /**
       * @name Information and statistics
       */
      //@{

      /**
       * Return a view on the WaveSpeedEstimator for a given dimension @p dim
       * and choice of number type @p Number (which can be a scalar float, or
       * double, as well as a VectorizedArray holding packed scalars).
       */
      template <int dim, typename Number>
      auto view() const
      {
        return View<dim, Number>{
            hyperbolic_system_->template view<dim, Number>(), *this};
      }

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      dealii::ObserverPointer<const HyperbolicSystem> hyperbolic_system_;

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
    template <int dim, typename Number>
    class WaveSpeedEstimatorView
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

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
      WaveSpeedEstimatorView(
          const View &view,
          const WaveSpeedEstimator<ScalarNumber> &wave_speed_estimator)
          : view_(view)
          , wave_speed_estimator_(wave_speed_estimator)
      {
      }

      /**
       * For two given 1D primitive states riemann_data_i and
       * riemann_data_j, compute an estimate for an upper bound of the
       * maximum wavespeed lambda.
       */
      Number compute(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of the maximum wavespeed
       * lambda.
       */
      Number compute(const PrecomputedVectorView &pv,
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

    private:
      //@}
      /**
       * @name Internal data
       */
      //@{

      const View view_;
      const WaveSpeedEstimator<ScalarNumber> &wave_speed_estimator_;

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    WaveSpeedEstimatorView<dim, Number>::compute(
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


    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number
    WaveSpeedEstimatorView<dim, Number>::compute(
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
