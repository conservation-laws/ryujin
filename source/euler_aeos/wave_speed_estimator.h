//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "hyperbolic_system.h"

#include <gpu.h>
#include <newton.h>
#include <observer_pointer.h>
#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// #define DEBUG_WAVE_SPEED_ESTIMATOR

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
    class WaveSpeedEstimator : public dealii::ParameterAcceptor
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      /**
       * Alias for the view on the wave speed estimator for a given
       * dimension @p dim, choice of number type @p Number, and memory
       * space @p MemorySpace.
       */
      template <int dim,
                typename Number = double,
                typename MemorySpace = dealii::MemorySpace::Host>
      using View = WaveSpeedEstimatorView<dim, Number, MemorySpace>;

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
        return View<dim, Number, MemorySpace>{
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
       * Number of components in a primitive state, we store \f$[\rho, v,
       * p, a, gamma]\f$, thus, 5.
       */
      static constexpr unsigned int riemann_data_size = 5;

      /**
       * The array type to store the expanded primitive state for the
       * Riemann solver \f$[\rho, v, p, a]\f$
       */
      using primitive_type = std::array<Number, riemann_data_size>;

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
          const WaveSpeedEstimator<ScalarNumber> & /*wave_speed_estimator*/)
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

      //@}

    protected:
      /**
       * @name Internal methods
       */
      //@{

      /**
       * The function c(gamma) as defined in (A.3) of
       * @cite ClaytonGuermondPopov-2022, with a simplified cut-off for
       * gamma > 3.
       *
       * Cost: 0x pow, 1x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number c(const Number &gamma_Z) const;


      /**
       * The factor alpha = 2 a (1 - b rho) / (gamma - 1) used in the
       * two-rarefaction and shock-shock bounds of
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 0x pow, 1x division, 0x sqrt
       */
      DEAL_II_HOST_DEVICE Number alpha(const Number &rho,
                                       const Number &gamma,
                                       const Number &a) const;


#ifndef DOXYGEN
      /*
       * See @cite GuermondPopov2016b, page 912, (3.4), generalized to the
       * Noble-Abel stiffened gas equation of state, see
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 1x pow, 6x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number f(const primitive_type &riemann_data,
                                   const Number p_star) const;


      /*
       * See @cite GuermondPopov2016b, page 912, (3.3), generalized to the
       * Noble-Abel stiffened gas equation of state, see
       * @cite ClaytonGuermondPopov-2022.
       *
       * Cost: 2x pow, 12x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number phi(const primitive_type &riemann_data_i,
                                     const primitive_type &riemann_data_j,
                                     const Number p_in) const;
#endif


      /**
       * See @cite ClaytonGuermondPopov-2022
       *
       * The approximate Riemann solver is based on a function phi(p) that is
       * montone increasing in p, concave down and whose (weak) third
       * derivative is non-negative and locally bounded. Because we
       * actually do not perform any iteration for computing our wavespeed
       * estimate we can get away by only implementing a specialized
       * variant of the phi function that computes phi(p_max). It inlines
       * the implementation of the "f" function and eliminates all
       * unnecessary branches in "f".
       *
       * Cost: 0x pow, 4x division, 2x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      phi_of_p_max(const primitive_type &riemann_data_i,
                   const primitive_type &riemann_data_j) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.7)
       *
       * Cost: 0x pow, 2x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number lambda1_minus(
          const primitive_type &riemann_data, const Number p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.8)
       *
       * Cost: 0x pow, 2x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number lambda3_plus(
          const primitive_type &primitive_state, const Number p_star) const;


      /**
       * See @cite GuermondPopov2016b, page 912, (3.9)
       *
       * For two given primitive states <code>riemann_data_i</code> and
       * <code>riemann_data_j</code>, and a guess p_2, compute an upper bound
       * for lambda.
       *
       * Cost: 0x pow, 4x division, 2x sqrt (inclusive)
       */
      DEAL_II_HOST_DEVICE Number
      compute_lambda(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j,
                     const Number p_star) const;


      /**
       * Compute the best available, but expensive, upper bound on the
       * expansion-shock case as described in §5.4, Eqn. (5.7) and (5.8) in
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 5x pow, 11x division, 1x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      p_star_RS_full(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;


      /**
       * Compute the best available, but expensive, upper bound on the
       * shock-shock case as described in §5.5, Eqn. (5.10) and (5.12) in
       * @cite ClaytonGuermondPopov-2022
       *
       * Cost: 2x pow, 11x division, 5x sqrt (inclusive)
       */
      DEAL_II_HOST_DEVICE Number
      p_star_SS_full(const primitive_type &riemann_data_i,
                     const primitive_type &riemann_data_j) const;


      /*
       * Compute only the failsafe the failsafe bound for \f$\tilde
       * p_2^\ast\f$ (5.11) in @cite ClaytonGuermondPopov-2022
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */
      DEAL_II_HOST_DEVICE Number
      p_star_failsafe(const primitive_type &riemann_data_i,
                      const primitive_type &riemann_data_j) const;


      /*
       * Compute a simultaneous upper bound on (5.7) second formula for
       * \tilde p_2^\ast (5.8) first formula for \tilde p_1^\ast (5.11)
       * formula for \tilde p_2^\ast in @cite ClaytonGuermondPopov-2022
       *
       * Cost: 3x pow, 9x division, 2x sqrt
       *
       * @todo improve documentation
       */
      DEAL_II_HOST_DEVICE Number
      p_star_interpolated(const primitive_type &riemann_data_i,
                          const primitive_type &riemann_data_j) const;


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

      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    /*
     * The WaveSpeedEstimatorView is a guaranteed maximal wavespeed (GMS)
     * estimate for the extended Riemann problem outlined in
     * @cite ClaytonGuermondPopov-2022. For extenstions on handling negative
     * pressures, we follow @cite clayton2023robust (see §4.6).
     *
     * In contrast to the algorithm outlined in above reference the
     * algorithm takes a couple of shortcuts to significantly decrease the
     * computational footprint. These simplifications still guarantee that
     * we have an upper bound on the maximal wavespeed - but the number
     * bound might be larger. In particular:
     *
     *  - We do not check and treat the case phi(p_min) > 0. This
     *    corresponds to two expansion waves, see §5.2 in the reference. In
     *    this case we have
     *
     *      0 < p_star < p_min <= p_max.
     *
     *    And due to the fact that p_star < p_min the wavespeeds reduce to
     *    a left wavespeed v_L - a_L and right wavespeed v_R + a_R. This
     *    implies that it is sufficient to set p_2 to ANY value provided
     *    that p_2 <= p_min hold true in order to compute the correct
     *    wavespeed.
     *
     *    If p_2 > p_min then a more pessimistic bound is computed.
     *
     *  - FIXME: Simplification in p_star_RS
     */


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "rho_left: " << rho_i << std::endl;
      std::cout << "u_left: " << u_i << std::endl;
      std::cout << "p_left: " << p_i << std::endl;
      std::cout << "gamma_left: " << gamma_i << std::endl;
      std::cout << "a_left: " << a_i << std::endl;
      std::cout << "rho_right: " << rho_j << std::endl;
      std::cout << "u_right: " << u_j << std::endl;
      std::cout << "p_right: " << p_j << std::endl;
      std::cout << "gamma_right: " << gamma_j << std::endl;
      std::cout << "a_right: " << a_j << std::endl;
#endif

      const Number p_max = std::max(p_i, p_j) + pinf;
      const Number phi_p_max = phi_of_p_max(riemann_data_i, riemann_data_j);

      if (!view_.compute_strict_bounds()) {
#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
        const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);
        const Number p_strict =
            ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));
        std::cout << "   p^*_strict = " << p_strict << "\n";
        std::cout << "   phi(p_*_s) = "
                  << phi(riemann_data_i, riemann_data_j, p_strict) << "\n";
        std::cout << "-> lambda_str = "
                  << compute_lambda(riemann_data_i, riemann_data_j, p_strict)
                  << std::endl;
#endif

        const Number p_star_tilde =
            p_star_interpolated(riemann_data_i, riemann_data_j);
        const Number p_star_backup =
            p_star_failsafe(riemann_data_i, riemann_data_j);

        const Number p_2 =
            ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
                phi_p_max,
                Number(0.),
                std::min(p_star_tilde, p_star_backup),
                std::min(p_max, p_star_tilde));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
        std::cout << "   p^*_tilde  = " << p_2 << "\n";
        std::cout << "   phi(p_*_t) = "
                  << phi(riemann_data_i, riemann_data_j, p_2) << "\n";
        std::cout << "-> lambda_max = "
                  << compute_lambda(riemann_data_i, riemann_data_j, p_2)
                  << std::endl;
#endif

        return compute_lambda(riemann_data_i, riemann_data_j, p_2);
      }

      const Number p_star_RS = p_star_RS_full(riemann_data_i, riemann_data_j);
      const Number p_star_SS = p_star_SS_full(riemann_data_i, riemann_data_j);

      const Number p_2 =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              phi_p_max, Number(0.), p_star_SS, std::min(p_max, p_star_RS));

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "   p^*_tilde  = " << p_2 << "\n";
      std::cout << "   phi(p_*_t) = "
                << phi(riemann_data_i, riemann_data_j, p_2) << "\n";
      std::cout << "-> lambda_max = "
                << compute_lambda(riemann_data_i, riemann_data_j, p_2)
                << std::endl;
#endif

      return compute_lambda(riemann_data_i, riemann_data_j, p_2);
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
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::c(
        const Number &gamma) const
    {
      /*
       * We implement the continuous and monotonic function c(gamma) as
       * defined in (A.3) on page A469 of @cite ClaytonGuermondPopov-2022.
       * But with a simplified quick cut-off for the case gamma > 3:
       *
       *   c(gamma)^2 = 1                                    for gamma <= 5 / 3
       *   c(gamma)^2 = (3 * gamma + 11) / (6 * gamma + 6)   in between
       *   c(gamma)^2 = max(1/2, 5 / 6 - slope (gamma - 3))  for gamma > 3
       *
       * Due to the fact that the function is monotonic we can simply clip
       * the values without checking the conditions:
       */

      constexpr ScalarNumber slope =
          ScalarNumber(-0.34976871477801828189920753948709);

      const Number first_radicand = (ScalarNumber(3.) * gamma + Number(11.)) /
                                    (ScalarNumber(6.) * gamma + Number(6.));

      const Number second_radicand =
          Number(5. / 6.) + slope * (gamma - Number(3.));

      Number radicand = std::min(first_radicand, second_radicand);
      radicand = std::min(Number(1.), radicand);
      radicand = std::max(Number(1. / 2.), radicand);

      return std::sqrt(radicand);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::alpha(
        const Number &rho, const Number &gamma, const Number &a) const
    {
      const auto covolume_b = view_.eos_covolume_constant();

      const Number numerator =
          ScalarNumber(2.) * a * (Number(1.) - covolume_b * rho);

      const Number denominator = gamma - Number(1.);

      return safe_division(numerator, denominator);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::f(
        const primitive_type &riemann_data, const Number p_star) const
    {
      constexpr ScalarNumber min = std::numeric_limits<ScalarNumber>::min();

      const auto covolume_b = view_.eos_covolume_constant();
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho, u, p, gamma, a] = riemann_data;

      const Number one_minus_b_rho = Number(1.) - covolume_b * rho;
      const Number gamma_minus_one = gamma - Number(1.);

      const Number Az =
          ScalarNumber(2.) * one_minus_b_rho / (rho * (gamma + Number(1.)));

      const Number Bz = gamma_minus_one / (gamma + Number(1.)) * (p + pinf);

      const Number radicand = safe_division(Az, p_star + pinf + Bz);

      /* true_value is shock case */
      const Number true_value = (p_star - p) * std::sqrt(radicand);

      const auto exponent = ScalarNumber(0.5) * gamma_minus_one / gamma;

      const Number ratio = safe_division(p_star + pinf, p + pinf);
      const Number factor = ryujin::pow(ratio, exponent) - Number(1.);

      /* false_value is rarefaction case */
      const auto false_value = ScalarNumber(2.) * a * one_minus_b_rho * factor /
                               std::max(gamma_minus_one, Number(min));

      return ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_star, p, true_value, false_value);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::phi(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_in) const
    {
      const Number &u_i = riemann_data_i[1];
      const Number &u_j = riemann_data_j[1];

      return f(riemann_data_i, p_in) + f(riemann_data_j, p_in) + u_j - u_i;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::phi_of_p_max(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto covolume_b = view_.eos_covolume_constant();
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number p_max = std::max(p_i, p_j) + pinf;

      const Number radicand_inverse_i =
          safe_division(ScalarNumber(0.5) * rho_i,
                        Number(1.) - covolume_b * rho_i) *
          ((gamma_i + Number(1.)) * p_max +
           (gamma_i - Number(1.)) * (p_i + pinf));

      const Number value_i =
          safe_division(p_max - p_i, std::sqrt(radicand_inverse_i));

      const Number radicand_inverse_j =
          safe_division(ScalarNumber(0.5) * rho_j,
                        Number(1.) - covolume_b * rho_j) *
          ((gamma_j + Number(1.)) * p_max +
           (gamma_j - Number(1.)) * (p_j + pinf));

      const Number value_j =
          safe_division(p_max - p_j, std::sqrt(radicand_inverse_j));

      return value_i + value_j + u_j - u_i;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::lambda1_minus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor =
          ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;

      const Number tmp = safe_division(positive_part(p_star - p), p + pinf);

      return u - a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::lambda3_plus(
        const primitive_type &riemann_data, const Number p_star) const
    {
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho, u, p, gamma, a] = riemann_data;

      const auto factor =
          ScalarNumber(0.5) * (gamma + ScalarNumber(1.)) / gamma;

      const Number tmp = safe_division(positive_part(p_star - p), p + pinf);

      return u + a * std::sqrt(Number(1.) + factor * tmp);
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::compute_lambda(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j,
        const Number p_star) const
    {
      const Number nu_11 = lambda1_minus(riemann_data_i, p_star);
      const Number nu_32 = lambda3_plus(riemann_data_j, p_star);

      return std::max(positive_part(nu_32), negative_part(nu_11));
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::p_star_RS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      /*
       * First get p_min, p_max.
       *
       * Then, we get gamma_min/max, and alpha_min/max. Note that the
       * *_min/max values are associated with p_min/max and are not
       * necessarily the minimum/maximum of *_i vs *_j.
       */

      const Number p_min = std::min(p_i, p_j);
      const Number p_max = std::max(p_i, p_j);

      const Number gamma_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number numerator =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              p_max + pinf,
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_min + alpha_max - (u_j - u_i)));

      /*
       * The admissible set is p_min >= pinf. But numerically let's avoid
       * division by zero and ensure positivity:
       */
      const Number p_ratio = safe_division(p_min + pinf, p_max + pinf);

      /*
       * Here, we use a trick: The r-factor only shows up in the formula
       * for the case \gamma_min = \gamma_m, otherwise the r-factor
       * vanishes. We can accomplish this by using the following modified
       * exponent (where we substitute gamma_m by gamma_min):
       */
      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      /*
       * Compute (5.7) first formula for \tilde p_1^\ast and (5.8)
       * second formula for \tilde p_2^\ast at the same time:
       */

      const Number first_exponent =
          (gamma_M - Number(1.)) / (ScalarNumber(2.) * gamma_M);

      const Number first_exponent_inverse =
          safe_division(Number(1.), first_exponent);

      const Number first_denom =
          alpha_hat_min * ryujin::pow(p_ratio, r_exponent - first_exponent) +
          alpha_max;

      const Number p_1_tilde =
          (p_max + pinf) * ryujin::pow(safe_division(numerator, first_denom),
                                       first_exponent_inverse) -
          pinf;

      /*
       * Compute (5.7) second formula for \tilde p_2^\ast and (5.8) first
       * formula for \tilde p_1^\ast at the same time:
       */

      const Number second_exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);

      const Number second_exponent_inverse =
          safe_division(Number(1.), second_exponent);

      Number second_denom =
          alpha_hat_min * ryujin::pow(p_ratio, -second_exponent) +
          alpha_max * ryujin::pow(p_ratio, r_exponent);

      const Number p_2_tilde =
          (p_max + pinf) * ryujin::pow(safe_division(numerator, second_denom),
                                       second_exponent_inverse) -
          pinf;

      const Number p_star = std::min(p_1_tilde, p_2_tilde);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_RS_full = " << p_star << std::endl;
#endif
      return p_star;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::p_star_SS_full(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      const Number gamma_m = std::min(gamma_i, gamma_j);

      const Number alpha_hat_i = c(gamma_i) * alpha(rho_i, gamma_i, a_i);
      const Number alpha_hat_j = c(gamma_j) * alpha(rho_j, gamma_j, a_j);

      /*
       * Compute (5.10) formula for \tilde p_1^\ast:
       *
       * Cost: 2x pow, 4x division, 0x sqrt
       */

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              p_j + pinf,
              Number(0.),
              Number(0.),
              positive_part(alpha_hat_i + alpha_hat_j - (u_j - u_i)));

      const Number denominator =
          alpha_hat_i *
              ryujin::pow(safe_division(p_i + pinf, p_j + pinf), -exponent) +
          alpha_hat_j;

      const Number p_1_tilde =
          (p_j + pinf) * ryujin::pow(safe_division(numerator, denominator),
                                     exponent_inverse) -
          pinf;

      const auto p_2_tilde = p_star_failsafe(riemann_data_i, riemann_data_j);

      const Number p_star = std::min(p_1_tilde, p_2_tilde);

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_SS_full = " << p_star << std::endl;
#endif
      return p_star;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::p_star_failsafe(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto covolume_b = view_.eos_covolume_constant();
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;

      /*
       * Compute (5.11) formula for \tilde p_2^\ast:
       *
       * Cost: 0x pow, 3x division, 3x sqrt
       */

      const Number p_max = std::max(p_i, p_j) + pinf;

      const Number radicand_i = safe_division(
          ScalarNumber(2.) * (Number(1.) - covolume_b * rho_i) * p_max,
          rho_i * ((gamma_i + Number(1.)) * p_max +
                   (gamma_i - Number(1.)) * (p_i + pinf)));

      const Number x_i = std::sqrt(radicand_i);

      const Number radicand_j = safe_division(
          ScalarNumber(2.) * (Number(1.) - covolume_b * rho_j) * p_max,
          rho_j * ((gamma_j + Number(1.)) * p_max +
                   (gamma_j - Number(1.)) * (p_j + pinf)));

      const Number x_j = std::sqrt(radicand_j);

      const Number a = x_i + x_j;
      const Number b =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::equal>(
              a, Number(0.), Number(0.), u_j - u_i);

      const Number c = -(p_i + pinf) * x_i - (p_j + pinf) * x_j;

      const Number base = safe_division(
          std::abs(-b +
                   std::sqrt(positive_part(b * b - ScalarNumber(4.) * a * c))),
          std::abs(ScalarNumber(2.) * a));

      const Number p_2_tilde = base * base - pinf;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_failsafe = " << p_2_tilde << std::endl;
#endif
      return p_2_tilde;
    }


    template <int dim, typename Number, typename MemorySpace>
    DEAL_II_HOST_DEVICE_ALWAYS_INLINE Number
    WaveSpeedEstimatorView<dim, Number, MemorySpace>::p_star_interpolated(
        const primitive_type &riemann_data_i,
        const primitive_type &riemann_data_j) const
    {
      const auto pinf = view_.eos_interpolation_pinfty();

      const auto &[rho_i, u_i, p_i, gamma_i, a_i] = riemann_data_i;
      const auto &[rho_j, u_j, p_j, gamma_j, a_j] = riemann_data_j;
      const auto alpha_i = alpha(rho_i, gamma_i, a_i);
      const auto alpha_j = alpha(rho_j, gamma_j, a_j);

      /*
       * First get p_min, p_max.
       *
       * Then, we get gamma_min/max, and alpha_min/max. Note that the
       * *_min/max values are associated with p_min/max and are not
       * necessarily the minimum/maximum of *_i vs *_j.
       */

      const Number p_min = std::min(p_i, p_j) + pinf;
      const Number p_max = std::max(p_i, p_j) + pinf;

      const Number gamma_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, gamma_i, gamma_j);

      const Number alpha_min =
          ryujin::compare_and_apply_mask<dealii::SIMDComparison::less_than>(
              p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_min = c(gamma_min) * alpha_min;

      const Number gamma_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, gamma_i, gamma_j);

      const Number alpha_max = ryujin::compare_and_apply_mask<
          dealii::SIMDComparison::greater_than_or_equal>(
          p_i, p_j, alpha_i, alpha_j);

      const Number alpha_hat_max = c(gamma_max) * alpha_max;

      const Number gamma_m = std::min(gamma_i, gamma_j);
      const Number gamma_M = std::max(gamma_i, gamma_j);

      const Number p_ratio = safe_division(p_min, p_max);

      /*
       * Here, we use a trick: The r-factor only shows up in the formula
       * for the case \gamma_min = \gamma_m, otherwise the r-factor
       * vanishes. We can accomplish this by using the following modified
       * exponent (where we substitute gamma_m by gamma_min):
       */
      const Number r_exponent =
          (gamma_M - gamma_min) / (ScalarNumber(2.) * gamma_min * gamma_M);

      /*
       * Compute a simultaneous upper bound on
       *   (5.7) second formula for \tilde p_2^\ast
       *   (5.8) first formula for \tilde p_1^\ast
       *   (5.11) formula for \tilde p_2^\ast
       */

      const Number exponent =
          (gamma_m - Number(1.)) / (ScalarNumber(2.) * gamma_m);
      const Number exponent_inverse = Number(1.) / exponent;

      const Number numerator =
          positive_part(alpha_hat_min + /*SIC!*/ alpha_max - (u_j - u_i));

      Number denominator = alpha_hat_min * ryujin::pow(p_ratio, -exponent) +
                           alpha_hat_max * ryujin::pow(p_ratio, r_exponent);

      const auto temp = safe_division(numerator, denominator);

      const Number p_tilde = p_max * ryujin::pow(temp, exponent_inverse) - pinf;

#ifdef DEBUG_WAVE_SPEED_ESTIMATOR
      std::cout << "p_star_interpolated = " << p_tilde << std::endl;
#endif
      return p_tilde;
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
