//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <compile_time_options.h>
#include <multicomponent_vector.h>
#include <newton.h>
#include <simd.h>

namespace ryujin
{
  namespace Skeleton
  {
    template <typename ScalarNumber = double>
    class LimiterParameters : public dealii::ParameterAcceptor
    {
    public:
      LimiterParameters(const std::string &subsection = "/Limiter")
          : ParameterAcceptor(subsection)
      {
        iterations_ = 2;
        add_parameter(
            "iterations", iterations_, "Number of limiter iterations");
      }

      ACCESSOR_READ_ONLY(iterations);

    private:
      unsigned int iterations_;
    };


    /**
     * The convex limiter.
     *
     * @ingroup SkeletonEquations
     */
    template <int dim, typename Number = double>
    class Limiter
    {
    public:
      /**
       * @name Typedefs and constexpr constants
       */
      //@{

      using View = HyperbolicSystemView<dim, Number>;

      using ScalarNumber = typename View::ScalarNumber;

      using state_type = typename View::state_type;

      using flux_contribution_type = typename View::flux_contribution_type;

      using PrecomputedVector = typename View::PrecomputedVector;

      using Parameters = LimiterParameters<ScalarNumber>;

      //@}
      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * Limiter<dim, Number> limiter;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter.reset(i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter.accumulate(
       *       js, U_j, flux_j, scaled_c_ij, beta_ij, affine_shift);
       *   }
       *   limiter.bounds(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * The number of stored entries in the bounds array.
       */
      static constexpr unsigned int n_bounds = 0;

      /**
       * Array type used to store accumulated bounds.
       */
      using Bounds = std::array<Number, n_bounds>;

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      Limiter(const HyperbolicSystem &hyperbolic_system,
              const Parameters &parameters,
              const PrecomputedVector &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * Reset temporary storage
       */
      void reset(const unsigned int /*i*/,
                 const state_type & /*new_U_i*/,
                 const flux_contribution_type & /*new_flux_i*/)
      {
        // empty
      }

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const unsigned int * /*js*/,
                      const state_type & /*U_j*/,
                      const flux_contribution_type & /*flux_j*/,
                      const dealii::Tensor<1, dim, Number> & /*scaled_c_ij*/,
                      const Number & /*beta_ij*/,
                      const state_type & /*affine_shift*/)
      {
        // empty
      }

      /**
       * Return the computed bounds (with relaxation applied).
       */
      Bounds bounds(const Number /*hd_i*/) const
      {
        auto relaxed_bounds = bounds_;

        return relaxed_bounds;
      }

      /**
       * Given two bounds bounds_left, bounds_right, this function computes
       * a larger, combined Bounds set that this is a (convex) superset of
       * the two.
       */
      static Bounds combine_bounds(const Bounds & /*bounds_left*/,
                                   const Bounds & /*bounds_right*/)
      {
        return Bounds{};
      }

      //*}
      /** @name Convex limiter */
      //@{

      /**
       * Given a state \f$\mathbf U\f$ and an update \f$\mathbf P\f$ this
       * function computes and returns the maximal coefficient \f$t\f$,
       * obeying \f$t_{\text{min}} < t < t_{\text{max}}\f$, such that the
       * selected local minimum principles are obeyed.
       */
      std::tuple<Number, bool> limit(const Bounds & /*bounds*/,
                                     const state_type & /*U*/,
                                     const state_type & /*P*/,
                                     const Number /*t_min*/ = Number(0.),
                                     const Number t_max = Number(1.))
      {
        return {t_max, true};
      }

      //*}
      /**
       * @name Verify invariant domain property
       */
      //@{

      /**
       * Returns whether the state @p U is located in the invariant domain
       * described by @p bounds. If @p U is a vectorized state then the
       * function returns true if all vectorized values are located in the
       * invariant domain.
       */
      static bool
      is_in_invariant_domain(const HyperbolicSystem & /*hyperbolic_system*/,
                             const Bounds & /*bounds*/,
                             const state_type & /*U*/)
      {
        return true;
      }

    private:
      //@}
      /** @name Arguments and internal fields */
      //@{

      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;
      const PrecomputedVector &precomputed_values;

      Bounds bounds_;
      //@}
    };
  } // namespace Skeleton
} // namespace ryujin
