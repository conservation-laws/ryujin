//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

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
    class Limiter : public dealii::ParameterAcceptor
    {
    public:
      Limiter(const std::string &subsection = "/Limiter")
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
    class LimiterView
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

      using PrecomputedVectorView = typename View::PrecomputedVectorView;

      using Parameters = Limiter<ScalarNumber>;

      //@}
      /**
       * @name Computation and manipulation of bounds
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
      LimiterView(const HyperbolicSystem &hyperbolic_system,
                  const Parameters &parameters)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
      {
      }

      /**
       * Given a state @p U_i and an index @p i return "strict" bounds,
       * i.e., a minimal convex set containing the state.
       */
      Bounds projection_bounds_from_state(const PrecomputedVectorView & /*pv*/,
                                          const unsigned int /*i*/,
                                          const state_type & /*U_i*/) const
      {
        return Bounds{};
      }

      /**
       * Given two bounds bounds_left, bounds_right, this function computes
       * a larger, combined set of bounds that this is a (convex) superset
       * of the two.
       */
      Bounds combine_bounds(const Bounds & /*bounds_left*/,
                            const Bounds & /*bounds_right*/) const
      {
        return Bounds{};
      }

      /**
       * This function applies a relaxation to a given a (strict) bound @p
       * bounds using a non dimensionalized measure @p hd (that should
       * scale as $h^d$, where $h$ is the local mesh size).
       */
      Bounds fully_relax_bounds(const Bounds & /*bounds*/,
                                const Number & /*hd*/) const
      {
        return Bounds{};
      }

      //@}
      /**
       * @name Stencil-based computation of bounds
       *
       * Intended usage:
       * ```
       * LimiterView<dim, Number> limiter_view;
       * for (unsigned int i = n_internal; i < n_owned; ++i) {
       *   // ...
       *   limiter_view.reset(pv, i, U_i, flux_i);
       *   for (unsigned int col_idx = 1; col_idx < row_length; ++col_idx) {
       *     // ...
       *     limiter_view.accumulate(pv, js, U_j, flux_j, scaled_c_ij,
       * affine_shift);
       *   }
       *   limiter_view.bounds(hd_i);
       * }
       * ```
       */
      //@{

      /**
       * Reset temporary storage
       */
      void reset(const PrecomputedVectorView & /*pv*/,
                 const unsigned int /*i*/,
                 const state_type & /*new_U_i*/,
                 const flux_contribution_type & /*new_flux_i*/)
      {
        // empty
      }

      /**
       * When looping over the sparsity row, add the contribution associated
       * with the neighboring state U_j.
       */
      void accumulate(const PrecomputedVectorView & /*pv*/,
                      const unsigned int * /*js*/,
                      const state_type & /*U_j*/,
                      const flux_contribution_type & /*flux_j*/,
                      const dealii::Tensor<1, dim, Number> & /*scaled_c_ij*/,
                      const state_type & /*affine_shift*/)
      {
        // empty
      }

      /**
       * Return the computed bounds (with relaxation applied).
       */
      Bounds bounds(const Number hd_i) const
      {
        auto relaxed_bounds = fully_relax_bounds(bounds_, hd_i);

        return relaxed_bounds;
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
                                     const Number t_max = Number(1.)) const
      {
        return {t_max, true};
      }

    private:
      //@}
      /** @name Arguments and internal fields */
      //@{

      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;

      Bounds bounds_;
      //@}
    };
  } // namespace Skeleton
} // namespace ryujin
