//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/concept/hyperbolic_system.h>

#include <deal.II/base/tensor.h>

#include <array>
#include <concepts>
#include <string>
#include <tuple>

namespace ryujin
{
  namespace Concept
  {
    /**
     * The view of a limiter.
     *
     * @ingroup Concepts
     */
    template <typename V, int dim, typename Number>
    concept LimiterView =
        std::same_as<typename V::Bounds, std::array<Number, V::n_bounds>> &&
        requires(V view,
                 const V &const_view,
                 const typename V::PrecomputedVectorView &precomputed,
                 const typename V::state_type &U,
                 const typename V::flux_contribution_type &flux,
                 const typename V::Bounds &bounds,
                 const dealii::Tensor<1, dim, Number> &scaled_c_ij,
                 const unsigned int i,
                 const unsigned int *js,
                 const Number hd_i) {
          {
            const_view.iterations()
          } -> std::convertible_to<unsigned int>;

          /* Used by HyperbolicModule::step(): */
          view.reset(precomputed, i, U, flux);
          view.accumulate(precomputed, js, U, flux, scaled_c_ij, U);
          {
            const_view.bounds(hd_i)
          } -> std::same_as<typename V::Bounds>;
          {
            const_view.limit(bounds, U, U)
          } -> std::same_as<std::tuple<Number, bool>>;

          /* Used by SolutionTransfer: */
          {
            const_view.projection_bounds_from_state(precomputed, i, U)
          } -> std::same_as<typename V::Bounds>;
          {
            const_view.combine_bounds(bounds, bounds)
          } -> std::same_as<typename V::Bounds>;
          {
            const_view.fully_relax_bounds(bounds, hd_i)
          } -> std::same_as<typename V::Bounds>;
        };


    /**
     * A limiter: constructed from the hyperbolic system and the subsection
     * of its run time parameters, it hands out views.
     *
     * @ingroup Concepts
     */
    template <typename T, typename System, int dim, typename Number>
    concept Limiter =
        std::constructible_from<T, const System &, const std::string &> &&
        LimiterView<view_type<T, dim, Number>, dim, Number>;
  } // namespace Concept
} // namespace ryujin
