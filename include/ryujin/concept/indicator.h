//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/concept/hyperbolic_system.h>

#include <deal.II/base/tensor.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    /**
     * The view of an indicator.
     *
     * @ingroup Concepts
     */
    template <typename V, int dim, typename Number>
    concept IndicatorView =
        requires(V view,
                 const V &const_view,
                 const typename V::PrecomputedVectorView &precomputed,
                 const typename V::state_type &U,
                 const typename V::flux_contribution_type &flux,
                 const dealii::Tensor<1, dim, Number> &c_ij,
                 const unsigned int i,
                 const unsigned int *js,
                 const Number hd_i) {
          view.reset(precomputed, i, U, flux);
          view.accumulate(precomputed, js, U, flux, c_ij);
          {
            const_view.alpha(hd_i)
          } -> std::same_as<Number>;
        };


    /**
     * An indicator: constructed from the hyperbolic system and the
     * subsection of its run time parameters, it hands out views.
     *
     * @ingroup Concepts
     */
    template <typename T, typename System, int dim, typename Number>
    concept Indicator =
        std::constructible_from<T, const System &, const std::string &> &&
        IndicatorView<view_type<T, dim, Number>, dim, Number>;
  } // namespace Concept
} // namespace ryujin
