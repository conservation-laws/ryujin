//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
//

#pragma once

#include "hyperbolic_system.h"

#include <simd.h>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    template <typename ScalarNumber = double>
    class RiemannSolverParameters : public dealii::ParameterAcceptor
    {
    public:
      RiemannSolverParameters(const std::string &subsection = "/RiemannSolver")
          : ParameterAcceptor(subsection)
      {
        use_greedy_wavespeed_ = false;
        add_parameter("use greedy wavespeed",
                      use_greedy_wavespeed_,
                      "Use a greedy wavespeed estimate instead of a guaranteed "
                      "upper bound "
                      "on the maximal wavespeed (for convex fluxes).");

        use_averaged_entropy_ = false;
        add_parameter("use averaged entropy",
                      use_averaged_entropy_,
                      "In addition to the wavespeed estimate based on the Roe "
                      "average and "
                      "flux gradients of the left and right state also enforce "
                      "an entropy "
                      "inequality on the averaged Krŭzkov entropy.");

        random_entropies_ = 0;
        add_parameter(
            "random entropies",
            random_entropies_,
            "In addition to the wavespeed estimate based on the Roe average "
            "and "
            "flux gradients of the left and right state also enforce an "
            "entropy "
            "inequality on the prescribed number of random Krŭzkov entropies.");
      }

      ACCESSOR_READ_ONLY(use_greedy_wavespeed);
      ACCESSOR_READ_ONLY(use_averaged_entropy);
      ACCESSOR_READ_ONLY(random_entropies);

    private:
      bool use_greedy_wavespeed_;
      bool use_averaged_entropy_;
      unsigned int random_entropies_;
    };


    /**
     * A fast estimate for a sufficient maximal wavespeed of the 1D Riemann
     * problem. The wavespeed estimate is based on a guaranteed upper bound
     * on the maximal wavespeed for convex fluxes, see Example 79.17 on
     * page 333 of @cite GuermondErn2021. As well as an augmented "Roe
     * average" based on an entropy inequality of a suitable Krŭzkov
     * entropy, see @cite ryujin-2023-5 Section 4.
     *
     * @ingroup ScalarConservationEquations
     */
    template <int dim, typename Number = double>
    class RiemannSolver
    {
    public:
      /**
       * @copydoc HyperbolicSystemView
       */
      using HyperbolicSystemView = HyperbolicSystemView<dim, Number>;

      /**
       * @copydoc HyperbolicSystemView::state_type
       */
      using state_type = typename HyperbolicSystemView::state_type;

      /**
       * @copydoc HyperbolicSystemView::n_precomputed_values
       */
      static constexpr unsigned int n_precomputed_values =
          HyperbolicSystemView::n_precomputed_values;

      /**
       * @copydoc HyperbolicSystemView::precomputed_state_type
       */
      using precomputed_state_type =
          typename HyperbolicSystemView::precomputed_state_type;

      /**
       * @copydoc HyperbolicSystemView::ScalarNumber
       */
      using ScalarNumber = typename get_value_type<Number>::type;

      /**
       * @copydoc RiemannSolverParameters
       */
      using Parameters = RiemannSolverParameters<ScalarNumber>;

      /**
       * @name Compute wavespeed estimates
       */
      //@{

      /**
       * Constructor taking a HyperbolicSystem instance as argument
       */
      RiemannSolver(
          const HyperbolicSystem &hyperbolic_system,
          const Parameters &parameters,
          const MultiComponentVector<ScalarNumber, n_precomputed_values>
              &precomputed_values)
          : hyperbolic_system(hyperbolic_system)
          , parameters(parameters)
          , precomputed_values(precomputed_values)
      {
      }

      /**
       * For two states @p u_i, @p u_j, precomputed values @p prec_i,
       * @p prec_j, and a (normalized) "direction" n_ij
       * compute an upper bound estimate for the wavespeed.
       */
      Number compute(const Number &u_i,
                     const Number &u_j,
                     const precomputed_state_type &prec_i,
                     const precomputed_state_type &prec_j,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

      /**
       * For two given states U_i a U_j and a (normalized) "direction" n_ij
       * compute an estimate for an upper bound of lambda.
       */
      Number compute(const state_type &U_i,
                     const state_type &U_j,
                     const unsigned int i,
                     const unsigned int *js,
                     const dealii::Tensor<1, dim, Number> &n_ij) const;

    private:
      const HyperbolicSystem &hyperbolic_system;
      const Parameters &parameters;

      const MultiComponentVector<ScalarNumber, n_precomputed_values>
          &precomputed_values;
      //@}
    };


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */

    template <int dim, typename Number>
    DEAL_II_ALWAYS_INLINE inline Number RiemannSolver<dim, Number>::compute(
        const state_type &U_i,
        const state_type &U_j,
        const unsigned int i,
        const unsigned int *js,
        const dealii::Tensor<1, dim, Number> &n_ij) const
    {
      const auto view = hyperbolic_system.view<dim, Number>();

      using pst = typename HyperbolicSystemView::precomputed_state_type;

      const auto u_i = view.state(U_i);
      const auto u_j = view.state(U_j);

      const auto &pv = precomputed_values;
      const auto prec_i = pv.template get_tensor<Number, pst>(i);
      const auto prec_j = pv.template get_tensor<Number, pst>(js);

      return compute(u_i, u_j, prec_i, prec_j, n_ij);
    }


  } // namespace ScalarConservation
} // namespace ryujin
