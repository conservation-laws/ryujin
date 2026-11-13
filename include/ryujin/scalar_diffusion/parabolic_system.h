//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/base/convenience_macros.h>
#include <ryujin/base/simd.h>
#include <ryujin/interface/parabolic_system.h>

#include <deal.II/base/parameter_acceptor.h>

namespace ryujin
{
  namespace ScalarDiffusion
  {
    /**
     * A power-law diffusion model for a scalar quantity \f$u\f$,
     * \f{align}
     *   \partial_t u = \nabla\cdot\big(k(u)\nabla u\big),
     *   \qquad k(u) = \nu\,|u|^m,
     * \f}
     * with a diffusion coefficient \f$\nu\ge0\f$ and an exponent
     * \f$m\ge0\f$. The choice \f$m=0\f$ recovers linear diffusion; for
     * \f$m>0\f$ the model is the porous medium equation
     * \f$\partial_t u = \tfrac{\nu}{m+1}\Delta(|u|^m u)\f$, see
     * @cite Vazquez2007. Paired with a scalar conservation equation the
     * model describes a (degenerate) convection-diffusion equation
     * @cite Carrillo1999.
     *
     * @ingroup ScalarDiffusionEquations
     */
    class ParabolicSystem final : public Interface::ParabolicSystem
    {
    public:
      /**
       * Constructor.
       */
      ParabolicSystem(const std::string &subsection = "/B - Equation");

      /**
       * The diffusion coefficient \f$\nu\f$.
       */
      ACCESSOR_READ_ONLY(diffusion_coefficient)

      /**
       * The exponent \f$m\f$ of the power law.
       */
      ACCESSOR_READ_ONLY(exponent)

      /**
       * Whether the diffusivity is independent of the state.
       */
      bool is_linear() const
      {
        return exponent_ == 0.;
      }

      /**
       * For a given state @p u return the diffusivity
       * \f$k(u) = \nu\,|u|^m\f$.
       */
      template <typename Number>
      Number diffusivity(const Number &u) const;

    private:
      /**
       * @name Run time options
       */
      //@{

      double diffusion_coefficient_;
      double exponent_;

      //@}
    }; /* ParabolicSystem */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : Interface::ParabolicSystem(
              "Power-law diffusion model",
              /* no components of its own: acts on the hyperbolic state */
              {},
              {},
              subsection)
    {
      diffusion_coefficient_ = 1.e-2;
      add_parameter("diffusion coefficient",
                    diffusion_coefficient_,
                    "The diffusion coefficient nu of the diffusivity k(u) = "
                    "nu |u|^m");

      exponent_ = 0.;
      add_parameter("diffusion exponent",
                    exponent_,
                    "The exponent m of the diffusivity k(u) = nu |u|^m");
    }


    template <typename Number>
    inline Number ParabolicSystem::diffusivity(const Number &u) const
    {
      using ScalarNumber = typename get_value_type<Number>::type;

      const auto nu = ScalarNumber(diffusion_coefficient_);
      if (is_linear())
        return Number(nu);

      return nu * ryujin::pow(std::abs(u), ScalarNumber(exponent_));
    }
  } // namespace ScalarDiffusion
} // namespace ryujin
