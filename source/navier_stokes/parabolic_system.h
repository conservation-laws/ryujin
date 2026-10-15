//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <convenience_macros.h>

#include <ryujin/interface/parabolic_system.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/tensor.h>

#include <array>
#include <functional>
#include <iomanip>

namespace ryujin
{
  namespace NavierStokes
  {
    /**
     * A Newtonian  fluid viscosity model with a heat-flux governed by
     * Fourier's law. This class describes the parabolic system part of the
     * combined compressible Navier-Stokes equations.
     *
     * @ingroup NavierStokesEquations
     */
    class ParabolicSystem final : public Interface::ParabolicSystem
    {
    public:
      /**
       * Constructor.
       */
      ParabolicSystem(const std::string &subsection = "/B - Equation");

      /**
       * Whether the parabolic system is the identity.
       */
      static constexpr bool is_identity = false;

      ACCESSOR_READ_ONLY(mu)
      ACCESSOR_READ_ONLY(lambda)
      ACCESSOR_READ_ONLY(cv_inverse_kappa)

    private:
      /**
       * @name Run time options
       */
      //@{


      double mu_;
      double lambda_;
      double cv_inverse_kappa_;


      //@}
    }; /* ParabolicSystem */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : Interface::ParabolicSystem(
              "Newtonian fluid viscosity model with Fourier-law heat flux",
              /* no components of its own: acts on the hyperbolic state */
              {},
              {},
              subsection)
    {
      mu_ = 1.e-3;
      add_parameter("mu", mu_, "The shear viscosity constant");

      lambda_ = 0.;
      add_parameter("lambda", lambda_, "The bulk viscosity constant");

      cv_inverse_kappa_ = 1.866666666666666e-2;
      add_parameter("kappa",
                    cv_inverse_kappa_,
                    "Scaled thermal conductivity constant: c_v^{-1} kappa");
    }

  } // namespace NavierStokes
} // namespace ryujin
