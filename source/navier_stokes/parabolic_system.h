//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2022 - 2023 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>
#include <convenience_macros.h>

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
    class ParabolicSystem final : public dealii::ParameterAcceptor
    {
    public:
      /**
       * The name of the hyperbolic system as a string.
       */
      static inline const std::string problem_name =
          "Newtonian fluid viscosity model with Fourier-law heat flux";

      /**
       * This parabolic subsystem represents an identity.
       */
      static constexpr bool is_identity = false;

      /**
       * Constructor.
       */
      ParabolicSystem(const std::string &subsection = "/B - Equation");

      ACCESSOR_READ_ONLY(mu)
      ACCESSOR_READ_ONLY(lambda)
      ACCESSOR_READ_ONLY(cv_inverse_kappa)

      unsigned int n_auxiliary_state_vectors() const
      {
        return auxiliary_component_names_.size();
      }

      ACCESSOR_READ_ONLY(auxiliary_component_names);

    private:
      /**
       * @name Runtime parameters, internal fields and methods
       */
      //@{

      double mu_;
      double lambda_;
      double cv_inverse_kappa_;

      //@}

      const std::vector<std::string> auxiliary_component_names_;
    }; /* ParabolicSystem */


    /*
     * -------------------------------------------------------------------------
     * Inline definitions
     * -------------------------------------------------------------------------
     */


    inline ParabolicSystem::ParabolicSystem(const std::string &subsection)
        : ParameterAcceptor(subsection)
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
