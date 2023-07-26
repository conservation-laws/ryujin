//
// SPDX-License-Identifier: MIT
// Copyright (C) 2020 - 2023 by the ryujin authors
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

      void print_solver_statistics(std::ostream &output) const
      {
        // FIXME remove
        bool use_gmg_velocity_ = true;
        bool use_gmg_internal_energy_ = true;
        unsigned int n_iterations_velocity_ = 42;
        unsigned int n_iterations_internal_energy_ = 42;

        output << "        [ " << std::setprecision(2) << std::fixed
               << n_iterations_velocity_
               << (use_gmg_velocity_ ? " GMG vel -- " : " CG vel -- ")
               << n_iterations_internal_energy_
               << (use_gmg_internal_energy_ ? " GMG int ]" : " CG int ]")
               << std::endl;
      }

    private:
      /**
       * @name Runtime parameters, internal fields and methods
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

  } // namespace Euler
} // namespace ryujin
