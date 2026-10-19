//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include "barotropic_equation_of_state.h"

namespace ryujin
{
  namespace BarotropicEquationOfStateLibrary
  {
    /**
     * Trivial equation of state for pressureless Euler
     *
     * @ingroup EulerEquations
     */
    class Pressureless : public BarotropicEquationOfState
    {
    public:
      Pressureless(const std::string &subsection)
          : BarotropicEquationOfState("pressureless", subsection)
      {
      }

      double pressure(double /*rho*/) const final
      {
        return 0.;
      }

      double specific_internal_energy(double /*rho*/) const final
      {
        return 0.;
      }

      double speed_of_sound(double /*rho*/) const final
      {
        return 0.;
      }
    };
  } // namespace BarotropicEquationOfStateLibrary
} // namespace ryujin
