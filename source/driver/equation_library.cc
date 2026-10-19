//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/equation_library.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_aeos/hyperbolic_description.h>
#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>
#include <ryujin/navier_stokes/parabolic_description.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/shallow_water/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>
#include <ryujin/typed/equation.h>

#include <memory>
#include <string>
#include <vector>

namespace ryujin
{
  namespace
  {
    /**
     * The creator of the equation that pairs @p HD with @p PD under the
     * name @p name.
     */
    template <typename HD, typename PD = StubParabolicDescription>
    EquationLibrary::Creator equation(const std::string &name)
    {
      return [name]() -> std::unique_ptr<Interface::Equation> {
        return std::make_unique<Equation<HD, PD, NUMBER>>(name);
      };
    }
  } // namespace


  template <>
  auto EquationLibrary::defaults() -> std::vector<Creator>
  {
    std::vector<Creator> creators;

#if defined(RYUJIN_WITH_EQUATION_EULER)
    creators.push_back(equation<Euler::HyperbolicDescription>("euler"));
#endif
#if defined(RYUJIN_WITH_EQUATION_EULER_AEOS)
    creators.push_back(
        equation<EulerAEOS::HyperbolicDescription>("euler aeos"));
#endif
#if defined(RYUJIN_WITH_EQUATION_EULER_BAROTROPIC)
    creators.push_back(
        equation<EulerBarotropic::HyperbolicDescription>("euler barotropic"));
#endif
#if defined(RYUJIN_WITH_EQUATION_EULER_POISSON)
    creators.push_back(
        equation<Euler::HyperbolicDescription,
                 EulerPoisson::ParabolicDescription>("euler poisson"));
#endif
#if defined(RYUJIN_WITH_EQUATION_EULER_POISSON) &&                             \
    defined(RYUJIN_WITH_EQUATION_EULER_AEOS)
    creators.push_back(
        equation<EulerAEOS::HyperbolicDescription,
                 EulerPoisson::ParabolicDescription>("euler poisson aeos"));
#endif
#if defined(RYUJIN_WITH_EQUATION_EULER_POISSON) &&                             \
    defined(RYUJIN_WITH_EQUATION_EULER_BAROTROPIC)
    creators.push_back(equation<EulerBarotropic::HyperbolicDescription,
                                EulerPoisson::ParabolicDescription>(
        "euler poisson barotropic"));
#endif
#if defined(RYUJIN_WITH_EQUATION_NAVIER_STOKES)
    creators.push_back(
        equation<Euler::HyperbolicDescription,
                 NavierStokes::ParabolicDescription>("navier stokes"));
#endif
#if defined(RYUJIN_WITH_EQUATION_SCALAR_CONSERVATION)
    creators.push_back(equation<ScalarConservation::HyperbolicDescription>(
        "scalar conservation"));
#endif
#if defined(RYUJIN_WITH_EQUATION_SHALLOW_WATER)
    creators.push_back(
        equation<ShallowWater::HyperbolicDescription>("shallow water"));
#endif

    return creators;
  }

  /* instantiations */

  template class Library<Interface::Equation>;
} // namespace ryujin
