//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/concept/hyperbolic_description.h>
#include <ryujin/concept/parabolic_description.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_aeos/hyperbolic_description.h>
#include <ryujin/euler_atmospheric/hyperbolic_description.h>
#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/euler_poisson/parabolic_description.h>
#include <ryujin/navier_stokes/parabolic_description.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/shallow_water/hyperbolic_description.h>
#include <ryujin/stub/parabolic_module.h>

#include <iostream>
#include <utility>

//
// Test Concept::HyperbolicDescription and Concept::ParabolicDescription:
// every pairing of the ryujin executable satisfies the concepts in all
// dimensions for the configured NUMBER type, and the Navier-Stokes module
// pairs with the Euler equations only.
//

using namespace ryujin;

template <typename HD, typename PD, typename Number>
constexpr bool pairs = []<int... dim>(std::integer_sequence<int, dim...>) {
  return ((Concept::HyperbolicDescription<HD, dim, Number> &&
           Concept::ParabolicDescription<PD, HD, dim, Number>)&&...);
}(std::integer_sequence<int, 1, 2, 3>());

#define CHECK(HD, PD)                                                          \
  std::cout << #HD " + " #PD ": " << pairs<HD, PD, NUMBER> << std::endl;

int main()
{
  std::cout << std::boolalpha;

  CHECK(Euler::HyperbolicDescription, StubParabolicDescription)
  CHECK(EulerAEOS::HyperbolicDescription, StubParabolicDescription)
  CHECK(EulerAtmospheric::HyperbolicDescription, StubParabolicDescription)
  CHECK(EulerBarotropic::HyperbolicDescription, StubParabolicDescription)
  CHECK(ScalarConservation::HyperbolicDescription, StubParabolicDescription)
  CHECK(ShallowWater::HyperbolicDescription, StubParabolicDescription)

  CHECK(Euler::HyperbolicDescription, EulerPoisson::ParabolicDescription)
  CHECK(EulerAEOS::HyperbolicDescription, EulerPoisson::ParabolicDescription)
  CHECK(EulerBarotropic::HyperbolicDescription,
        EulerPoisson::ParabolicDescription)

  CHECK(Euler::HyperbolicDescription, NavierStokes::ParabolicDescription)
  CHECK(EulerAEOS::HyperbolicDescription, NavierStokes::ParabolicDescription)
  CHECK(ShallowWater::HyperbolicDescription, NavierStokes::ParabolicDescription)
}
