//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/concept/hyperbolic_description.h>
#include <ryujin/euler/hyperbolic_description.h>
#include <ryujin/euler_aeos/hyperbolic_description.h>
#include <ryujin/euler_barotropic/hyperbolic_description.h>
#include <ryujin/scalar_conservation/hyperbolic_description.h>
#include <ryujin/shallow_water/hyperbolic_description.h>

#include <iostream>

//
// Test Concept::HyperbolicSystemViewEquilibratesStates: only the shallow
// water equations form equilibrated states.
//

using namespace ryujin;

template <typename HD, int dim, typename Number>
constexpr bool equilibrates_states =
    Concept::HyperbolicSystemViewEquilibratesStates<
        Concept::view_type<typename HD::HyperbolicSystem, dim, Number>,
        dim,
        Number>;

template <typename HD, typename Number>
constexpr bool equilibrates =
    equilibrates_states<HD, 1, Number> && equilibrates_states<HD, 2, Number> &&
    equilibrates_states<HD, 3, Number> &&
    equilibrates_states<HD, 1, Concept::vectorized_type<Number>> &&
    equilibrates_states<HD, 2, Concept::vectorized_type<Number>> &&
    equilibrates_states<HD, 3, Concept::vectorized_type<Number>>;

#define CHECK(HD)                                                              \
  std::cout << #HD ": " << equilibrates<HD, NUMBER> << std::endl;

int main()
{
  std::cout << std::boolalpha;

  CHECK(Euler::HyperbolicDescription)
  CHECK(EulerAEOS::HyperbolicDescription)
  CHECK(EulerBarotropic::HyperbolicDescription)
  CHECK(ScalarConservation::HyperbolicDescription)
  CHECK(ShallowWater::HyperbolicDescription)
}
