//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2026 by the ryujin authors
//

#include <ryujin/scalar_conservation/initial_state_function.h>
#include <ryujin/scalar_conservation/initial_state_library.h>
#include <ryujin/scalar_conservation/initial_state_uniform.h>

namespace ryujin
{
  namespace ScalarConservation
  {
    /**
     * The built-in initial states.
     */
    template <int dim, typename Number>
    auto defaults()
    {
      using L = InitialStateLibrary<HyperbolicSystem, dim, Number>;
      return std::vector<typename L::Creator>{
          L::template creator<Function<dim, Number>>(),
          L::template creator<Uniform<dim, Number>>(),
      };
    }
  } // namespace ScalarConservation

  template <>
  auto InitialStateLibrary<ScalarConservation::HyperbolicSystem, 1, NUMBER>::
      defaults() -> std::vector<Creator>
  {
    return ScalarConservation::defaults<1, NUMBER>();
  }

  template <>
  auto InitialStateLibrary<ScalarConservation::HyperbolicSystem, 2, NUMBER>::
      defaults() -> std::vector<Creator>
  {
    return ScalarConservation::defaults<2, NUMBER>();
  }

  template <>
  auto InitialStateLibrary<ScalarConservation::HyperbolicSystem, 3, NUMBER>::
      defaults() -> std::vector<Creator>
  {
    return ScalarConservation::defaults<3, NUMBER>();
  }

  /* instantiations */

  template class Library<
      InitialState<ScalarConservation::HyperbolicSystem, 1, NUMBER>,
      const ScalarConservation::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<ScalarConservation::HyperbolicSystem, 2, NUMBER>,
      const ScalarConservation::HyperbolicSystem &,
      const std::string &>;
  template class Library<
      InitialState<ScalarConservation::HyperbolicSystem, 3, NUMBER>,
      const ScalarConservation::HyperbolicSystem &,
      const std::string &>;
} // namespace ryujin
