//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020 - 2024 by the ryujin authors
//

#pragma once

#include "flux.h"

namespace ryujin
{
  namespace FluxLibrary
  {
    /**
     * A generic flux description parsed from a user-provided string
     *
     * @ingroup ScalarConservation
     */
    class Burgers : public Flux
    {
    public:
      Burgers(const std::string &subsection)
          : Flux("burgers", subsection)
      {
        flux_formula_ = "f(u)={0.5u^2}";
      }


      double value(const double state,
                   const unsigned int /*direction*/) const override
      {
        return 0.5 * state * state;
      }


      double gradient(const double state,
                      const unsigned int /*direction*/) const override
      {
        return state;
      }
    };
  } // namespace FluxLibrary
} // namespace ryujin
