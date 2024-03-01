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
    class KPP : public Flux
    {
    public:
      KPP(const std::string &subsection)
          : Flux("kpp", subsection)
      {
        flux_formula_ = "f(u)={sin(u),cos(u)}";
      }


      double value(const double state,
                   const unsigned int direction) const override
      {
        switch (direction) {
        case 0:
          return std::sin(state);
        case 1:
          return std::cos(state);
        default:
          AssertThrow(false,
                      dealii::ExcMessage(
                          "KPP is only defined in (1 or) 2 space dimensions"));
          __builtin_trap();
        }
      }


      double gradient(const double state,
                      const unsigned int direction) const override
      {
        switch (direction) {
        case 0:
          return std::cos(state);
        case 1:
          return -std::sin(state);
        default:
          AssertThrow(false,
                      dealii::ExcMessage(
                          "KPP is only defined in (1 or) 2 space dimensions"));
          __builtin_trap();
        }
      }
    };
  } // namespace FluxLibrary
} // namespace ryujin
