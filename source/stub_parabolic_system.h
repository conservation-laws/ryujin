//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2025 - 2026 by the ryujin authors
//

#pragma once

#include <compile_time_options.h>

#include <ryujin/interface/parabolic_system.h>

namespace ryujin
{
  /**
   * The parabolic subsystem of a purely hyperbolic equation. This is just
   * the identity operator.
   *
   * @ingroup ParabolicModule
   */
  class StubParabolicSystem final : public Interface::ParabolicSystem
  {
  public:
    /**
     * Constructor.
     */
    StubParabolicSystem(const std::string &subsection = "/ParabolicSystem")
        : Interface::ParabolicSystem("Identity", subsection)
    {
    }

    /**
     * Whether the parabolic system is the identity.
     */
    static constexpr bool is_identity = true;

    /**
     * No state components.
     */
    std::vector<std::string>
    state_names(const unsigned int /*dim*/) const override
    {
      return {};
    }

    /**
     * No precomputed values.
     */
    std::vector<std::string>
    precomputed_names(const unsigned int /*dim*/) const override
    {
      return {};
    }
  }; /* StubParabolicSystem */
} // namespace ryujin
