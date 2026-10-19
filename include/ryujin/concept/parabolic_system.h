//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/interface/parabolic_system.h>

#include <concepts>
#include <string>

namespace ryujin
{
  namespace Concept
  {
    /**
     * A parabolic system: constructed from the subsection of its run time
     * parameters.
     *
     * @ingroup ParabolicModule
     */
    template <typename T>
    concept ParabolicSystem =
        std::derived_from<T, Interface::ParabolicSystem> &&
        std::constructible_from<T, const std::string &>;
  } // namespace Concept
} // namespace ryujin
