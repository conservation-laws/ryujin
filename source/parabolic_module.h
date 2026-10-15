//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

namespace ryujin
{
  /**
   * Compatibility using declaration that selects the ParabolicModule class
   * type of the ParabolicDescription for the given HyperbolicDescription.
   *
   * @ingroup ParabolicModule
   */
  template <typename HyperbolicDescription,
            typename ParabolicDescription,
            int dim,
            typename Number = double>
  using ParabolicModule = typename ParabolicDescription::
      template ParabolicModule<HyperbolicDescription, dim, Number>;
} /* namespace ryujin */
