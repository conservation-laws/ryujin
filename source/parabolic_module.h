//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2025 by the ryujin authors
//

#pragma once

namespace ryujin
{
  /**
   * Compatibility using declaration that selects the correct
   * ParabolicModule class type from the equation-specific HyperbolicDescription
   * proxy class.
   *
   * @ingroup ParabolicModule
   */
  template <typename HyperbolicDescription, int dim, typename Number = double>
  using ParabolicModule =
      typename HyperbolicDescription::template ParabolicModule<dim, Number>;
} /* namespace ryujin */
