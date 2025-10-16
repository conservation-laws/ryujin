//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023 - 2024 by the ryujin authors
//

#pragma once

namespace ryujin
{
  /**
   * Compatibility using declaration that selects the correct
   * ParabolicModule class type from the equation-specific Description
   * proxy class.
   *
   * @ingroup ParabolicModule
   */
  template <typename Description, int dim, typename Number = double>
  using ParabolicModule =
      typename Description::template ParabolicModule<dim, Number>;
} /* namespace ryujin */
