//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <ryujin/driver/equation.h>

namespace ryujin
{
  std::map<std::string, Equation> &equations()
  {
    static std::map<std::string, Equation> equations;
    return equations;
  }
} // namespace ryujin
