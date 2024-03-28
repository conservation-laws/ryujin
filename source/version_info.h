//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by the ryujin authors
//

#pragma once

#include <ostream>

namespace ryujin
{
  /**
   * Print git revision and version info.
   */
  void print_revision_and_version(std::ostream &stream);

  /**
   * Print compile time parameters.
   */
  void print_compile_time_parameters(std::ostream &stream);
} // namespace ryujin
