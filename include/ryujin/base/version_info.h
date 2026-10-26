//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 - 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>

#include <ostream>
#include <string>

namespace ryujin
{
  /**
   * Print git revision and version info.
   *
   * @ingroup Utilities
   */
  void print_revision_and_version(std::ostream &stream);


  /**
   * Return the version and the short git revision of ryujin, such as
   * "ryujin v2.5.0 1a2b3c4d".
   *
   * @ingroup Utilities
   */
  std::string version_string();
} // namespace ryujin
