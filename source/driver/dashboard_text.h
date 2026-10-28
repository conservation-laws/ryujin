//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/driver/dashboard.h>

#include <string>

namespace ryujin
{
  /**
   * Render the snapshot @p data as plain text, for the log file and for
   * terminals without the Dashboard.
   *
   * @ingroup Driver
   */
  std::string render_text(const DashboardData &data);
} // namespace ryujin
