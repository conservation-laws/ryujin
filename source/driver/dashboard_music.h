//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <ryujin/base/compile_time_options.h>
#include <ryujin/driver/dashboard.h>

namespace ryujin
{
  /**
   * Play the tune @p music in an endless loop, in the background thread of
   * the audio device, until the program exits or this function is called
   * with a different tune. Switching to a different tune closes the audio
   * device and opens a new one, and DashboardMusic::none only closes it,
   * which ends the background thread. Does nothing if @p music is already
   * playing.
   *
   * The function is thread safe.
   *
   * @ingroup Driver
   */
#if defined(WITH_MINIAUDIO)
  void play_dashboard_music(DashboardMusic music);
#else
  inline void play_dashboard_music(DashboardMusic /*music*/) {}
#endif
} // namespace ryujin
