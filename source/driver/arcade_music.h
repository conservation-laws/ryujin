//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

namespace ryujin
{
  /**
   * Start the music of SplashScreen::arcade: an original tune in the style
   * of the arcade racing games of the eighties, played in a background
   * thread in an endless loop until the program exits. Does nothing if
   * the music is already playing, or if there is no audio device.
   *
   * @ingroup Driver
   */
  void play_arcade_music();
} // namespace ryujin
