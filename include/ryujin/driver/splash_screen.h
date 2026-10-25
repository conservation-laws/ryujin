//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <cstdint>
#include <string>

namespace ryujin
{
  /**
   * The splash screens of ryujin.
   *
   * @ingroup Driver
   */
  enum class SplashScreen : std::uint8_t {
    /**
     * No splash screen.
     */
    none,

    /**
     * The logo, the name and the version, the equation, the dimension, and
     * the number of MPI ranks, and the copyright of ryujin, centered on the
     * screen above a few free lines for the following output. The logo is
     * scaled to the size of the terminal, and omitted if the terminal is
     * too small. It is only shown if stdout reaches an interactive
     * terminal of at least 80 columns.
     */
    logo,

    /**
     * An intro in the style of an arcade game of the eighties that
     * announces the equation and the dimension, and the number of MPI
     * ranks as credits. The intro waits at "PRESS START" for a key press
     * on stdin (under mpirun: the return key), unless there is no input.
     * It is only shown if stdout reaches an interactive terminal of at
     * least 80x24 characters.
     */
    arcade,
  };


  /**
   * Play the splash screen @p splash_screen for the equation @p equation
   * in dimension @p dim, run with @p n_ranks MPI ranks. Call this
   * function on rank 0 only.
   *
   * @ingroup Driver
   */
  void play_splash_screen(const std::string &equation,
                          int dim,
                          unsigned int n_ranks,
                          SplashScreen splash_screen = SplashScreen::none);
} // namespace ryujin
