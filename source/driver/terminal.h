//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <termios.h>

#include <optional>
#include <string_view>
#include <utility>

namespace ryujin::terminal
{
  /**
   * Return whether stdout reaches the controlling terminal, either
   * directly or through the pseudo terminals of launchers, such as mpirun,
   * that forward the output of their child processes.
   *
   * @ingroup Driver
   */
  bool stdout_reaches_terminal();


  /**
   * Return whether stdin and stdout both are terminals.
   *
   * @ingroup Driver
   */
  bool interactive();


  /**
   * The output to the controlling terminal (/dev/tty), written to directly
   * instead of through stdout. Under mpirun, stdout of rank 0 is forwarded
   * by mpirun, which delays the output if all cores are busy. If there is
   * no controlling terminal we write to stdout instead.
   *
   * @ingroup Driver
   */
  class Output
  {
  public:
    /**
     * Open the controlling terminal.
     */
    Output();

    ~Output();

    Output(const Output &) = delete;
    Output &operator=(const Output &) = delete;

    /**
     * Return whether we write to the controlling terminal, and not to
     * stdout.
     */
    bool controlling_terminal() const;

    /**
     * Return the number of columns and lines of the terminal we write to,
     * if known.
     */
    std::optional<std::pair<int, int>> size() const;

    /**
     * Write all of @p output. This function is async-signal-safe.
     */
    void write(std::string_view output) const;

  private:
    /** The file descriptor of /dev/tty, or -1. */
    int terminal_;
  };


  /**
   * Single key input from stdin: switch the terminal to single key mode if
   * stdin is the terminal and we are in its foreground, and restore the
   * previous mode on destruction.
   *
   * @ingroup Driver
   */
  class Keyboard
  {
  public:
    Keyboard();

    ~Keyboard();

    Keyboard(const Keyboard &) = delete;
    Keyboard &operator=(const Keyboard &) = delete;

    /**
     * Return whether we may wait for a key press at all. We may not if
     * stdin is the terminal but we are in the background.
     */
    bool may_wait() const;

    /**
     * Return whether a key has been pressed, or there is no more input.
     */
    bool key_pressed() const;

    /**
     * Restore the previous mode of the terminal, and discard the keys not
     * read. This function is async-signal-safe.
     */
    void restore() const;

  private:
    /** The previous mode, if we switched to single key mode. */
    std::optional<termios> previous_mode_;
    /** The return value of may_wait(). */
    bool may_wait_ = true;
  };
} // namespace ryujin::terminal
