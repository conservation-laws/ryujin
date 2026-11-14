//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#pragma once

#include <sys/types.h>
#include <termios.h>

#include <chrono>
#include <optional>
#include <string>
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
   * stdin reaches it, directly or through a launcher such as mpirun that
   * forwards stdin, if stdout reaches the same terminal, and if the
   * process that reads the terminal is in its foreground. Restore the
   * previous mode on destruction, or from a watchdog process if we die
   * without cleaning up.
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
     * Return whether we have switched the terminal to single key mode, in
     * which we read keys from stdin as they are pressed.
     */
    bool readable() const;

    /**
     * Wait up to @p timeout for input and return the keys read, an empty
     * string if there is none, or std::nullopt if there is no more input.
     */
    std::optional<std::string> read(std::chrono::milliseconds timeout) const;

    /**
     * Restore the previous mode of the terminal, and discard the keys not
     * read. This function is async-signal-safe.
     */
    void restore() const;

  private:
    /** The terminal that stdin reaches, or -1. */
    int terminal_ = -1;
    /** The write end of the pipe to the watchdog process, or -1. */
    int pipe_ = -1;
    /** The watchdog process that restores the terminal if we die. */
    pid_t watchdog_ = -1;
    /** The previous mode, if we switched to single key mode. */
    std::optional<termios> previous_mode_;
  };
} // namespace ryujin::terminal
