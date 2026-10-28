//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/terminal.h>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/sysmacros.h>
#elif defined(__APPLE__)
#include <libproc.h>
#include <sys/param.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>

namespace ryujin::terminal
{
  namespace
  {
#if defined(__linux__)
    /* The pseudo terminals that launchers forward: */
    constexpr std::string_view pseudo_terminals = "/dev/pts/";

    /*
     * Return the parent and the controlling terminal of the process @p pid,
     * read from /proc/<pid>/stat:
     */
    std::pair<pid_t, dev_t> process_info(pid_t pid)
    {
      std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
      const std::string stat(std::istreambuf_iterator<char>(file), {});

      /* Skip over the process name, which may contain spaces: */
      const auto position = stat.rfind(')');
      if (position == std::string::npos)
        return {0, 0};

      std::istringstream fields(stat.substr(position + 1));
      char state = 0;
      pid_t parent = 0, group = 0, session = 0;
      unsigned int tty = 0;
      fields >> state >> parent >> group >> session >> tty;

      const unsigned int major = (tty >> 8) & 0xfff;
      const unsigned int minor = (tty & 0xff) | ((tty >> 12) & 0xfff00);
      return {parent, makedev(major, minor)};
    }

    /*
     * Return the device and the path of stdout of the process @p pid if it
     * is a character device, read from /proc/<pid>/fd/1:
     */
    std::optional<std::pair<dev_t, std::string>> stdout_device(pid_t pid)
    {
      const auto path = "/proc/" + std::to_string(pid) + "/fd/1";

      struct stat output = {};
      if (::stat(path.c_str(), &output) != 0 || !S_ISCHR(output.st_mode))
        return std::nullopt;

      std::error_code error;
      const auto target = std::filesystem::read_symlink(path, error);
      return std::pair{output.st_rdev, error ? "" : target.string()};
    }
#elif defined(__APPLE__)
    /* The pseudo terminals that launchers forward: */
    constexpr std::string_view pseudo_terminals = "/dev/ttys";

    /*
     * Return the parent and the controlling terminal of the process @p pid,
     * queried with libproc:
     */
    std::pair<pid_t, dev_t> process_info(pid_t pid)
    {
      proc_bsdinfo info = {};
      constexpr int size = sizeof(info);
      if (::proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) != size)
        return {0, 0};

      const auto tty = static_cast<dev_t>(info.e_tdev);
      return {static_cast<pid_t>(info.pbi_ppid), tty == NODEV ? 0 : tty};
    }

    /*
     * Return the device and the path of stdout of the process @p pid if it
     * is a character device, queried with libproc:
     */
    std::optional<std::pair<dev_t, std::string>> stdout_device(pid_t pid)
    {
      vnode_fdinfowithpath vnode = {};
      constexpr int size = sizeof(vnode);
      constexpr int flavor = PROC_PIDFDVNODEPATHINFO;
      if (::proc_pidfdinfo(pid, STDOUT_FILENO, flavor, &vnode, size) != size)
        return std::nullopt;

      const auto &stat = vnode.pvip.vip_vi.vi_stat;
      if (!S_ISCHR(stat.vst_mode))
        return std::nullopt;

      return std::pair{static_cast<dev_t>(stat.vst_rdev),
                       std::string(vnode.pvip.vip_path)};
    }
#endif
  } // namespace


  bool stdout_reaches_terminal()
  {
#if defined(__linux__) || defined(__APPLE__)
    const dev_t terminal = process_info(getpid()).second;
    if (terminal == 0)
      return false;

    /*
     * Follow stdout up the process tree as long as it is a pseudo terminal
     * other than the controlling terminal, i.e., a launcher forwards it:
     */
    pid_t pid = getpid();
    for (int depth = 0; depth < 16 && pid > 1; ++depth) {
      const auto output = stdout_device(pid);
      if (!output)
        return false;
      if (output->first == terminal)
        return true;
      if (!output->second.starts_with(pseudo_terminals))
        return false;

      pid = process_info(pid).first;
    }
    return false;
#else
    /*
     * Without a way to inspect other processes we cannot look past a
     * launcher and only accept stdout being the controlling terminal
     * itself:
     */
    return ::isatty(STDOUT_FILENO) && ::tcgetpgrp(STDOUT_FILENO) != -1;
#endif
  }


  bool interactive()
  {
    return ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO);
  }


  Output::Output()
      : terminal_(::open("/dev/tty", O_WRONLY | O_NOCTTY | O_CLOEXEC))
  {
  }


  Output::~Output()
  {
    if (terminal_ >= 0)
      ::close(terminal_);
  }


  bool Output::controlling_terminal() const
  {
    return terminal_ >= 0;
  }


  std::optional<std::pair<int, int>> Output::size() const
  {
    winsize size{};
    const int descriptor = terminal_ >= 0 ? terminal_ : STDOUT_FILENO;
    if (::ioctl(descriptor, TIOCGWINSZ, &size) != 0 || size.ws_col == 0 ||
        size.ws_row == 0)
      return std::nullopt;
    return std::pair<int, int>{size.ws_col, size.ws_row};
  }


  void Output::write(std::string_view output) const
  {
    const int descriptor = terminal_ >= 0 ? terminal_ : STDOUT_FILENO;
    while (!output.empty()) {
      const auto n = ::write(descriptor, output.data(), output.size());
      if (n < 0 && errno != EINTR)
        return;
      output.remove_prefix(std::max<ssize_t>(n, 0));
    }
  }


  Keyboard::Keyboard()
  {
    if (!::isatty(STDIN_FILENO))
      return;

    termios mode = {};
    if (::tcgetpgrp(STDIN_FILENO) != ::getpgrp() ||
        ::tcgetattr(STDIN_FILENO, &mode) != 0) {
      may_wait_ = false;
      return;
    }

    previous_mode_ = mode;
    mode.c_lflag &= ~(ICANON | ECHO);
    mode.c_cc[VMIN] = 1;
    mode.c_cc[VTIME] = 0;
    ::tcsetattr(STDIN_FILENO, TCSANOW, &mode);
  }


  Keyboard::~Keyboard()
  {
    restore();
  }


  bool Keyboard::may_wait() const
  {
    return may_wait_;
  }


  bool Keyboard::key_pressed() const
  {
    /* Under mpirun, stdin is a pipe that receives whole lines: */
    pollfd input = {STDIN_FILENO, POLLIN, 0};
    if (::poll(&input, 1, 0) <= 0)
      return false;

    std::array<char, 64> buffer;
    [[maybe_unused]] const auto n =
        ::read(STDIN_FILENO, buffer.data(), buffer.size());
    return true;
  }


  void Keyboard::restore() const
  {
    if (previous_mode_)
      ::tcsetattr(STDIN_FILENO, TCSAFLUSH, &*previous_mode_);
  }
} // namespace ryujin::terminal
