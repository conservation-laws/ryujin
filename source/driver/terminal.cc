//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2026 by the ryujin authors
//

#include <driver/terminal.h>

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
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
#include <vector>

namespace ryujin::terminal
{
  namespace
  {
    /* The parent, the controlling terminal, and the job control state: */
    struct ProcessInfo {
      pid_t parent = 0;
      dev_t terminal = 0;
      bool foreground = false;
    };

    /* A terminal, and whether the process holding it is in the foreground: */
    struct Terminal {
      std::string path;
      bool foreground = false;
    };

#if defined(__linux__)
    /* The pseudo terminals that launchers forward: */
    constexpr std::string_view pseudo_terminals = "/dev/pts/";

    /*
     * Return the parent and the controlling terminal of the process @p pid,
     * and whether it is in the foreground of the terminal, read from
     * /proc/<pid>/stat:
     */
    ProcessInfo process_info(pid_t pid)
    {
      std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
      const std::string stat(std::istreambuf_iterator<char>(file), {});

      /* Skip over the process name, which may contain spaces: */
      const auto position = stat.rfind(')');
      if (position == std::string::npos)
        return {};

      std::istringstream fields(stat.substr(position + 1));
      char state = 0;
      pid_t parent = 0, group = 0, session = 0, foreground = 0;
      unsigned int tty = 0;
      fields >> state >> parent >> group >> session >> tty >> foreground;

      const unsigned int major = (tty >> 8) & 0xfff;
      const unsigned int minor = (tty & 0xff) | ((tty >> 12) & 0xfff00);
      return {parent, makedev(major, minor), group == foreground};
    }

    /*
     * Return the device and the path of the file descriptor @p fd of the
     * process @p pid if it is a character device, read from
     * /proc/<pid>/fd/<fd>:
     */
    std::optional<std::pair<dev_t, std::string>> character_device(pid_t pid,
                                                                  int fd)
    {
      const auto path =
          "/proc/" + std::to_string(pid) + "/fd/" + std::to_string(fd);

      struct stat output = {};
      if (::stat(path.c_str(), &output) != 0 || !S_ISCHR(output.st_mode))
        return std::nullopt;

      std::error_code error;
      const auto target = std::filesystem::read_symlink(path, error);
      return std::pair{output.st_rdev, error ? "" : target.string()};
    }

    /*
     * Return whether the process @p parent holds the other end of the pipe
     * that is the file descriptor @p fd of the process @p pid: the write
     * end of stdin, or the read end of stdout:
     */
    bool parent_holds_pipe(pid_t pid, pid_t parent, int fd)
    {
      std::error_code error;
      const auto pipe = std::filesystem::read_symlink(
          "/proc/" + std::to_string(pid) + "/fd/" + std::to_string(fd), error);
      if (error || !pipe.string().starts_with("pipe:"))
        return false;

      const unsigned long access = fd == STDIN_FILENO ? O_WRONLY : O_RDONLY;

      const auto directory = "/proc/" + std::to_string(parent);
      std::filesystem::directory_iterator other(directory + "/fd", error);
      for (; !error && other != std::filesystem::directory_iterator();
           other.increment(error)) {
        std::error_code link_error;
        if (std::filesystem::read_symlink(other->path(), link_error) != pipe)
          continue;

        std::ifstream file(directory + "/fdinfo/" +
                           other->path().filename().string());
        std::string line;
        while (std::getline(file, line))
          if (line.starts_with("flags:") &&
              (std::stoul(line.substr(6), nullptr, 8) & O_ACCMODE) == access)
            return true;
      }
      return false;
    }
#elif defined(__APPLE__)
    /* The pseudo terminals that launchers forward: */
    constexpr std::string_view pseudo_terminals = "/dev/ttys";

    /*
     * Return the parent and the controlling terminal of the process @p pid,
     * and whether it is in the foreground of the terminal, queried with
     * libproc:
     */
    ProcessInfo process_info(pid_t pid)
    {
      proc_bsdinfo info = {};
      constexpr int size = sizeof(info);
      if (::proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) != size)
        return {};

      const auto tty = static_cast<dev_t>(info.e_tdev);
      return {static_cast<pid_t>(info.pbi_ppid),
              tty == NODEV ? 0 : tty,
              info.pbi_pgid == info.e_tpgid};
    }

    /*
     * Return the device and the path of the file descriptor @p fd of the
     * process @p pid if it is a character device, queried with libproc:
     */
    std::optional<std::pair<dev_t, std::string>> character_device(pid_t pid,
                                                                  int fd)
    {
      vnode_fdinfowithpath vnode = {};
      constexpr int size = sizeof(vnode);
      constexpr int flavor = PROC_PIDFDVNODEPATHINFO;
      if (::proc_pidfdinfo(pid, fd, flavor, &vnode, size) != size)
        return std::nullopt;

      const auto &stat = vnode.pvip.vip_vi.vi_stat;
      if (!S_ISCHR(stat.vst_mode))
        return std::nullopt;

      return std::pair{static_cast<dev_t>(stat.vst_rdev),
                       std::string(vnode.pvip.vip_path)};
    }

    /*
     * Return whether the process @p parent holds the other end of the pipe
     * that is the file descriptor @p fd of the process @p pid:
     */
    bool parent_holds_pipe(pid_t pid, pid_t parent, int fd)
    {
      pipe_fdinfo pipe = {};
      constexpr int size = sizeof(pipe);
      constexpr int flavor = PROC_PIDFDPIPEINFO;
      if (::proc_pidfdinfo(pid, fd, flavor, &pipe, size) != size)
        return false;

      const int bytes = ::proc_pidinfo(parent, PROC_PIDLISTFDS, 0, nullptr, 0);
      if (bytes <= 0)
        return false;
      std::vector<proc_fdinfo> fds(bytes / sizeof(proc_fdinfo));
      const int filled =
          ::proc_pidinfo(parent, PROC_PIDLISTFDS, 0, fds.data(), bytes);
      fds.resize(std::max(filled, 0) / sizeof(proc_fdinfo));

      return std::ranges::any_of(fds, [&](const proc_fdinfo &other) {
        pipe_fdinfo end = {};
        return other.proc_fdtype == PROX_FDTYPE_PIPE &&
               ::proc_pidfdinfo(parent, other.proc_fd, flavor, &end, size) ==
                   size &&
               end.pipeinfo.pipe_handle == pipe.pipeinfo.pipe_peerhandle;
      });
    }
#endif

    /*
     * Return the terminal that the file descriptor @p fd, stdin or stdout,
     * reaches: follow @p fd up the process tree as long as a launcher
     * forwards it, i.e., through a pseudo terminal other than the
     * controlling terminal (Open MPI stdout), or through a pipe whose other
     * end the parent holds (Open MPI stdin, MPICH). With a controlling
     * terminal, a pipe on stdout captures the output instead (ctest):
     */
    std::optional<Terminal> forwarded_terminal(int fd)
    {
#if defined(__linux__) || defined(__APPLE__)
      pid_t pid = getpid();
      for (int depth = 0; depth < 16 && pid > 1; ++depth) {
        const auto info = process_info(pid);
        const auto device = character_device(pid, fd);
        if (device && info.terminal != 0 && device->first == info.terminal)
          return Terminal{device->second, info.foreground};

        const bool forwarded =
            info.terminal != 0 && fd == STDOUT_FILENO
                ? device && device->second.starts_with(pseudo_terminals)
                : parent_holds_pipe(pid, info.parent, fd);
        if (!forwarded)
          return std::nullopt;

        pid = info.parent;
      }
      return std::nullopt;
#else
      /*
       * Without a way to inspect other processes we cannot look past a
       * launcher and only accept the controlling terminal itself:
       */
      const pid_t foreground = ::tcgetpgrp(fd);
      if (!::isatty(fd) || foreground == -1)
        return std::nullopt;
      return Terminal{::ttyname(fd), foreground == ::getpgrp()};
#endif
    }


    /*
     * Set the mode of @p terminal to @p mode, also from a background
     * process group of its session, in which the kernel would otherwise
     * stop us with SIGTTOU. This function is async-signal-safe:
     */
    void set_mode(int terminal, const termios &mode, int when)
    {
      sigset_t stop, previous;
      ::sigemptyset(&stop);
      ::sigaddset(&stop, SIGTTOU);
      ::pthread_sigmask(SIG_BLOCK, &stop, &previous);
      ::tcsetattr(terminal, when, &mode);
      ::pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    }
  } // namespace


  bool stdout_reaches_terminal()
  {
    return forwarded_terminal(STDOUT_FILENO).has_value();
  }


  bool interactive()
  {
    return ::isatty(STDIN_FILENO) && ::isatty(STDOUT_FILENO);
  }


  Output::Output()
      : terminal_(::open("/dev/tty", O_WRONLY | O_NOCTTY | O_CLOEXEC))
  {
    if (terminal_ < 0)
      if (const auto terminal = forwarded_terminal(STDOUT_FILENO))
        terminal_ =
            ::open(terminal->path.c_str(), O_WRONLY | O_NOCTTY | O_CLOEXEC);
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
    /*
     * Switch the terminal that stdin reaches, directly or through a
     * launcher such as mpirun, if it is the terminal of stdout, and the
     * process that reads it is in its foreground:
     */
    const auto input = forwarded_terminal(STDIN_FILENO);
    if (!input || !input->foreground)
      return;
    const auto output = forwarded_terminal(STDOUT_FILENO);
    if (!output || output->path != input->path)
      return;

    terminal_ = ::open(input->path.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
    termios mode = {};
    if (terminal_ < 0 || ::tcgetattr(terminal_, &mode) != 0)
      return;

    /*
     * Restore the terminal from a watchdog process if we die without
     * cleaning up, e.g., by the SIGKILL of mpirun: the watchdog leaves our
     * session, so that it survives us, and restores the terminal as soon
     * as the pipe that we hold reaches its end:
     */
    int ends[2];
    if (::pipe(ends) != 0)
      return;
    ::fcntl(ends[1], F_SETFD, FD_CLOEXEC);
    const long open_max = ::sysconf(_SC_OPEN_MAX);

    watchdog_ = ::fork();
    if (watchdog_ == 0) {
      /* Only async-signal-safe calls in the child of a threaded process: */
      ::setsid();
      for (int fd = 0; fd < open_max; ++fd)
        if (fd != ends[0] && fd != terminal_)
          ::close(fd);
      char c;
      while (::read(ends[0], &c, 1) < 0 && errno == EINTR)
        ;
      ::tcsetattr(terminal_, TCSANOW, &mode);
      ::_exit(0);
    }

    ::close(ends[0]);
    if (watchdog_ < 0) {
      ::close(ends[1]);
      return;
    }
    pipe_ = ends[1];

    previous_mode_ = mode;
    mode.c_lflag &= ~(ICANON | ECHO);
    mode.c_cc[VMIN] = 1;
    mode.c_cc[VTIME] = 0;
    set_mode(terminal_, mode, TCSANOW);
  }


  Keyboard::~Keyboard()
  {
    restore();
    if (pipe_ >= 0) {
      ::close(pipe_);
      ::waitpid(watchdog_, nullptr, 0);
    }
    if (terminal_ >= 0)
      ::close(terminal_);
  }


  bool Keyboard::readable() const
  {
    return previous_mode_.has_value();
  }


  std::optional<std::string>
  Keyboard::read(std::chrono::milliseconds timeout) const
  {
    pollfd input = {STDIN_FILENO, POLLIN, 0};
    if (::poll(&input, 1, static_cast<int>(timeout.count())) <= 0)
      return std::string();

    /* A closed pipe polls as ready and reads nothing: */
    std::array<char, 64> buffer;
    const auto n = ::read(STDIN_FILENO, buffer.data(), buffer.size());
    if (n < 0 && (errno == EINTR || errno == EAGAIN))
      return std::string();
    if (n <= 0)
      return std::nullopt;
    return std::string(buffer.data(), n);
  }


  void Keyboard::restore() const
  {
    if (previous_mode_)
      set_mode(terminal_, *previous_mode_, TCSAFLUSH);
  }
} // namespace ryujin::terminal
