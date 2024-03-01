Installation instructions
=========================

Necessary tools and library dependencies
----------------------------------------

ryujin requires [deal.II](https://dealii.org) version 9.3.0 or newer
compiled with support enabled for MPI and P4est.

On Debian and Ubuntu you can conveniently install all necessary libraries
and tools with `apt`.
  - On <b>Debian</b> you can run the following command (as root user) to
    install deal.II, all development libraries and necessary tools:
    ```
    apt install libdeal.ii-dev cmake cmake-curses-gui numdiff make g++ ninja-build git
    ```
  - On <b>Ubuntu LTS 22.04</b> you can install the current deal.II version
    from a <a href="https://launchpad.net/~ginggs/+archive/ubuntu/deal.ii-9.5.1-backports">PPA</a>
    ```
    sudo add-apt-repository ppa:ginggs/deal.ii-9.5.1-backports
    sudo apt update
    sudo apt install libdeal.ii-dev
    sudo apt install libdeal.ii-dev cmake cmake-curses-gui numdiff make g++ ninja-build git
    ```

If you are on a MAC it might be easiest to simply install our precompiled
MAC bundle for deal.II which you can find on the [Github download
page](https://github.com/dealii/dealii/releases/tag/v9.5.2).

If you are on Windows we strongly recommend to set up the [<b>Windows
Subsystem for Linux v2</b>
(WSL)](https://learn.microsoft.com/en-us/windows/wsl/about) and install
<b>Ubuntu LTS 22.04</b> from the Microsoft store. Then, simply launch the
Ubuntu App which starts a Bash shell. You can then proceed with the
installation instructions for Ubuntu LTS 22.04 summarized above. You can
find a helpful tutorial on how to use the linux command line
[here](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview).

Generic installation instructions how to download, install, or manually
compile deal.II can be found on the
[deal.II homepage](https://dealii.org/download.html) and on the
[deal.II wiki](https://github.com/dealii/dealii/wiki).
Make sure that deal.II is configured with MPI and P4est support and that
the following additional tools are in your `PATH`: git, numdiff, cmake,
make.


Retrieving and compiling ryujin
-------------------------------

At this point you should have deal.II an all necessary command line tools
installed. You then simply check out the ryujin repository and run `make`
to compile ryujin:
```
git clone https://github.com/conservation-laws/ryujin
cd ryujin
git submodule init
git submodule update
make release
```

Note that ryujin uses the [CMake](https://cmake.org/) buildsystem. The
`Makefile` found in the repository only contains a number of convenience
targets and its use is entirely optional. If invoked it will create a
subdirectory `build` and run cmake to configure the project. The executable
will be located in <code>build/run</code>. The convenience Makefile
contains the following additional targets:
  - `make debug`:  switch to debug build and compile program
  - `make release`:  switch to release build and compile program
  - `make edit_cache`:  runs ccmake in the build directory
