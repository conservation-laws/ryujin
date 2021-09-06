/**
 * @page Installation Installation instructions
 *
 * <h2>Necessary tools and library dependencies</h2>
 *
 * ryujin requires deal.II version 9.2.0 or newer compiled with enabled
 * support for MPI and P4est. On Debian and Ubuntu you can conveniently
 * install all necessary libraries and tools with apt.
 *
 * <h4>For Debian testing:</h4>
 *
 * Run as root user:
 * @code
 *   apt install libdeal.ii-dev cmake make g++ ninja-build git
 * @endcode
 *
 * <h4>For Debian stable:</h4>
 *
 * In order to enable buster-backports https://backports.debian.org and
 * install all necessary packages run as root user:
 * @code
 *   mkdir -p /etc/apt/sources.list.d/
 *   echo deb http://deb.debian.org/debian buster-backports main > /etc/apt/sources.list.d/buster-backports.conf
 *   apt update
 *
 *   apt install libdeal.ii-dev/buster-backports libp4est-dev/buster-backports
 * @endcode
 *
 * Also make sure to have numdiff, cmake, g++, ninja and git installed:
 * @code
 *   sudo apt install numdiff
 *   apt install cmake make g++ ninja-build git ca-certificates
 * @endcode
 *
 * <h4>For Ubuntu LTS 20.04:</h4>
 *
 * In order to add the backports <a href="https://launchpad.net/~ginggs/+archive/ubuntu/deal.ii-9.3.0-backports">PPA</a>
 * issue the following commands:
 * @code
 *   sudo add-apt-repository ppa:ginggs/deal.ii-9.3.0-backports
 *   sudo apt update
 *   sudo apt install libdeal.ii-dev
 * @endcode
 *
 * Also make sure to have numdiff, cmake, g++, ninja and git installed:
 * @code
 *   sudo apt install numdiff
 *   sudo apt install cmake make g++ ninja-build git ca-certificates
 * @endcode
 *
 * <h4>Manual compilation and installation</h4>
 * Instructions how to manyally compile and install deal.II can be found
 * on the
 * <a href="https://www.dealii.org/">deal.II homepage</a>,
 * the
 * <a href="https://www.dealii.org/developer/readme.html">README</a>,
 * and on the
 * <a href="https://github.com/dealii/dealii/wiki">deal.II wiki</a>.
 * Make sure that deal.II is configured with MPI and P4est support and that
 * the following additional tools are in your path: git, numdiff, cmake,
 * make.
 *
 * <h2>Retrieving and compiling ryujin</h2>
 *
 * Simply check out the repository and run make:
 *
 * @code
 * git clone https://github.com/conservation-laws/ryujin
 * cd ryujin
 * git submodule init
 * git submodule update
 * make release
 * @endcode
*
 * @note The <code>Makefile</code> found in the repository only contains a
 * number of convenience targets and its use is entirely optional. It will
 * create a subdirectory <code>build</code> and run cmake to configure the
 * project. The executable will be located in <code>build/run</code>.
 * The convenience Makefile contains the following additional targets:
 * @code
 * make debug       -  switch to debug build and compile program
 * make release     -  switch to release build and compile program
 * make edit_cache  -  runs ccmake in the build directory
 * make edit        -  open build/run/ryujin.prm in default editor
 * make run         -  run the program (with default config file build/run/ryujin.prm)
 * @endcode
 */
