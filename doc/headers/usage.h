// clang-format off

/**
 * @page Usage Usage instructions
 *
 * <h4>Convenience Makefile</h4>
*
 * The <code>Makefile</code> found in the repository only contains a number
 * of convenience targets and its use is entirely optional. It will create
 * a subdirectory <code>build</code> and run cmake to configure the
 * project. The executable will be located in <code>build/run</code>. The
 * convenience Makefile contains the following additional targets:
 * @code
 * make debug       -  switch to debug build and compile program
 * make release     -  switch to release build and compile program
 * make edit_cache  -  runs ccmake in the build directory
 * make edit        -  open build/run/ryujin.prm in default editor
 * make run         -  run the program (with default config file build/run/ryujin.prm)
 * @endcode
 *
 *
 * <h4>Runtime parameter files</h4>
 *
 * The build system creates an executable <code>build/run/ryujin</code>
 * that takes a parameter file location as single (optional) argument. If
 * run without an argument it will try to open a parameter file
 * <code>ryujin.prm</code> in the working directory of the executable (that
 * is <code>build/run</code>):
 * @code
 * cd build/run
 * ./ryujin   # uses ryujin.prm
 * ./ryujin my_parameter_file.prm
 * @endcode
 * You can find a number of example parameter files in the <code>prm</code>
 * subdirectory:
 * @code
 *   default.prm                  - all available parameters and their default values
 *   euler-validation.prm         - analytic isentropic vortex solution for validating the Euler configuration
 *   navier_stokes-validation.prm - analytic Becker solution for validating the Navier-Stokes configuration
 *   navier_stokes-airfoil.prm    - compute C_F/C_P values for the ONERA OAT15a airfoil
 *   navier_stokes-shocktube.prm  - Navier Stokes shocktube benchmark (see Daru & Tenaud, Computers & Fluids, 38(3):664-676, 2009)
 * @endcode
 *
 * <h4>Compile time options</h4>
 *
 * You can set and change compile time options by specifying them on the
 * command line during configuration with <code>cmake</code>, or
 * conveniently after an initial configure via <code>ccmake</code>, for
 * example by using the makefile command
 *
 * @code
 * make edit_cache
 * @endcode
 *
 * or by calling ccmake by hand:
 *
 * @code
 * cd build
 * ccmake .
 * @endcode
 *
 * The most important compile-time options are <code>DIM</code> to select
 * the spatial dimension ("1","2", or "3") and
 * <code>CMAKE_BUILD_TYPE</code> to switch between a debug build ("debug")
 * or a release build ("release"). Other compile-time options are
 * <code>NUMBER</code> to specify whether the program should use
 * double-precision ("double") or single-precision ("float") arithmetic,
 * and various debugging and instrumentation toggles:
 * @code
 * CMAKE_BUILD_TYPE             - build ryujing in "Release" or "Debug" mode
 * DIM                          - the spatial dimension, default to 2
 * NUMBER                       - select "double" for double precision or "float" for single precision
 *
 * ASYNC_MPI_EXCHANGE           - enable asynchronous "communication hiding" MPI exchange (defaults to OFF)
 * CHECK_BOUNDS                 - enable additional bounds checking (defaults to OFF)
 * CUSTOM_POW                   - use a custom SIMD implementation also for serial pow (default to ON)
 * DEBUG_OUTPUT                 - enable debug output (defaults to OFF)
 * DENORMALS_ARE_ZERO           - disable floating point denormals (defaults to ON)
 * FORCE_DEAL_II_SPARSE_MATRIX  - always use deal.II sparse matrix for preliminary assembly instead of Trilinos
 *
 * WITH_CALLGRIND               - enable Valgrind/Callgrind stetoscope mode (default to OFF)
 * WITH_DOXYGEN                 - enable support for doxygen and build documentation
 * WITH_EOSPAC                  - enable support for the EOSPAC6/Sesame tabulated equation of state database (autodetection)
 * WITH_LIKWID                  - enable support for Likwid stetoscope mode (library for Intel performance counters, defaults to OFF)
 * WITH_OPENMP                  - enable support for multithreading via OpenMP (autodetection)
 * @endcode
 *
 * Most of the other compile-time options are marked as advanced and only
 * visible in ccmake by changing to the advanced view (press "t"). They are
 * best kept at their default values.
 */
