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
 * The build system create an executable <code>build/run/ryujin</code> that
 * takes an optional parameter file as single argument. If run without an
 * argument it will try to open a parameter file <code>ryujin.prm</code> in
 * the working directory of the executable (that is
 * <code>build/run</code>):
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
 *   navier_stokes-validation.prm - analytic becker solution for validating the Navier-Stokes configuration
 *   navier_stokes-airfoil.prm    - compute C_F/C_P values for the ONERA OAT15a airfoil
 *   navier_stokes-shocktube.prm  - Navier Stokes shocktube benchmark (see Daru & Tenaud, Computers & Fluids, 38(3):664-676, 2009)
 * @endcode
 *
 * <h4>Compile time options</h4>
 *
 * For a complete list of compile-time options have a look at the
 * @ref CompileTimeOptions "Compile Time Options page".
 *
 * You can set and change compile time options conveniently via
 * <code>ccmake</code>, for example by using the convenience makefile
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
 * and various debugging and instrumentation toggles.
 *
 * Most of the other compile-time options are marked as advanced and only
 * visible in ccmake by changing to the advanced view (press "t"). They are
 * best kept at their default values.
 */
