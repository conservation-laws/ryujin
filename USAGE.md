Usage Usage instructions
========================

Convenience Makefile
--------------------

The `Makefile` found in the repository only contains a number of
convenience targets and its use is entirely optional. It will create a
subdirectory `build` and run cmake to configure the project. The executable
will be located in `build/run`. The convenience Makefile contains the
following additional targets:
  - `make debug`:  switch to debug build and compile program
  - `make release`:  switch to release build and compile program
  - `make edit_cache`:  runs ccmake in the build directory

Executable and runtime parameter files
--------------------------------------

After compiling ryujin you should end up with an executable located in the
build directory at `build/run/ryujin`. The executable takes a parameter
file location as single (optional) argument. If run without an argument it
will try to open a parameter file `ryujin.prm` in the working directory of
the executable (that is `build/run`):
```
cd build/run
./ryujin # uses ryujin.prm
./ryujin my_parameter_file.prm
```
You can find a number of example parameter files in the `prm` subdirectory.

Specifically, the `prm/verification` subdirectory contains parameter files
and <i>baseline</i> output vectors that document expected error and
convergence rates for various analytical solutions. All configurations
compare the simulation result at final time to a known analytic solution
and record the normalized L1, L2, and L\infty error norms (summed up over
all components). All test configurations should be run with double floating
point precision.

The `prm/benchmarks` directory contains parameter files for well known and
popular <i>benchmark</i> configurations. These configurations typically do
not have an analytical solution, but the expected solution structure is
well known. They are thus usually compared in the <i>eyeball norm</i>.
For example, to run the Mach 3 cylinder with 2.36M gridpoints on a machine
with 16 cores and two threads per core you could use:
```
cd build/run
DEAL_II_NUM_THREADS=2 mpirun -np 16 ryujin prm/benchmarks/euler-mach3-cylinder-2d.prm
```
Warning: This is a rather large computation. For quick tests you might want
to decrease the resolution by lowering the number set by `set mesh
refinement` in the prm file.

You can obtain a full list of supported runtime parameters and their
default values by invoking the `./ryujin` executable without creating a
`ryujin.prm` in the path:
```
% ./ryujin
[INFO] initiating flux capacitor
[INFO] Default parameter file »ryujin.prm« not found.
[INFO] Creating template parameter files...
% ls
[...]
default_parameters-euler-2d-description.prm
default_parameters-navier_stokes-2d-description.prm
default_parameters-shallow_water-2d-description.prm
default_parameters-euler_aeos-2d-description.prm
default_parameters-scalar_conservation-2d-description.prm
```
Consult the above parameter files with detailed annotated configuration
options (and their 1d and 3d counterparts) for details.

Output file format
------------------

ryujin supports outputting temporal snapshots in `.vtu` format. (In
principle you can modify ryujin to output any file format supported by
deal.II, have a look at `source/vtu_output.template.h`) You can use
Paraview to open and inspect `.vtu` files, which you can install via your
package manager or obtain [here](https://www.paraview.org/).

ryujin has some rudimentary support for outputting instantaneous, time
averaged, or space integrated primitive values (and their second moments)
on user defined level sets.

Controlling parallelism and screen output
-----------------------------------------

ryujin uses MPI and thread parallelization. You typically control the
degree of MPI parallelism by invoking ryujin with an MPI launcher program.
The number of threads created by each rank is controlled with the
`DEAL_II_NUM_THREADS` environment variable. For example, you can run ryujin
on 8 ranks with 4 threads each as follows:
```
DEAL_II_NUM_THREADS=4 mpirun -np 8 ./ryujin
```


Compile time options
--------------------

ryujin has a number of compile time options to fine tune certain behavior.
You can set and change compile time options by specifying them on the
command line during configuration with `cmake`, or `ccmake`. You can invoke
`ccmake` as follows:
```
cd build
ccmake .
```
You can also invoke `ccmake` conveniently with the convenience `Makefile`
from anywhere in the source directory by running
```
make edit_cache
```

The most important compile-time option is `CMAKE_BUILD_TYPE` that is used
to switch between a debug build configuration (`Debug`) and a release build
configuration (`Release`). This option can also be set quickly with the
convenience `Makefile` and typing `make debug`, or `make release` from the
top level directory of ryujin.

Changing other compile-time options is rarely needed. And if in doubt they
are best kept at their default values. For completeness we list all
configuration options here:
  - `CMAKE_BUILD_TYPE`: build ryujin in "Release" or "Debug" mode
  - `NUMBER`: select "double" for double precision or "float" for single precision (defaults to double)
  - `EXPENSIVE_BOUNDS_CHECK`: enable additional bounds checking (defaults to OFF)
  - `DEBUG_OUTPUT`: enable debug output (defaults to OFF)
  - `ASYNC_MPI_EXCHANGE`: enable asynchronous "communication hiding" MPI exchange (defaults to OFF)
  - `DENORMALS_ARE_ZERO`: disable floating point denormals (defaults to ON)
  - `FORCE_DEAL_II_SPARSE_MATRIX`: prefer deal.II sparse matrix for preliminary assembly instead of Trilinos
  - `SANITIZER`: enable address and UBSAN sanitizers for DEBUG build
  - `WITH_CALLGRIND`: enable Valgrind/Callgrind stetoscope mode (default to OFF)
  - `WITH_DOXYGEN`: enable support for doxygen and build documentation
  - `WITH_EOSPAC`: enable support for the EOSPAC6/Sesame tabulated equation of state database (autodetection)
  - `WITH_LIKWID`: enable support for Likwid stetoscope mode (library for Intel performance counters, defaults to OFF)
  - `WITH_OPENMP`: enable support for multithreading via OpenMP (autodetection)
  - `WITH_VALGRIND`: enable support for Valgrind profiling
