/**
 * @defgroup BaseLayer Base
 *
 * Basic building blocks that do not depend on any other part of ryujin:
 * SIMD support, memory management for the host and device memory spaces,
 * linear algebra, and a collection of general utilities. The
 * corresponding headers are found in <code>include/ryujin/base</code> and
 * <code>include/ryujin/linear_algebra</code>.
 */


/**
 * @defgroup SIMD SIMD
 * @ingroup BaseLayer
 *
 * SIMD related functions and classes (<code>base/simd.h</code>).
 */


/**
 * @defgroup GPU GPU support
 * @ingroup BaseLayer
 *
 * This module contains classes and functions used for managing data
 * mirrored between the host and default (device) memory spaces: the
 * MirroredStorage base class implementing a unified interface for
 * querying residency and transferring data between memory spaces, the
 * TransferPolicy describing whether such transfers happen explicitly or
 * implicitly, and the Mirrored convenience wrapper that maintains a
 * single "POD style" object on both memory spaces (<code>base/gpu.h</code>).
 */


/**
 * @defgroup Utilities Utilities
 * @ingroup BaseLayer
 *
 * Helper functions, macros and classes: loop abstractions for the host and
 * device memory spaces, MPI ensembles, timers and instrumentation, a
 * Newton solver, lazy initialization, and parameter conversion
 * (<code>include/ryujin/base</code>).
 */


/**
 * @defgroup LinearAlgebra Linear Algebra
 * @ingroup BaseLayer
 *
 * This module contains the classes and functions used for storing and
 * operating on linear algebra objects: distributed vectors that hold a
 * fixed number of components per vector entry, the state vector, the
 * sparsity pattern and sparse matrix used for the stencil-based
 * operations, and the corresponding "view" classes that hand out access
 * to the underlying data in either the host or the device memory space
 * (<code>include/ryujin/linear_algebra</code>).
 */


/**
 * @defgroup DiscretizationLayer Discretization
 *
 * Mesh generation, the finite element discretization, precomputed offline
 * data, mesh adaptation, and time integration. The corresponding headers
 * are found in <code>include/ryujin/discretization</code> and
 * <code>include/ryujin/geometries</code>.
 */


/**
 * @defgroup MeshAndOfflineData Mesh, offline data, and time integration
 * @ingroup DiscretizationLayer
 *
 * The Discretization class holding the triangulation and finite element,
 * the OfflineData class holding the precomputed matrices and sparsity
 * patterns, the MeshAdaptor, and the TimeIntegrator
 * (<code>include/ryujin/discretization</code>).
 */


/**
 * @defgroup FiniteElement Finite element helpers
 * @ingroup DiscretizationLayer
 *
 * Some helper functions for local index handling, dof renumbering,
 * sparsity pattern creation, reference cells, and tensor product point
 * kernels (<code>include/ryujin/discretization</code>).
 */


/**
 * @defgroup Geometries Geometries
 * @ingroup DiscretizationLayer
 *
 * The library of computational domains that can be selected with the
 * run time parameter "geometry", together with the helper classes they
 * use (<code>include/ryujin/geometries</code>).
 */


/**
 * @defgroup FrameworkLayer Framework
 *
 * The framework connects the equation descriptions with the
 * discretization. It consists of three parts:
 *
 *  - @ref Interfaces "Interfaces" (<code>include/ryujin/interface</code>):
 *    abstract base classes that are independent of the equation. The
 *    driver and the discretization only use these.
 *  - @ref Concepts "Concepts" (<code>include/ryujin/concept</code>): the
 *    C++20 concepts a hyperbolic description (HD) or a parabolic
 *    description (PD) has to satisfy.
 *  - @ref Typed "Typed implementations" (<code>include/ryujin/typed</code>
 *    and <code>include/ryujin/stub</code>): implementations of the
 *    interfaces that are templated on HD and PD.
 *
 * At run time, the objects are created and owned as follows. The ryujin
 * executable (<code>source/main.cc</code>) selects one of the registered
 * equations and the dimension with the run time parameters "equation" and
 * "dimension"; the equation constructs the Simulation and the
 * TaskComposition. A client program constructs them directly:
 *
 * @dot
 digraph G
{
  graph[rankdir="LR",bgcolor="transparent"];

  node [fontname="FreeSans",fontsize=12,
        shape=record,height=0.2,width=0.4,
        color="black", fillcolor="white", style="filled"];
  edge [color="black",fontname="FreeSans",fontsize=11];

  composition [label="TaskComposition", URL="\ref ryujin::TaskComposition"];
  loop        [label="Scheduler", URL="\ref ryujin::Scheduler"];
  simulation  [label="Simulation\<HD, PD\>", URL="\ref ryujin::Simulation"];
  tasks       [label="Logger | MeshAdaptor | VTUOutput | Quantities | ErrorEvaluation | Checkpointing", URL="\ref ryujin::Interface::Task"];

  ensemble    [label="MPIEnsemble", URL="\ref ryujin::MPIEnsemble"];
  discr       [label="Discretization", URL="\ref ryujin::Discretization"];
  offline     [label="OfflineData", URL="\ref ryujin::OfflineData"];
  hyperbolic  [label="HyperbolicModule", URL="\ref ryujin::Interface::HyperbolicModule"];
  parabolic   [label="ParabolicModule", URL="\ref ryujin::Interface::ParabolicModule"];
  integrator  [label="TimeIntegrator", URL="\ref ryujin::TimeIntegrator"];
  transfer    [label="SolutionTransfer", URL="\ref ryujin::Interface::SolutionTransfer"];

  composition -> simulation;
  composition -> tasks;
  composition -> loop;
  loop -> simulation [label="runs"];
  loop -> tasks [label="runs"];
  tasks -> simulation [label="use"];
  simulation -> ensemble;
  simulation -> discr;
  simulation -> offline;
  simulation -> hyperbolic;
  simulation -> parabolic;
  simulation -> integrator;
  simulation -> transfer;
}
 * @enddot
 */


/**
 * @defgroup Interfaces Interfaces
 * @ingroup FrameworkLayer
 *
 * Abstract base classes in namespace ryujin::Interface that hide the
 * equation dependent types from the driver and the discretization, and
 * Interface::Task, the interface of all tasks that the Scheduler runs
 * (<code>include/ryujin/interface</code>).
 */


/**
 * @defgroup Concepts Concepts
 * @ingroup FrameworkLayer
 *
 * C++20 concepts in namespace ryujin::Concept that a hyperbolic
 * description and a parabolic description have to satisfy: the
 * hyperbolic system, indicator, limiter, wave speed estimator, and the
 * parabolic system and module (<code>include/ryujin/concept</code>).
 */


/**
 * @defgroup Typed Typed implementations
 * @ingroup FrameworkLayer
 *
 * Implementations of the interfaces in namespace ryujin that are
 * templated on the hyperbolic description HD and the parabolic
 * description PD: the explicit hyperbolic update, initial values,
 * solution transfer, and postprocessing. It also contains the stub
 * parabolic module used by equations without a parabolic part
 * (<code>include/ryujin/typed</code> and
 * <code>include/ryujin/stub</code>).
 */


/**
 * @defgroup EquationsLayer Equations
 *
 * The hyperbolic and parabolic descriptions of all equations that ryujin
 * can solve. Each equation lives in its own subdirectory of
 * <code>include/ryujin</code> and provides a hyperbolic system, an
 * indicator, a limiter, a wave speed estimator, and a library of initial
 * states. Some also provide a parabolic system and module.
 */


/**
 * @defgroup EulerEquations The Compressible Euler Equations
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving the
 * compressible Euler equations of gas dynamics with a polytropic gas
 * equation of state (<code>include/ryujin/euler</code>).
 */


/**
 * @defgroup EulerAEOSEquations The Euler Equations with Arbitrary Equation of State
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving the
 * compressible Euler equations of gas dynamics with an arbitrary equation
 * of state (<code>include/ryujin/euler_aeos</code>).
 */


/**
 * @defgroup EulerBarotropicEquations The Barotropic Euler Equations
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving the
 * barotropic Euler equations, where the pressure is a function of the
 * density only (<code>include/ryujin/euler_barotropic</code>).
 */


/**
 * @defgroup EulerPoissonEquations The Euler-Poisson Equations
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving the
 * compressible Euler equations coupled to a Poisson problem for the
 * electrostatic potential (<code>include/ryujin/euler_poisson</code>).
 */


/**
 * @defgroup NavierStokesEquations The Compressible Navier Stokes Equations
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving the
 * compressible Navier Stokes equations
 * (<code>include/ryujin/navier_stokes</code>).
 */


/**
 * @defgroup ScalarConservationEquations The Scalar Conservation Equations
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving various
 * scalar conservation equations
 * (<code>include/ryujin/scalar_conservation</code>).
 */


/**
 * @defgroup ShallowWaterEquations The Shallow Water Equations
 * @ingroup EquationsLayer
 *
 * This module contains classes and functions related to solving the
 * shallow water equations (<code>include/ryujin/shallow_water</code>).
 */


/**
 * @defgroup DriverLayer Driver and Output
 *
 * Setting up and running the simulation, and writing out results. The corresponding headers are found in
 * <code>include/ryujin/driver</code> and
 * <code>include/ryujin/output</code>.
 */


/**
 * @defgroup Driver Driver
 * @ingroup DriverLayer
 *
 * The Environment that initializes MPI, the thread pools, and the
 * instrumentation, the registry of equations, the Simulation, the
 * TaskComposition that assembles the Simulation and the tasks, the
 * Scheduler that runs them, and write_parameter_files() that writes the
 * default parameter files (<code>include/ryujin/driver</code>).
 *
 * <h3>Equation dispatch</h3>
 *
 * The ryujin executable selects the equation and the dimension at run
 * time. Every equation registers itself with a constant
 * EquationRegistration<HD, PD> at namespace scope in
 * <code>source/driver/equation_*.cc</code>. Its constructor runs during
 * static initialization and inserts an Equation under the given name into
 * the map returned by equations(). An Equation holds three std::function
 * objects, so that main() (<code>source/main.cc</code>) can call them
 * without knowing HD, PD, or the dimension. main() reads the run time
 * parameters "dimension" and "equation", looks up the equation, and calls
 * Equation::run, which switches over the dimension and constructs the
 * Simulation and the TaskComposition. If the parameter file does not
 * exist, main() instead opens the parameter editor (see below) in a
 * serial run on an interactive terminal, and otherwise calls
 * Equation::create_parameter_files of every registered equation:
 *
 * @dot
 digraph G
{
  graph[rankdir="TB",bgcolor="transparent",nodesep=0.4];

  node [fontname="FreeSans",fontsize=12,
        shape=record,height=0.2,width=0.4,
        color="black", fillcolor="white", style="filled"];
  edge [color="black",fontname="FreeSans",fontsize=11];

  main      [label="main()\nreads \"dimension\" and \"equation\""];
  registry  [label="equations()", URL="\ref ryujin::equations"];
  equation  [label="{Equation | {<run> run(dim, ...) | <prm> create_parameter_files()}}", URL="\ref ryujin::Equation"];

  subgraph cluster_registration {
    label="EquationRegistration\<HD, PD\>"; URL="\ref ryujin::EquationRegistration";
    fontname="FreeSans"; style="rounded"; color="gray";
    constructor [label="constructor(name)", URL="\ref ryujin::EquationRegistration"];
    run         [label="run_simulation\<dim\>()", URL="\ref ryujin::EquationRegistration"];
    prm         [label="create_parameter_file\<dim\>()", URL="\ref ryujin::EquationRegistration"];
  }

  simulation  [label="Simulation\<HD, PD, dim\>", URL="\ref ryujin::Simulation"];
  composition [label="TaskComposition\<dim\>\nsees Interface::Simulation", URL="\ref ryujin::TaskComposition"];
  scheduler   [label="Scheduler::run()", URL="\ref ryujin::Scheduler"];
  write       [label="write_parameter_files()", URL="\ref ryujin::write_parameter_files"];

  registry -> constructor [label="inserts Equation", dir="back"];
  main -> registry [label="at(equation)"];
  registry -> equation;
  equation:run -> run [label="switch over dim"];
  equation:prm -> prm [label="dim = 1, 2, 3"];
  run -> simulation [label="constructs"];
  run -> composition [label="constructs"];
  composition -> scheduler [label="run()"];
  prm -> write;
}
 * @enddot
 *
 * Only EquationRegistration and the Simulation depend on HD and PD. The
 * TaskComposition and all tasks use the Simulation through
 * Interface::Simulation and are compiled once per dimension.
 *
 * <h3>Parameter editor</h3>
 *
 * With <code>ryujin --edit [file]</code> (or <code>--editor</code>), or if
 * the parameter file does not exist in a serial run on an interactive
 * terminal, main() calls edit_parameter_file(). Without a parameter file it
 * starts from the first equation in 2D. Equation::with_parameters constructs
 * the Simulation and the TaskComposition to declare all parameters, which
 * the editor shows in a terminal editor (ParameterModel,
 * run_parameter_editor()) with the colors and the animation of the
 * Dashboard, as configured in the subsection "M - Logger". Changing the
 * equation or the dimension opens the editor again for the new equation,
 * keeping all values that still apply. "save" writes the parameter file,
 * "save as" writes it under a new name and continues with that file,
 * "load" opens the editor again with the values of a different parameter
 * file, and "quit" leaves the editor (and ryujin) without running the
 * simulation. Before discarding unsaved changes the editor asks for a
 * confirmation.
 *
 * <h3>Splash screen</h3>
 *
 * Before running the equation, main() shows a splash screen on rank 0,
 * see play_splash_screen(), if stdout reaches an interactive terminal.
 * The option <code>--splash=&lt;name&gt;</code> selects it: "logo" (the
 * default) shows the logo, the name, and the copyright of ryujin, "arcade"
 * plays an intro in the style of an arcade game of the eighties and waits
 * for a key press, and "none" skips it.
 *
 * <h3>Scheduler</h3>
 *
 * The Scheduler runs a list of tasks (see Interface::Task). The active
 * tasks are the required tasks, followed by the tasks named in the run
 * time parameter "tasks". After calling initialize(), start(), and
 * prepare() on every active task, the Scheduler runs cycles of four
 * phases. Each task belongs to exactly one TaskPhase; within a phase the
 * Scheduler executes the tasks of that phase in the order of the active
 * tasks. The diagram lists the tasks that TaskComposition assigns to each
 * phase:
 *
 * @dot
 digraph G
{
  graph[rankdir="TB",bgcolor="transparent",nodesep=0.4];

  node [fontname="FreeSans",fontsize=12,
        shape=record,height=0.2,width=0.4,
        color="black", fillcolor="white", style="filled"];
  edge [color="black",fontname="FreeSans",fontsize=11];

  select    [label="select the active tasks:\nrequired tasks, then the tasks\nof the run time parameter \"tasks\""];
  setup     [label="{initialize() | start() | prepare()}", URL="\ref ryujin::Interface::Task"];
  pre_step  [label="{TaskPhase::pre_step | VTUOutput, Quantities,\nErrorEvaluation}", URL="\ref ryujin::TaskPhase"];
  final     [label="t \>= t_final?", shape=diamond];
  adapt     [label="{TaskPhase::adapt | MeshAdaptor}", URL="\ref ryujin::TaskPhase"];
  changed   [label="prepare_required?", shape=diamond];
  prepare   [label="prepare()", URL="\ref ryujin::Interface::Task"];
  step      [label="{TaskPhase::step | Simulation (chooses tau)}", URL="\ref ryujin::TaskPhase"];
  advance   [label="t += tau,\nsynchronize wall time"];
  post_step [label="{TaskPhase::post_step | Logger, Checkpointing}", URL="\ref ryujin::TaskPhase"];
  finalize  [label="finalize()\nin reverse order", URL="\ref ryujin::Interface::Task"];

  select -> setup;
  setup -> pre_step;
  pre_step -> final;
  final -> finalize [label="yes"];
  final -> adapt [label="no"];
  adapt -> changed;
  changed -> prepare [label="yes"];
  changed -> step [label="no"];
  prepare -> step;
  step -> advance;
  advance -> post_step;
  post_step -> pre_step [label="next cycle", constraint=false];
}
 * @enddot
 *
 * The first required task is the primary task. In TaskComposition this is
 * the Simulation, which sets up the discretization in prepare() and
 * chooses the time step size tau in the step phase. All values the
 * Scheduler shares with the tasks, such as t, tau, the cycle, and
 * whether a timer tick has been reached, are stored in the TaskContext.
 * Tasks decide in which cycles to do their work with a Trigger.
 */


/**
 * @defgroup Output Output
 * @ingroup DriverLayer
 *
 * Writing out VTU files, computing and recording quantities of interest,
 * and evaluating errors against an analytic solution
 * (<code>include/ryujin/output</code>).
 */
