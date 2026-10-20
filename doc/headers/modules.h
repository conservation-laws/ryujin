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
 * At run time, the objects are created and owned as follows:
 *
 * @dot
 digraph G
{
  graph[rankdir="LR",bgcolor="transparent"];

  node [fontname="FreeSans",fontsize=12,
        shape=record,height=0.2,width=0.4,
        color="black", fillcolor="white", style="filled"];
  edge [color="black",fontname="FreeSans",fontsize=11];

  dispatch   [label="EquationDispatch", URL="\ref ryujin::EquationDispatch"];
  equation   [label="Equation\<HD, PD\>", URL="\ref ryujin::Equation"];
  loop       [label="Scheduler", URL="\ref ryujin::Scheduler"];
  simulation [label="Simulation\<HD, PD\>", URL="\ref ryujin::Simulation"];

  ensemble   [label="MPIEnsemble", URL="\ref ryujin::MPIEnsemble"];
  discr      [label="Discretization", URL="\ref ryujin::Discretization"];
  offline    [label="OfflineData", URL="\ref ryujin::OfflineData"];
  hyperbolic [label="HyperbolicModule", URL="\ref ryujin::Interface::HyperbolicModule"];
  parabolic  [label="ParabolicModule", URL="\ref ryujin::Interface::ParabolicModule"];
  integrator [label="TimeIntegrator", URL="\ref ryujin::TimeIntegrator"];
  adaptor    [label="MeshAdaptor", URL="\ref ryujin::MeshAdaptor"];
  transfer   [label="SolutionTransfer", URL="\ref ryujin::Interface::SolutionTransfer"];
  output     [label="VTUOutput | Quantities | ErrorEvaluation", URL="\ref Output"];

  dispatch -> equation [label="selects"];
  equation -> loop [label="runs"];
  loop -> simulation [label="drives"];
  equation -> simulation [label="creates"];
  simulation -> ensemble;
  simulation -> discr;
  simulation -> offline;
  simulation -> hyperbolic;
  simulation -> parabolic;
  simulation -> integrator;
  simulation -> adaptor;
  simulation -> transfer;
  simulation -> output;
}
 * @enddot
 */


/**
 * @defgroup Interfaces Interfaces
 * @ingroup FrameworkLayer
 *
 * Abstract base classes in namespace ryujin::Interface that hide the
 * equation dependent types from the driver and the discretization
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
 * Selecting an equation, setting up and running the simulation, and
 * writing out results. The corresponding headers are found in
 * <code>include/ryujin/driver</code> and
 * <code>include/ryujin/output</code>.
 */


/**
 * @defgroup Driver Driver
 * @ingroup DriverLayer
 *
 * The EquationDispatch class that selects an equation by name, the
 * library of available equations, the Simulation, and the Scheduler
 * (<code>include/ryujin/driver</code>).
 */


/**
 * @defgroup Output Output
 * @ingroup DriverLayer
 *
 * Writing out VTU files, computing and recording quantities of interest,
 * and evaluating errors against an analytic solution
 * (<code>include/ryujin/output</code>).
 */
