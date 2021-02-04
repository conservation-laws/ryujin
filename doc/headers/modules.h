/**
 * @defgroup CompileTimeOptions Compile time options
 *
 * Certain configuration options are provided as compile time constants to
 * improve the performance of critical computer kernel sections.
 */


/**
 * @defgroup Mesh Mesh generation and discretization
 *
 * Several classes and helper functions for creating meshes for a number of
 * benchmark configurations and controlling the finite element
 * discretization.
 */


/**
 * @defgroup InitialValues Initial values and manufactured solutions
 *
 * Several classes and helper functions for initial value configuration.
 */


/**
 * @defgroup FiniteElement Finite element formulation
 *
 * Some helper functions for local index handling, dof renumbering,
 * sparsity pattern and matrix assembly.
 */


/**
 * @defgroup EulerModule Euler Module
 *
 * This module contains classes and functions used during different stages
 * of the explicit Euler update performed in EulerStep::euler_step() and
 * higher-order time-stepping primitives based on the low-order)
 * EulerStep::euler_step() update.
 */


/**
 * @defgroup NavierModule Navier Stokes Module
 *
 * This module contains classes and functions used during different stages
 * of the implicit parabolic update in the Strang splitting.
 */


/**
 * @defgroup Miscellaneous Miscellaneous
 *
 * Miscellaneous helper functions, macros and classes.
 */


/**
 * @defgroup SIMD SIMD
 *
 * SIMD related functions and classes.
 */


/**
 * @defgroup TimeLoop Time loop
 *
 * This module contains classes and functions used in TimeLoop::run().
 */
