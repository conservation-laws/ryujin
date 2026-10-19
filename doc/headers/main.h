/**
 * @mainpage
 *
 * This is the main starting page for the technical ryujin class and
 * function documentation. A full usage guide can be found at
 * https://conservation-laws.org/ryujin/
 *
 * The source code is organized in the following parts. An arrow points
 * from a part of the code to a part it uses. Everything uses the base
 * layer:
 *
 * @dot
 digraph G
{
  graph[rankdir="TB",bgcolor="transparent",compound=true,nodesep=0.3];

  node [fontname="FreeSans",fontsize=13,
        shape=record,height=0.2,width=0.4,
        color="black", fillcolor="white", style="filled"];
  edge [color="black",fontname="FreeSans",fontsize=11];

  subgraph cluster_driver {
    label="Driver and Output"; URL="\ref DriverLayer";
    fontname="FreeSans"; style="rounded"; color="gray";
    driver [label="Driver", URL="\ref Driver"];
    output [label="Output", URL="\ref Output"];
  }

  subgraph cluster_equations {
    label="Equations"; URL="\ref EquationsLayer";
    fontname="FreeSans"; style="rounded"; color="gray";
    equations [label="{Euler | Euler AEOS | Euler barotropic | Euler-Poisson | Navier-Stokes | Scalar conservation | Shallow water}",
               URL="\ref EquationsLayer"];
  }

  subgraph cluster_framework {
    label="Framework"; URL="\ref FrameworkLayer";
    fontname="FreeSans"; style="rounded"; color="gray";
    typed      [label="Typed implementations", URL="\ref Typed"];
    concepts   [label="Concepts", URL="\ref Concepts"];
    interfaces [label="Interfaces", URL="\ref Interfaces"];
  }

  subgraph cluster_discretization {
    label="Discretization"; URL="\ref DiscretizationLayer";
    fontname="FreeSans"; style="rounded"; color="gray";
    mesh       [label="Mesh, offline data,\ntime integration", URL="\ref MeshAndOfflineData"];
    fe         [label="Finite element helpers", URL="\ref FiniteElement"];
    geometries [label="Geometries", URL="\ref Geometries"];
  }

  subgraph cluster_base {
    label="Base"; URL="\ref BaseLayer";
    fontname="FreeSans"; style="rounded"; color="gray";
    linalg    [label="Linear algebra", URL="\ref LinearAlgebra"];
    simd      [label="SIMD", URL="\ref SIMD"];
    gpu       [label="GPU support", URL="\ref GPU"];
    utilities [label="Utilities", URL="\ref Utilities"];
  }

  driver -> equations [label="selects"];
  driver -> typed [label="instantiates"];
  equations -> typed [label="initial states"];
  equations -> concepts [label="satisfy", style="dashed"];
  typed -> concepts [label="constrained by"];
  typed -> interfaces [label="implements"];
  mesh -> interfaces [label="implements"];
  output -> interfaces;
  mesh -> fe;
  mesh -> geometries;
  interfaces -> linalg [lhead=cluster_base];
}
 * @enddot
 *
 * At the bottom, the @ref BaseLayer "base layer" provides SIMD
 * vectorization, data that is mirrored between the host and device memory
 * spaces, and distributed vectors and sparse matrices. The
 * @ref DiscretizationLayer "discretization layer" builds the mesh, the
 * finite element space, and the precomputed offline data on top of it.
 * The @ref FrameworkLayer "framework" connects equations with the
 * discretization: abstract @ref Interfaces "interfaces" hide all equation
 * dependent types, @ref Concepts "concepts" specify what an equation has
 * to provide, and @ref Typed "typed implementations" carry out the
 * hyperbolic update, initial values, and postprocessing for a given
 * hyperbolic and parabolic description. Each of the
 * @ref EquationsLayer "equations" is such a pair of descriptions. Finally,
 * the @ref Driver "driver" selects an equation at run time, sets up the
 * simulation, and runs the time loop, and the @ref Output "output" classes
 * write out results.
 *
 * ryujin is based on discretization approaches and algorithms that have
 * been developed in a number of publications
 * \cite GuermondPopov2016
 * \cite GuermondPopov2016b
 * \cite GuermondEtAl2018
 * \cite GuermondEtAl2018SW
 * \cite ryujin-2021-1
 * \cite ryujin-2021-2
 * \cite ryujin-2021-3
 * \cite ClaytonGuermondPopov-2022
 * \cite ryujin-2023-4.
 * A complete list of references can be found in the \ref citelist.
 */
