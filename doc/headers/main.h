/**
 * @mainpage
 *
 * This is the main starting page for the technical ryujin class and
 * function documentation. A full usage guide can be found at
 * https://conservation-laws.org
 *
 * The program is organized into the following modules:
 *
 * @dot
 digraph G
{
  graph[rankdir="TB",bgcolor="transparent"];

  node [fontname="FreeSans",fontsize=15,
        shape=record,height=0.2,width=0.4,
        color="black", fillcolor="white", style="filled"];
  edge [color="black", weight=10];

  simd  [label="SIMD", URL="\ref SIMD"];
  fe    [label="Finite element formulation", URL="\ref FiniteElement"];
  misc  [label="Miscellaneous", URL="\ref Miscellaneous"];
  mesh  [label="Discretization", URL="\ref Mesh"];
  hyperbolic [label="Hyperbolic Module", URL="\ref HyperbolicModule"];
  parabolic [label="Parabolic Module", URL="\ref ParabolicModule"];
  loop  [label="Time Integration and Postprocessing", URL="\ref TimeLoop"];

  formulation [label="PDE formulation", URL="\ref Description"];

  euler [label="Euler Equations", URL="\ref EulerEquations"];
  shallow [label="Shallow Water Equations", URL="\ref ShallowWaterEquations"];
  navier [label="Navier Stokes Equations", URL="\ref NavierStokesEquations"];

  fe   -> mesh  [color="black",style="solid"];
  simd -> hyperbolic [color="black",style="solid"];
  fe   -> hyperbolic [color="black",style="solid"];
  misc -> hyperbolic [color="black",style="solid"];
  mesh -> hyperbolic [color="black",style="solid"];
  simd -> parabolic [color="black",style="solid"];
  fe   -> parabolic [color="black",style="solid"];
  misc -> parabolic [color="black",style="solid"];
  mesh -> parabolic [color="black",style="solid"];
  misc -> loop  [color="black",style="solid"];
  hyperbolic -> loop [color="black",style="solid"];
  parabolic -> loop [color="black",style="solid"];
  euler -> formulation [color="blak",style="solid"];
  shallow -> formulation [color="black",style="solid"];
  navier -> formulation [color="black",style="solid"];
  formulation -> hyperbolic [color="black",style="solid"];
  formulation -> parabolic [color="black",style="solid"];
}
 * @enddot
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
