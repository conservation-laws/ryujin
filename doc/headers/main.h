/**
 * @mainpage
 *
 * This is the main starting page for the ryujin class and function
 * documentation. The program is organized into the following modules:
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
  mesh  [label="Mesh generation and discretization", URL="\ref Mesh"];
  hyperbolic [label="Hyperbolic Module", URL="\ref HyperbolicModule"];
  parabolic [label="Parabolic Module", URL="\ref ParabolicModule"];
  loop  [label="Time loop", URL="\ref TimeLoop"];

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
}
 * @enddot
 *
 * In addition, the doxygen documentation contains information about
 * \ref Installation and \ref Usage.
 */
