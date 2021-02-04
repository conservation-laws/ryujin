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

  copt  [label="Compile time options", URL="\ref CompileTimeOptions"];
  simd  [label="SIMD", URL="\ref SIMD"];
  fe    [label="Finite element formulation", URL="\ref FiniteElement"];
  misc  [label="Miscellaneous", URL="\ref Miscellaneous"];
  mesh  [label="Mesh generation and discretization", URL="\ref Mesh"];
  euler [label="Euler Module", URL="\ref EulerModule"];
  dissipation [label="Dissipation Module", URL="\ref DissipationModule"];
  loop  [label="Time loop", URL="\ref TimeLoop"];

  fe   -> mesh  [color="black",style="solid"];
  copt -> euler [color="black",style="solid"];
  simd -> euler [color="black",style="solid"];
  fe   -> euler [color="black",style="solid"];
  misc -> euler [color="black",style="solid"];
  mesh -> euler [color="black",style="solid"];
  copt -> dissipation [color="black",style="solid"];
  simd -> dissipation [color="black",style="solid"];
  fe   -> dissipation [color="black",style="solid"];
  misc -> dissipation [color="black",style="solid"];
  mesh -> dissipation [color="black",style="solid"];
  misc -> loop  [color="black",style="solid"];
  euler -> loop [color="black",style="solid"];
  dissipation -> loop [color="black",style="solid"];
}
 * @enddot
 *
 * In addition, the doxygen documentation contains information about
 * \ref Installation and \ref Usage.
 */
