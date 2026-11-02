// Schardin's problem, Section 5.5 (Figure 3) of [1]: the domain
// D = (-0.65,0.5) x (-0.5,0.5) without the triangle with vertices
// (-0.2,0), (0.1,1/6), (0.1,-1/6), meshed with quadrilaterals.
//
// Boundary ids: 4 (dirichlet) on the left, 5 (dynamic) on the right, and
// 2 (slip) on the top, bottom, and the triangle.
//
// Generate schardin_triangle.msh with:
//   gmsh -2 -format msh22 -o schardin_triangle.msh \
//     2023-CGMPT-5.5-schardin_triangle.geo
//
// [1] B. Clayton, J.-L. Guermond, M. Maier, B. Popov, E. J. Tovar. Robust
//     second-order approximation of the compressible Euler equations with
//     an arbitrary equation of state. J. Comput. Phys., 478:111926, 2023.
//     DOI 10.1016/j.jcp.2023.111926, arXiv:2207.12832

lc = 0.072;

Point(1) = {-0.65, -0.5, 0, lc};
Point(2) = {0.5, -0.5, 0, lc};
Point(3) = {0.5, 0.5, 0, lc};
Point(4) = {-0.65, 0.5, 0, lc};

Point(5) = {-0.2, 0, 0, lc / 2};
Point(6) = {0.1, -1 / 6, 0, lc / 2};
Point(7) = {0.1, 1 / 6, 0, lc / 2};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 5};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7};
Plane Surface(1) = {1, 2};

Recombine Surface {1};
Mesh.RecombinationAlgorithm = 1;
Mesh.SubdivisionAlgorithm = 1;

Physical Curve(4) = {4};
Physical Curve(5) = {2};
Physical Curve(2) = {1, 3, 5, 6, 7};
Physical Surface(1) = {1};
