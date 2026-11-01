// Blast over a mountain, Section 4.2.4 (Figure 4.7) of [1]:
// D = [-20,20] x [0,20] km^2 with a cosine shaped mountain at the bottom,
//
//   y_b(x) = h0 / 2 (1 + cos(pi x / a))  for |x| <= a,  y_b = 0 otherwise,
//
// with h0 = 1.16 km and a = 3.4 km. All boundaries get boundary id 2
// (slip); h = lc.
//
// Generate rectangle_mountain.msh with:
//   gmsh -2 -format msh22 -o rectangle_mountain.msh \
//     2026-FMT-4.2.4-rectangle_mountain.geo
//
// [1] C. Farris, M. Maier, E. J. Tovar. Well-balanced second-order
//     approximation of the compressible atmospheric Euler equations.
//     arXiv:2606.19764, 2026.

h0 = 1.16;
a = 3.4;
lc = 0.1;
n = 68;

Point(1) = {-20, 0, 0, lc};
Point(2) = {-a, 0, 0, lc};
For i In {1:n-1}
  x = -a + 2 * a * i / n;
  Point(2 + i) = {x, h0 / 2 * (1 + Cos(Pi * x / a)), 0, lc};
EndFor
Point(n + 2) = {a, 0, 0, lc};
Point(n + 3) = {20, 0, 0, lc};
Point(n + 4) = {20, 20, 0, lc};
Point(n + 5) = {-20, 20, 0, lc};

Line(1) = {1, 2};
Spline(2) = {2:n+2};
Line(3) = {n + 2, n + 3};
Line(4) = {n + 3, n + 4};
Line(5) = {n + 4, n + 5};
Line(6) = {n + 5, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

Physical Curve(2) = {1, 2, 3, 4, 5, 6};
Physical Surface(1) = {1};
