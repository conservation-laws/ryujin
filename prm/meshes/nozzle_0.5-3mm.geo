// Gmsh project created on Fri Jun 21 11:10:30 2024
Point(1) = {-0.10, -0.145, 0.0, 1};
Point(2) = {0.10, -0.145, 0.0, 1};
Point(3) = {-0.025, -0.1, 0.0, 1};
Point(4) = {0.025, -0.1, 0.0, 1};
Point(5) = {-0.025, 0., 0.0, 1};
Point(6) = {0.025, 0., 0.0, 1};
Point(7) = {-0.15, 1., 0.0, 1};
Point(8) = {0.15, 1., 0.0, 1};
Point(9) = {-0.4, 1., 0.0, 1};
Point(10) = {0.4, 1., 0.0, 1};
Point(11) = {-0.4, 1.55, 0.0, 1};
Point(12) = {0.4, 1.55, 0.0, 1};
Point(13) = {-0.1, -0.425, 0., 1};
Point(14) = {0.1, -0.425, 0., 1};
Line(3) = {3, 1};
//+
Line(4) = {4, 2};
//+
Line(5) = {11, 9};
//+
Line(6) = {7, 9};
//+
Line(7) = {11, 12};
//+
Line(8) = {12, 10};
//+
Line(9) = {10, 8};
//+
Line(10) = {7, 8};
//+
Line(11) = {3, 4};
//+
Line(12) = {1, 2};

//+
Line(13) = {7, 5};
//+
Line(14) = {8, 6};
//+
Line(15) = {5, 3};
//+
Line(16) = {6, 4};
//+
Line(17) = {5, 6};
//+
Line(18) = {1, 13};
//+
Line(19) = {13, 14};
//+
Line(20) = {14, 2};
//+
Curve Loop(1) = {6, -5, 7, 8, 9, -10};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {13, 17, -14, -10};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {15, 11, -16, -17};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {3, 12, -4, -11};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {18, 19, 20, -12};
//+
Plane Surface(5) = {5};
//+

//+
Physical Surface("mesh", 21) = {1, 2, 3, 5, 4};
//+
Physical Curve(3) = {5, 6, 8, 9, 14, 13, 15, 16, 3, 4, 18, 20};
//+
Physical Curve(4) = {19};
//+
Physical Curve(0) = {7};
