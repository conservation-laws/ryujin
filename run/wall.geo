Point(1) = {0.0,      0.0,      0, 1};
Point(2) = {0.166667, 0.0,      0, 1};
Point(3) = {3.2,      0.0,      0, 1};
Point(4) = {0.0,      0.166667, 0, 1};
Point(5) = {0.166667, 0.166667, 0, 1};
Point(6) = {3.2,      0.5,      0, 1};
Point(7) = {0.0,      1.0,      0, 1};
Point(8) = {3.2,      1.0,      0, 1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 6};
Line(4) = {6, 8};
Line(5) = {8, 7};
Line(6) = {7, 4};
Line(7) = {4, 1};

Line(8) = {4, 5};
Line(9) = {5, 6};

// Boundary IDs:
Physical Line(0) = {3, 4};
Physical Line(1) = {2};
Physical Line(2) = {5, 6, 7};

Line Loop(1) = {1, 2, 3, -9, -8, 7};
Line Loop(2) = {8, 9, 4, 5, 6};

Plane Surface(1) = {1};
Plane Surface(2) = {2};

// The physical surface:
Physical Surface(1) = {1, 2};

Transfinite Curve {9}  = 29 Using Progression 1.038;
Transfinite Curve {2}  = 29 Using Progression 1.038;
Transfinite Curve {-1} =  3 Using Progression 1.38;
Transfinite Curve {-8} =  3 Using Progression 1.38;
Transfinite Curve {7}  =  4 Using Progression 1.0;
Transfinite Curve {3}  =  6 Using Progression 1.0;

Transfinite Curve {5}  = 21 Using Progression 1.0;
Transfinite Curve {4}  =  3 Using Progression 1.0;
Transfinite Curve {6}  =  3 Using Progression 1.0;

// Meshing parameters:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 20;
Show "*";
