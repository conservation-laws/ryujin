Point(1) = {0.,       0., 0, 1};
Point(2) = {0.166667, 0., 0, 1};
Point(3) = {3.2,      0., 0, 1};
Point(4) = {3.2,      1., 0, 1};
Point(5) = {0.,       1., 0, 1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 1};

// Boundary IDs:
Physical Line(0) = {3};
Physical Line(1) = {2};
Physical Line(2) = {4, 5, 1};

Line Loop(1) = {1, 2, 3, 4, 5};
Plane Surface(1) = {1};

// The physical surface:
Physical Surface(1) = {1};

Transfinite Curve {-1} =  3 Using Progression 2.0;
Transfinite Curve {2} = 49 Using Progression 1.01455;
Transfinite Curve {3} = 12 Using Progression 1.059;
Transfinite Curve {4} = 25 Using Progression 1.0;
Transfinite Curve {-5} = 12 Using Progression 1.059;

// Meshing parameters:
Mesh.Algorithm = 8;
Mesh.RecombineAll = 1;
Mesh.CharacteristicLengthFactor = 0.09;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 20;
Show "*";
//+
