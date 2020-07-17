function rotated_ellipse = plotEllipse(params);
%%% calculate ellipse points from output of EyeCameraCalc1
%%% input:
%%% params = result of ellipse fit (X0,Y0,a,b, ~, ~, phi)
%%% output:
%%% rotated_ellipse(x,y) = ellipse points
%%% cmn 2020

%%% parse values from EllipseParams
X0= params(1);
Y0 = params(2);
a = params(3);
b = params(4);
phi = params(7);

%%% rotation matrix
R = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];

% the ellipse
theta_r         = linspace(0,2*pi,360);
ellipse_x_r     =  a*cos( theta_r );
ellipse_y_r     =  b*sin( theta_r );
rotated_ellipse = R * [ellipse_x_r;ellipse_y_r];
rotated_ellipse(1,:) = rotated_ellipse(1,:)+X0;
rotated_ellipse(2,:) = rotated_ellipse(2,:)+Y0;

