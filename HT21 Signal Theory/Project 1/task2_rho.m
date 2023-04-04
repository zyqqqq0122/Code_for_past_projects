clc;
clear;
close all;
load Gaussian2D.mat;

[X,Y]=meshgrid(-5:0.1:5,-5:0.1:5);
m2=mean(s2,1);
cov2=cov(s2,1);
r2=cov2(1,2)/(cov2(1,1)^0.5*cov2(2,2)^0.5);
pdf=mvnpdf([X(:) Y(:)],m2,cov2);
pdf1=reshape(pdf,size(X));

figure(1);
mesh(X,Y,pdf1);
view(2);
xlabel('x');
ylabel('y');
title('X-Y Plane(\rho = 0.75)');

cov21=[cov2(1,1),-cov2(1,2);-cov2(2,1),cov2(2,2)];
pdf2=mvnpdf([X(:) Y(:)],m2,cov21);
pdf21=reshape(pdf2,size(X));
figure(2);
mesh(X,Y,pdf21);
view(2);
xlabel('x');
ylabel('y');
title('X-Y Plane(\rho = - 0.75)');
