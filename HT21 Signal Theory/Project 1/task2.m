close all;
clear all;
load Gaussian2D.mat;

m1=(s1);
cov1=cov(s1);
rho1=cov1(1,2)./(sqrt(cov1(1,1)).*sqrt(cov1(2,2)));
figure(1);
ksdensity(s1);
xlabel('x');
ylabel('y');
zlabel('Probability');
title('The Empirical Pdf of s1 (\rho=0.25)');

m2=(s2);
cov2=cov(s2);
rho2=cov2(1,2)./(sqrt(cov2(1,1)).*sqrt(cov2(2,2)));
figure(2);
ksdensity(s2);
xlabel('x');
ylabel('y');
zlabel('Probability');
title('The Empirical Pdf of s2 (\rho=0.75)');
