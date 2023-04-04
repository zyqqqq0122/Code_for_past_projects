clc;
clear;
close all;
load Gaussian1D.mat;

m1 = mean(s1);
var1 = var(s1);
x1 = sort(s1);
norm1=normpdf(x1,m1,var1^0.5);

m2 = mean(s2);
var2 = var(s2);
x2 = sort(s2);
norm2 = normpdf(x2,m2,var2^0.5);

m3 = mean(s3);
var3 = var(s3);
x3 = sort(s3);
norm3 = normpdf(x3,m3,var3^0.5);

x0 =-6:0.01:6;
norm0 = normpdf(x0,0.5,2^0.5);

figure;
plot(x0,norm0,x1,norm1,x2,norm2,x3,norm3);
title('The Probability Density Function of Gaussian1D');
xlabel('x');ylabel('Probability');
legend('Theoretical','s1','s2','s3');
grid on;