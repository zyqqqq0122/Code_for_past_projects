close all;
clear all;
load Gaussian1D.mat;

x=-6:0.001:6;
m0=0.5;
var0=2;
norm=normcdf(x,m0,sqrt(var0));
norm0 = normpdf(x,0.5,2^0.5);

[f1,x1]=ecdf(s1);
[f2,x2]=ecdf(s2);
[f3,x3]=ecdf(s3);

m1=mean(s1);
var1=1./length(s1).*sum((s1-m1).^2);

m2=mean(s2);
var2=1./length(s2).*sum((s2-m2).^2);

m3=mean(s3);
var3=1./length(s3).*sum((s3-m3).^2);

mean=[m1 m2 m3];
var=[var1 var2 var3];

figure;
grid on;
hold on;
plot(x1,f1,x2,f2,x3,f3,x,norm);
title('The Empirical Distribution Functions of Gaussian1D');
xlabel('x');
ylabel('Probability');
legend('s1','s2','s3','Theoretical','Location','Northwest');