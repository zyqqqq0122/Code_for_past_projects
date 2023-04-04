close all;
clear;

delta=0.1;
x=-1:0.01:1;
y=quantizer(x,delta);
scatter(x,y,3,'filled');
grid on;
title('Quantizer function');