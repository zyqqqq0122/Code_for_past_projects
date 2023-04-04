clear;
close all;
load coeffs.mat;

psv=db4;
s=rand(1,randi([2 20],1,1));
[cA,cD]=analysis_filter(s,psv);
s_rec=synthesis_filter(psv,cA,cD);