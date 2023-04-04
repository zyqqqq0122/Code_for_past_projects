clear all;
close all;
load spydata.mat;
load training.mat;
%%
for ii=1:31
L=ii;    
% L=9;

%%
for i=1:32
y(i)=received(i,1);
x(i)=training(i,1);
end
rY=xcorr(y);
RY=zeros(L+1,L+1);
for m=1:(L+1)
   for k=1:(L+1)
    RY(m,k)=rY(k-m+32);
   end
end
r=xcorr(x,y);
for j=1:(L+1)
    rXY(j,1)=r(j+31);
end
h=mldivide(RY,rXY);
rcvd=filter(h,1,received);
dttd=sign(rcvd);
mse=0;
othp=0;
for p=1:length(training)
    rr(p)=rcvd(p);
%     rr(p)=rcvd(p);
    mse=mse+(rr(p)-training(p,1)).^2;
    othp=othp+(rr(p)-training(p,1))*(training(p,1));
end

%%
mmse(ii)=mse/length(training);
othpp(ii)=othp/length(training);

% rPic=decoder(dttd,cPic);
% image(rPic);
% axis square;

%%
end

%%
figure;
plot(mmse);
grid on;
title('MSE of Different Filter Orders');
xlabel('L');
ylabel('MSE');
figure;
plot(othpp);
grid on;
title('Orthogonality Test');
xlabel('L');
ylabel('E');
%%
% erdttd=biterror(dttd,786);
% erPic=decoder(erdttd,cPic);
% figure(2);
% image(erPic);