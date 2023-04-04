clear all;
close all;
load spydata.mat;
load training.mat;
L=11;
n=(L+1):32;
for i=1:(32-L)
y(i)=received(i+L,1);
x(i)=training(i+L,1);
end
rY=xcorr(y);
RY=zeros(L+1,L+1);
for m=1:(L+1)
   for k=1:(L+1)
    RY(m,k)=rY(k-m+29);
   end
end
r=xcorr(x,y);
for j=1:(L+1)
    rXY(j,1)=r(j+28);
end
h=mldivide(RY,rXY);
rcvd=filter(h,1,received);
dttd=sign(rcvd);
rPic=decoder(dttd,cPic);
figure;
image(rPic);
axis square;