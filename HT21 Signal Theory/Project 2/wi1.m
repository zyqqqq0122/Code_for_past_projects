clear all;
close all;

load spydata.mat;
load training.mat;
L=9;

for i=1:(32-L)
y(i)=received(i+L,1);
x(i)=training(i+L,1);
end

rY=xcorr(y);
RY=zeros(L+1,L+1);

for m=1:(L+1)
   for k=1:(L+1)
    RY(m,k)=rY(k-m+32-L);
   end
end

r=xcorr(x,y);

for j=1:(L+1)
    rXY(j,1)=r(j+31-L);
end

h=mldivide(RY,rXY);
rcvd=filter(h,1,received);
dttd=sign(rcvd);
rPic=decoder(dttd,cPic);
image(rPic);
axis square;