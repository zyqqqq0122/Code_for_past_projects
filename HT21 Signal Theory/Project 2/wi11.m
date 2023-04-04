clear all;
close all;
load spydata.mat;
load training.mat;
%% 
L=8,
% for L=1:31
%%

for i=1:32
y(i)=received(i,1);
x(i)=training(i,1);
end
rY=xcorr(y,'biased');
RY=zeros(L+1,L+1);
for m=1:(L+1)
   for k=1:(L+1)
    RY(m,k)=rY(k-m+32);
   end
end
r=xcorr(x,y,'biased');
for j=1:(L+1)
    rXY(j,1)=r(j+31);
end
h=mldivide(RY,rXY);
rcvd=filter(h,1,received);
mse=0;
othp=0;
for p=L+1:length(training)
    rr(p)=rcvd(p);
    mse=mse+(rr(p)-training(p,1)).^2;
%     othp=othp+(rr(p)-training(p,1))*(training(p,1));
end

%%
% mmse(L)=mse/(32-L);
% othpp=othp/32;
dttd=sign(rcvd);
rPic=decoder(dttd,cPic);
image(rPic);
title('Order=8');
axis square;
% end
% stem(mmse);
% title('MMSE of Different Orders');
% xlabel('L');
% ylabel('MMSE');

% %%
% figure(2);
% erdttd2=biterror(dttd,2800);
% erPic2=decoder(erdttd2,cPic);
% image(erPic2);
% title('2800 Bit Errors');
% axis square;
