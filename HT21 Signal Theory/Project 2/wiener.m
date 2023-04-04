clear all;
close all;
load spydata.mat;
load training.mat;

L=3;
R=zeros((L+1),1);
% Z=training((L+1):32);
for i=1:(L+1)
   R(i,1)=received((33-i),1);
end

temp=xcorr(R);
for i=1:(L+1)
   for j=1:(L+1)
      acfR(i,j)=temp(L+i+1-j); 
   end
end

temp1=xcorr(training(L+1),R);
for kk=1:(L+1)
    rxy(kk,1)=temp1(L+2-kk); 
end

% for k=(L+1):32
%     temp1=xcorr(training(k),R);
%     for kk=1:(L+1)
%        rxy(kk,(k-L))=temp1(L+2-kk); 
%     end
% end

h=acfR\rxy;

rcvd=filter(h,1,received);
dttd=sign(rcvd);
rPic=decoder(dttd,cPic);
figure;
image(rPic);
axis square;