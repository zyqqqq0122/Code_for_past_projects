load("spydata.mat");
load("training.mat");
x=ones(29,1);
y=ones(29,1);
for i=4:32
    x(i-3,1)=received(i,1);
    y(i-3,1)=training(i,1);
end
b=1;
a=[4.1,0.5,2.1,0.1];
x_1=filter(b,a,x);
tmse=0;
for j=1:29
    tmse=tmse+(x_1(j,1)-y(j,1)).^2;
end
mse=tmse/29;
r_0=filter(b,a,received);
r=sign(r_0);
dpic=decoder(r,cPic);
figure(1);
image(dpic);
figure(2);
image(cPic);