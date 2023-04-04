clear all;
close all;
load spydata.mat;
load training.mat;

B1 = training(4:32); 
R1 = zeros(29,4);

for ii = 4:32
    for jj = 1:4
        R1(ii-3,jj) = received(ii-jj+1);
    end
end

coeff = R1\B1;

recoveredReceived = filter(coeff,1,received);
signalDetected = sign(recoveredReceived);
recoveredPic = decoder(signalDetected, cPic);
figure;
image(recoveredPic);
axis square;

tmse=0;
for j=1:29
    tmse=tmse+(recoveredReceived(j,1)-B1(j,1)).^2;
end
mse=tmse/29;