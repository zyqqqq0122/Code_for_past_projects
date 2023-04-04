clc;
clear;
close all;
load spydata.mat;
load training.mat;

%%
%Matrix Square = Order 4
B1 = training(5:32); 
A1 = zeros(28,5);
for ii = 5:32
    for jj = 1:5
        A1(ii-4,jj) = received(ii-jj+1);
    end
end

coeff4 = A1\B1;

recoveredReceived4 = filter(coeff4,1,received);
signalDetected4 = sign_k(recoveredReceived4);
recoveredPic4 = decoder(signalDetected4, cPic);
% figure;image(recoveredPic4);
% axis square

berror4 = numel(find(training~=signalDetected4(1:32)));
biterror4 = biterr(tocero(training), tocero(signalDetected4(1:32)));

%MSE
se4 = abs(training-recoveredReceived4(1:32)).^2;
mse4 = mean(se4(:));

err4 = immse(training, recoveredReceived4(1:32));

%Orthogonality
epsilon4 = (recoveredReceived4(1:32) - training).* received(1:32);
orth4 = mean(epsilon4(:));

%Non reference image quality measures
brisque4 = niqe(recoveredPic4);

%%

%Matrix Square = Order 5
B = training(6:32); 
A = zeros(27,6);
for ii = 6:32
    for jj = 1:6
        A(ii-5,jj) = received(ii-jj+1);
    end
end

coeff5 = A\B;

recoveredReceived5 = filter(coeff5,1,received);
signalDetected5 = sign_k(recoveredReceived5);
recoveredPic5 = decoder(signalDetected5, cPic);

%Number of detection errors on training bits
berror5 = numel(find(training~=signalDetected5(1:32)));
biterror5 = biterr(tocero(training), tocero(signalDetected5(1:32)));

%MSE
se5 = abs(training-recoveredReceived5(1:32)).^2;
mse5 = mean(se5(:));

err5 = immse(training, recoveredReceived5(1:32));

%Orthogonality
epsilon5 = (recoveredReceived5(1:32) - training).* received(1:32);
orth5 = mean(epsilon5(:));

%Non reference image quality measures
brisque5 = niqe(recoveredPic5);

%image(recoveredPic5);
%axis square

%Matrix Square = Order 6
B2 = training(7:32); 
A2 = zeros(26,7);
for ii = 7:32
    for jj = 1:7
        A2(ii-6,jj) = received(ii-jj+1);
    end
end

coeff6 = A2\B2;

recoveredReceived6 = filter(coeff6,1,received);
signalDetected6 = sign_k(recoveredReceived6);
recoveredPic6 = decoder(signalDetected6, cPic);
figure;subplot(1,3,1);image(recoveredPic6);title('order 6');
axis square

berror6 = numel(find(training~=signalDetected6(1:32)));
biterror6 = biterr(tocero(training), tocero(signalDetected6(1:32)));

se6 = abs(training-recoveredReceived6(1:32)).^2;
mse6 = mean(se6(:));

err6 = immse(training, recoveredReceived6(1:32));

%Orthogonality
epsilon6 = (recoveredReceived6(1:32) - training).* received(1:32);
orth6 = mean(epsilon6(:));

%Non reference image quality measures
brisque6 = niqe(recoveredPic6);

%Matrix Square = Order 7
B3 = training(8:32); 
A3 = zeros(25,8);
for ii = 8:32
    for jj = 1:8
        A3(ii-7,jj) = received(ii-jj+1);
    end
end

coeff7 = A3\B3;

recoveredReceived7 = filter(coeff7,1,received);
signalDetected7 = sign_k(recoveredReceived7);
recoveredPic7 = decoder(signalDetected7, cPic);
%figure;
% subplot(1,3,1);
% image(recoveredPic7);
% axis square

berror7 = numel(find(training~=signalDetected7(1:32)));
biterror7 = biterr(tocero(training), tocero(signalDetected7(1:32)));

se7 = abs(training-recoveredReceived7(1:32)).^2;
mse7 = mean(se7(:));

err7 = immse(training, recoveredReceived7(1:32));

%Orthogonality
epsilon7 = (recoveredReceived7(1:32) - training).* received(1:32);
orth7 = mean(epsilon7(:));

%Non reference image quality measures
brisque7 = niqe(recoveredPic7);

%Matrix Square = Order 8
B4 = training(9:32); 
A4 = zeros(24,9);
for ii = 9:32
    for jj = 1:9
        A4(ii-8,jj) = received(ii-jj+1);
    end
end

coeff8 = A4\B4;

recoveredReceived8 = filter(coeff8,1,received);
signalDetected8 = sign_k(recoveredReceived8);
recoveredPic8 = decoder(signalDetected8, cPic);
subplot(1,3,2);
image(recoveredPic8);title('order 8');
axis square

berror8 = numel(find(training~=signalDetected8(1:32)));
biterror8 = biterr(tocero(training), tocero(signalDetected8(1:32)));

se8 = abs(training-recoveredReceived8(1:32)).^2;
mse8 = mean(se8(:));

err8 = immse(training, recoveredReceived8(1:32));

%Orthogonality
epsilon8 = (recoveredReceived8(1:32) - training).* received(1:32);
orth8 = mean(epsilon8(:));

%Non reference image quality measures
brisque8 = niqe(recoveredPic8);

%Matrix Square = Order 9
B5 = training(10:32); 
A5 = zeros(23,10);
for ii = 10:32
    for jj = 1:10
        A5(ii-9,jj) = received(ii-jj+1);
    end
end

coeff9 = A5\B5;

recoveredReceived9 = filter(coeff9,1,received);
signalDetected9 = sign_k(recoveredReceived9);
recoveredPic9 = decoder(signalDetected9, cPic);
%image(recoveredPic9);
%axis square

berror9 = numel(find(training~=signalDetected9(1:32)));
biterror9 = biterr(tocero(training), tocero(signalDetected9(1:32)));

se9 = abs(training-recoveredReceived9(1:32)).^2;
mse9 = mean(se9(:));

err9 = immse(training, recoveredReceived9(1:32));

%Orthogonality
epsilon9 = (recoveredReceived9(1:32) - training).* received(1:32);
orth9 = mean(epsilon9(:));

%Non reference image quality measures
brisque9 = niqe(recoveredPic9);

%Matrix Square = Order 10
B6 = training(11:32); 
A6 = zeros(22,11);
for ii = 11:32
    for jj = 1:11
        A6(ii-10,jj) = received(ii-jj+1);
    end
end

coeff10 = A6\B6;

recoveredReceived10 = filter(coeff10,1,received);
signalDetected10 = sign_k(recoveredReceived10);
recoveredPic10 = decoder(signalDetected10, cPic);
%figure;
subplot(1,3,3);
image(recoveredPic10);title('order 10');
axis square

berror10 = numel(find(training~=signalDetected10(1:32)));
biterror10 = biterr(tocero(training), tocero(signalDetected10(1:32)));

se10 = abs(training-recoveredReceived10(1:32)).^2;
mse10 = mean(se10(:));

err10 = immse(training, recoveredReceived10(1:32));

%Orthogonality
epsilon10 = (recoveredReceived10(1:32) - training).* received(1:32);
orth10 = mean(epsilon10(:));

%Non reference image quality measures
brisque10 = niqe(recoveredPic10);

%Matrix Square = Order 11
B7 = training(12:32); 
A7 = zeros(21,12);
for ii = 12:32
    for jj = 1:12
        A7(ii-11,jj) = received(ii-jj+1);
    end
end

coeff11 = A7\B7;

recoveredReceived11 = filter(coeff11,1,received);
signalDetected11 = sign_k(recoveredReceived11);
recoveredPic11 = decoder(signalDetected11, cPic);
% image(recoveredPic11);
% axis square

biterror11 = biterr(tocero(training), tocero(signalDetected11(1:32)));

se11 = abs(training-recoveredReceived11(1:32)).^2;
mse11 = mean(se11(:));

err11 = immse(training, recoveredReceived11(1:32));

%Orthogonality
epsilon11 = (recoveredReceived11(1:32) - training).* received(1:32);
orth11 = mean(epsilon11(:));

%Non reference image quality measures
brisque11 = niqe(recoveredPic11);

%Matrix Square = Order 12
B8 = training(13:32); 
A8 = zeros(20,13);
for ii = 13:32
    for jj = 1:13
        A8(ii-12,jj) = received(ii-jj+1);
    end
end

coeff12 = A8\B8;

recoveredReceived12 = filter(coeff12,1,received);
signalDetected12 = sign_k(recoveredReceived12);
recoveredPic12 = decoder(signalDetected12, cPic);
%image(recoveredPic12);
%axis square

biterror12 = biterr(tocero(training), tocero(signalDetected12(1:32)));

se12 = abs(training-recoveredReceived12(1:32)).^2;
mse12 = mean(se11(:));

err12 = immse(training, recoveredReceived12(1:32));

%Orthogonality
epsilon12 = (recoveredReceived12(1:32) - training).* received(1:32);
orth12 = mean(epsilon12(:));

%Non reference image quality measures
brisque12 = niqe(recoveredPic12);


%Choosing order 8
%Introducing bit errors

%%ok


one_bit = introducing_errors(100, signalDetected8);
recoveredPic_1 = decoder(one_bit, cPic);
cross_1 = normxcorr2(recoveredPic_1(:,:,1), recoveredPic8(:,:,1)); 
maxcross = max(cross_1,[],'all');
figure;surf(cross_1); shading flat
figure;image(recoveredPic_1);axis square

fiveh_bit = introducing_errors(500, signalDetected8);
recoveredPic_500 = decoder(fiveh_bit, cPic);
cross_500 = normxcorr2(recoveredPic_500(:,:,1), recoveredPic8(:,:,1)); 
maxcross500 = max(cross_500,[],'all');
figure;subplot(1,3,1); surf(cross_500);shading flat
% figure;subplot(1,3,1);image(recoveredPic_500);axis square
title('500 errors');

eighth_bit = introducing_errors(750, signalDetected8);
recoveredPic_800 = decoder(eighth_bit, cPic);
cross_800 = normxcorr2(recoveredPic_800(:,:,1), recoveredPic8(:,:,1)); 
maxcross800 = max(cross_800,[],'all');
% figure;image(recoveredPic_800);axis square

thousand_bit = introducing_errors(970, signalDetected8);
recoveredPic_1000 = decoder(thousand_bit, cPic);
cross_1000 = normxcorr2(recoveredPic_1000(:,:,1), recoveredPic8(:,:,1));
maxcross1000 = max(cross_1000,[],'all');
% figure;image(recoveredPic_1000);axis square

thousand300_bit = introducing_errors(1250, signalDetected8);
recoveredPic_1300 = decoder(thousand300_bit, cPic);
cross_1300 = normxcorr2(recoveredPic_1300(:,:,1), recoveredPic8(:,:,1));
maxcross1300 = max(cross_1300,[],'all');
% figure;image(recoveredPic_1300);axis square

thousand500_bit = introducing_errors(1500, signalDetected8);
recoveredPic_1500 = decoder(thousand500_bit, cPic);
cross_1500 = normxcorr2(recoveredPic_1500(:,:,1), recoveredPic8(:,:,1));
maxcross1500 = max(cross_1500,[],'all');
subplot(1,3,2);surf(cross_1500), shading flat
% figure;
% subplot(1,3,2);
% image(recoveredPic_1500);axis square
title('1500 errors');

thousand800_bit = introducing_errors(2500, signalDetected8);
recoveredPic_1800 = decoder(thousand800_bit, cPic);
cross_1800 = normxcorr2(recoveredPic_1800(:,:,1), recoveredPic8(:,:,1));
maxcross1800 = max(cross_1800,[],'all');
subplot(1,3,3), surf(cross_1800), shading flat
% subplot(1,3,3);
% image(recoveredPic_1800);axis square
title('2500 errors');

% %Crosscorrelation graph
% maxcrosscorrelations = zeros(numel(signalDetected8),1);
% for ii = 1:numel(maxcrosscorrelations)
%     ii_bit = introducing_errors(ii, signalDetected8);
%     recoveredPic_ii = decoder(ii_bit, cPic);
%     cross_ii = normxcorr2(recoveredPic_ii(:,:,1), recoveredPic8(:,:,1));
%     maxcrosscorrelations(ii) = max(cross_ii,[],'all');
% end   
% 
% mincorr = min(maxcrosscorrelations);
% numbit = find(maxcrosscorrelations==mincorr);
% 
% %Minimal crosscorrelation point in vector
% maxbit_bit = introducing_errors(numbit, signalDetected8);
% recoveredPic_maxbit = decoder(maxbit_bit, cPic);
% cross_maxbit = normxcorr2(recoveredPic_maxbit(:,:,1), recoveredPic8(:,:,1));
% maxcross_maxbit = max(cross_maxbit,[],'all');
% figure, surf(cross_maxbit), shading flat
% figure;
% image(recoveredPic_maxbit);
% axis square


