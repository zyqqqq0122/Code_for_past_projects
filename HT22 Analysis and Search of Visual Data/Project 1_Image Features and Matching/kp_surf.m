clear;
close all;

img = imread('obj1_5.jpg');
img = rgb2gray(img);

kps_ori = detectSURFFeatures(img);
kps_ori = kps_ori.selectStrongest(311);
kp_ori=kps_ori.Location;
kp_ori=double(transpose(kp_ori));

kp_surf = zeros(10, 2, 311);
kp_surf(10,:,:)=kp_ori;

for n = 0 : 8

    scale_factor = 1.2 .^n;
    img_scal = imresize(img, scale_factor);
    kps = detectSURFFeatures(img_scal);
    kps = kps.selectStrongest(311);
    kp_scal=kps.Location;
    kp_scal=double(transpose(kp_scal));
    kp_surf(n+1,:,:) = kp_scal;

end
