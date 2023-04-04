close all;
clc;

step = [2^3 2^4 2^5 2^6];
fps = 30;
width = 176;
height = 144;
n_frames = 50;

% Video = yuv_import_y('foreman_qcif.yuv',[width height],n_frames);
Video = yuv_import_y('mother-daughter_qcif.yuv',[width height],n_frames);
frames = zeros(height,width,n_frames);
for i=1:n_frames
    frames(:,:,i) = Video{i,1};
end

max_shift = 10;
num_shift = (2*max_shift + 1)^2;
shift = zeros(2,num_shift);
index = 1;
for dx = -max_shift:1:max_shift
    for dy = -max_shift:1:max_shift
        shift(1,index) = dx;
        shift(2,index) = dy;
        index = index + 1;
    end
end

b = 16;
n_blocks = width*height/(b^2);

lbd1 = 0.001*(step.^2);
lbd2 = 0.002*(step.^2);

lbd11 = 0.001*(step.^2);
lbd22 = 0.002*(step.^2);
lbd33 = 0.0004*(step.^2);

video_rate1 = zeros(n_frames,length(step));
RecoVideo1 = zeros(height,width,n_frames,length(step));
PSNR1 = zeros(n_frames,length(step));
% intra & copy
RecoVideo2 = zeros(height,width,n_frames,length(step));
video_rate2 = zeros(n_frames,length(step));
PSNR2 = zeros(n_frames,length(step));
% intra, copy % motion compensation
RecoVideo3 = zeros(height,width,n_frames,length(step));
video_rate3 = zeros(n_frames,length(step));
PSNR3 = zeros(n_frames,length(step));

for i=1:length(step)
        u = 1:b;
        v = 1:b;
        block_num = 1;

        for j = 1:height/b
            for k = 1:width/b
                [RecoVideo2(b*(j-1)+v,b*(k-1)+u,1,i), rate] = ...
                    blk_coder(frames(b*(j-1)+v,b*(k-1)+u,1),step(i),b);
                video_rate2(1,i) = video_rate2(1,i) + rate;
            end
        end
        RecoVideo1(:,:,1,i) = RecoVideo2(:,:,1,i);
        video_rate1(1,i) = video_rate2(1,i);
        RecoVideo3(:,:,1,i) = RecoVideo2(:,:,1,i);
        video_rate3(1,i) = video_rate2(1,i);
        PSNR2(1,i) = PSNR(distortion(RecoVideo2(:,:,1,i),frames(:,:,1)));
        PSNR1(1,i) = PSNR2(1,i);
        PSNR3(1,i) = PSNR2(1,i);
end

num_replBocks = zeros(1,n_frames,length(step));
num_replBocks(:,1,:) = 0;

num_blocks_in = zeros(length(step),3);
num_blocks_in(:,1) = (height/b)*(width/b);

for f=2:n_frames
    
    for i=1:length(step)
        u = 1:b;
        v = 1:b;
        block_num = 1;
        shift_direction = motion(RecoVideo3(:,:,f-1,i),...
            frames(:,:,f),shift,b);

        for j = 1:height/b
            for k = 1:width/b
                R1=0;
                R2=0;
                R3=0;
               
                [b_coded, R1] = blk_coder(frames(b*(j-1)+v,b*...
                    (k-1)+u,f),step(i),b);
                
                Dist1 = distortion(b_coded, frames(b*(j-1)+v,b*...
                    (k-1)+u,f));
                Dist2 = distortion(RecoVideo2(b*(j-1)+v,b*(k-1)+...
                    u,f-1,i),frames(b*(j-1)+v,b*(k-1)+u,f));
                
                R2 = 1;         % copy
                R1 = R1 + 1;	% mode selection

                Cost1 = Dist1 + lbd1(i)*(R1); 
                Cost2 = Dist2 + lbd2(i)* R2;
                Costf = [Cost1 Cost2];
                [MinCost,ChosenMode] = min(Costf);
                
                if ChosenMode == 1
                    RecoVideo2(b*(j-1)+v,b*(k-1)+u,f,i) = b_coded;
                    video_rate2(f,i) = video_rate2(f,i) + R1;
                elseif ChosenMode == 2
                    RecoVideo2(b*(j-1)+v,b*(k-1)+u,f,i) = ...
                        RecoVideo2(b*(j-1)+v,b*(k-1)+u,f-1,i);

                    num_replBocks(1,f,i) = num_replBocks(1,f,i) + 1;
                    video_rate2(f,i) = video_rate2(f,i) + R2;
                end  
                
                % intra
                RecoVideo1(b*(j-1)+v,b*(k-1)+u,f,i) = b_coded;
                video_rate1(f,i) = video_rate1(f,i) + R1 - 1;
                
                % intra, copy & motion compensation
                dy = shift(1,shift_direction(block_num));
                dx = shift(2,shift_direction(block_num));
                y1 = b*(j-1) + dy + v;
                x1 = b*(k-1) + dx + u;
                blk_mvd = RecoVideo3(y1,x1,f-1,i);
                [blk_rec, R3] = residual_coder(blk_mvd,frames...
                    (y1,x1,f),step(i));
                
                Dist3 = distortion(blk_rec,frames(b*(j-1)+v,b*...
                    (k-1)+u,f));
                
                R2 = 2;         % copy
                R1 = R1 + 1;    % mode selection
                R3 = 2 + 10 + R3;

                Cost1 = Dist1 + lbd11(i)*R1; 
                Cost2 = Dist2 + lbd22(i)*R2;
                Cost3 = Dist3 + lbd33(i)*R3;
                Costf = [Cost1 Cost2 Cost3];
                [MinCost,ChosenMode] = min(Costf);
                
                if ChosenMode == 1
                    RecoVideo3(b*(j-1)+v,b*(k-1)+u,f,i) = b_coded;
                    video_rate3(f,i) = video_rate3(f,i) + R1;
                    num_blocks_in(i,1) = num_blocks_in(i,1) + 1;
                elseif ChosenMode == 2
                    RecoVideo3(b*(j-1)+v,b*(k-1)+u,f,i) = ...
                        RecoVideo3(b*(j-1)+v,b*(k-1)+u,f-1,i);

                    num_blocks_in(i,2) = num_blocks_in(i,2) + 1;
                    video_rate3(f,i) = video_rate3(f,i) + R2;
                 elseif ChosenMode == 3
                     RecoVideo3(b*(j-1)+v,b*(k-1)+u,f,i) = blk_rec;
                     video_rate3(f,i) = video_rate3(f,i) + R3;
                     num_blocks_in(i,3) = num_blocks_in(i,3) + 1;
                end  
                
                block_num = block_num + 1;
            end
        end
        PSNR3(f,i) = PSNR(distortion(RecoVideo3(:,:,f,i),frames(:,:,f)));
        PSNR2(f,i) = PSNR(distortion(RecoVideo2(:,:,f,i),frames(:,:,f)));
        PSNR1(f,i) = PSNR(distortion(RecoVideo1(:,:,f,i),frames(:,:,f)));
    end
end

Rate1 = mean(video_rate1,1);
Rate1 = (Rate1*fps)/1000;
PSNR1_video = mean(PSNR1,1);
Rate2 = mean(video_rate2,1);
Rate2 = (Rate2*fps)/1000;
PSNR2_video = mean(PSNR2,1);
Rate3 = mean(video_rate3,1);
Rate3 = (Rate3*fps)/1000;
PSNR3_video = mean(PSNR3,1);

blk_repnum = zeros(1,length(step));
blk_total = zeros(1,length(step));
for q = 1:length(step)
    for f = 1:n_frames
        blk_repnum(q) = blk_repnum(q) + num_replBocks(1,f,q);
    end
    blk_total(q) = n_blocks*n_frames;
end
plot_mat = [blk_total(:)-blk_repnum(:),blk_repnum(:)];

figure;
plot(Rate1,PSNR1_video,'-*','linewidth',0.75);
hold on;
plot(Rate2,PSNR2_video,'-*','linewidth',0.75);
plot(Rate3,PSNR3_video,'-*','linewidth',0.75);
hold off;
grid on;
title('Rate-PSNR Curve');
% legend('intra mode','intra & copy mode', 'intra, copy & inter mode',...
%     'Location','SouthEast');
legend('Intra-Frame','with Conditional Replenishment',...
    'with Motion Compensation','Location','SouthEast');
xlabel('Bit-Rate(kbit/s)');
ylabel('PSNR(dB)');

figure;
bar(step,plot_mat);
legend('intra mode','copy mode');
title('Number of Coded Blocks for Different Modes');
xlabel('Quzantization step');
ylabel('Number of Blocks');

figure;
bar(step,num_blocks_in);
legend('intra mode','copy mode','inter mode');
title('Number of Coded Blocks for Different Modes');
xlabel('Quzantization step');
ylabel('Number of Blocks');