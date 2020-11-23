close all;
clear;

video_name = 'gforeman';
iter = 40;
frame = 10;

mov_04 = load("../data/results/" + ...
    video_name + int2str(iter) + "_0.4/red.mat");
figure(1);
imshow(mov_04.mov_red(70:145, 100:235, frame));

mov_0004 = load("../data/results/" + ...
    video_name + int2str(iter) + "_0.004/red.mat");
figure(2);
imshow(mov_0004.mov_red(70:145, 100:235, frame));

mov_00004 = load("../data/results/" + ...
    video_name + int2str(iter) + "_0.0004/red.mat");
figure(3);
imshow(mov_00004.mov_red(70:145, 100:235, frame));

mov_raw = load("../data/results/" + ...
    video_name + int2str(iter) + "_0.0004/raw.mat");
figure(4);
imshow(mov_raw.mov_raw(70:145, 100:235, frame));


psnr_red_04 = psnr(double(mov_04.mov_red(70:145, 100:235, frame)), ...
    rescale(mov_raw.mov_raw(70:145, 100:235, frame)));
psnr_red_0004 = psnr(double(mov_0004.mov_red(70:145, 100:235, frame)), ...
    rescale(mov_raw.mov_raw(70:145, 100:235, frame)));
psnr_red_00004 = psnr(double(mov_00004.mov_red(70:145, 100:235, frame)), ...
    rescale(mov_raw.mov_raw(70:145, 100:235, frame)));
