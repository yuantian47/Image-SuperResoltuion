video_name = 'gforeman';
iter = 40;
frame = 10;

mov_red = load("../data/results/" + ...
    video_name + int2str(iter) + "/red.mat");
figure(1);
imshow(mov_red.mov_red(:, :, frame));

mov_ppp = load("../data/results/" + ...
    video_name + int2str(iter) + "/ppp.mat");
figure(2);
imshow(mov_ppp.mov_ppp(:, :, frame));

mov_bic = load("../data/results/" + ...
    video_name + int2str(iter) + "/bic.mat");
figure(3);
imshow(mov_bic.mov_bic(:, :, frame));

mov_raw = load("../data/results/" + ...
    video_name + int2str(iter) + "/raw.mat");
figure(4);
imshow(mov_raw.mov_raw(:, :, frame));

psnr_red = psnr(double(mov_red.mov_red(:, :, frame)), ...
    rescale(mov_raw.mov_raw(:, :, frame)));
psnr_ppp = psnr(double(mov_ppp.mov_ppp(:, :, frame)), ...
    rescale(mov_raw.mov_raw(:, :, frame)));
psnr_bic = psnr(double(mov_bic.mov_bic(:, :, frame)), ...
    rescale(mov_raw.mov_raw(:, :, frame)));