close all;
clear;

% Get raw image
video_name = "../data/foreman_qcif.y4m";
[mov_color_struct, mov_info] = yuv4mpeg2mov(video_name);
num_frame = 30;
mov_raw = preprocess_video(mov_color_struct, num_frame);

% Blur, downsample, and add noise
blur_kernel = fspecial('gaussian', [7 7], 1.5);
dp_factor = 4;
gaussian_noise_std = 1;
mov_lr = gen_lr_video(mov_raw, blur_kernel, dp_factor, gaussian_noise_std);




function mov_raw = preprocess_video(mov_color_struct, num_frame)
    video_size = size(mov_color_struct(1).cdata);
    mov_raw = zeros(video_size(1), video_size(2), num_frame);
    for i = 1:num_frame
        mov_raw(:, :, i) = rgb2gray(mov_color_struct(i).cdata);
    end
end

function mov_lr = gen_lr_video(mov_raw, blur_kernel, dp_factor, noise_std)
    mov_blur = imfilter(mov_raw, blur_kernel, 'symmetric', 'same', 'conv');
    mov_dp = mov_blur(1:dp_factor:end, 1:dp_factor:end, :);
    mov_lr = rescale(imnoise(rescale(mov_dp), 'gaussian', 0, ...
        noise_std/255.0), 0, 255);
end
