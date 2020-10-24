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

% Bicubic interpolation
mov_bic = imresize(mov_lr, dp_factor);

% Super resolution optimization tested
[H, mov_lex] = gen_H(mov_bic, blur_kernel);
S = gen_S(mov_bic, dp_factor);


function mov_raw = preprocess_video(mov_color_struct, num_frame)
    video_size = size(mov_color_struct(1).cdata);
    mov_raw = zeros(video_size(1), video_size(2), num_frame);
    for i = 1:num_frame
        mov_raw(:, :, i) = rgb2gray(mov_color_struct(i).cdata);
    end
end

function mov_lr = gen_lr_video(mov_raw, blur_kernel, dp_factor, noise_std)
    mov_blur = imfilter(mov_raw, blur_kernel, 'same', 'conv');
    mov_dp = mov_blur(1:dp_factor:end, 1:dp_factor:end, :);
    mov_lr = rescale(imnoise(rescale(mov_dp), 'gaussian', 0, ...
        noise_std/255.0), 0, 255);
end

function [H, mov_lex] = gen_H(mov, kernel)
    mov_size = size(mov);
    conv_kernel = rot90(kernel, 2);
    ke_size = size(conv_kernel);
    x = -(ke_size(1) - 1)/2 : (ke_size(1) - 1)/2;
    y = -(ke_size(2) - 1)/2 : (ke_size(2) - 1)/2;
    [ker_x, ker_y]= meshgrid(x, y);
    mov_lex = zeros(mov_size(1) * mov_size(2), mov_size(3));
    H_row = 1;
    row_idx = [];
    col_idx = [];
    val = [];
    for i = 0:mov_size(1)-1
        for j = 0:mov_size(2)-1
            mov_lex((i+1) + j*mov_size(1), :) = mov(i+1, j+1, :);
            for k = 1:ke_size(1)
                for l = 1:ke_size(2)
                   col = ker_x(k, l) + j;
                   if col < 0
                       continue
                   elseif col + 1 > mov_size(2)
                       continue
                   end
                   row = ker_y(k, l) + i;
                   if row < 0
                       continue
                   elseif row + 1 > mov_size(1)
                       continue 
                   end
                   row_idx(end + 1) = H_row;
                   col_idx(end + 1) = (row + 1) + col*mov_size(1);
                   val(end + 1) = conv_kernel(k, l);
                end
            end
            H_row = H_row + 1;
        end
    end
    H = sparse(row_idx, col_idx, val);
end

function S = gen_S(mov, dp_factor)
    mask_sub = zeros(1, dp_factor);
    mask_sub(1) = 1;
    mov_size = size(mov);
    row_repeat = uint8(mov_size(1)/dp_factor);
    col_repeat = uint8(mov_size(2)/dp_factor);
    x = repmat(mask_sub, 1, row_repeat);
    y = repmat(mask_sub, 1, col_repeat);
    [Y, X] = meshgrid(y, x);
    row_idx = [];
    col_idx = [];
    val = [];
    for i = 1:mov_size(1)
        for j = 1:mov_size(2)
            if X(i, j) == 1 && Y(i, j) == 1
                row_idx(end + 1) = i + (j-1)*mov_size(1);
                col_idx(end + 1) = i + (j-1)*mov_size(1);
                val(end + 1) = 1;
            end
        end
    end
    S = sparse(row_idx, col_idx, val, ...
        mov_size(1)*mov_size(2), mov_size(1)*mov_size(2));
end
