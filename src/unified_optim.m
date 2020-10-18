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

% Super resolution optimization
[H, mov_lex] = gen_H(mov_bic, blur_kernel);


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

function [H, mov_lex] = gen_H(mov, kernel)
    mov_size = size(mov);
    conv_kernel = rot90(kernel, 2);
    ke_size = size(conv_kernel);
    x = -(ke_size(1) - 1)/2 : (ke_size(1) - 1)/2;
    y = -(ke_size(2) - 1)/2 : (ke_size(2) - 1)/2;
    [ker_x, ker_y]= meshgrid(x, y);
%     H = sparse(mov_size(1)*mov_size(2), mov_size(1)*mov_size(2));
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
                       col = -col;
                   elseif col + 1 > mov_size(2)
                       continue
                       col = 2 * mov_size(2) - col;
                   end
                   row = ker_y(k, l) + i;
                   if row < 0
                       continue
                       row = -row;
                   elseif row + 1 > mov_size(1)
                       continue
                       row = 2 * mov_size(1) - row; 
                   end
                   row_idx(end + 1) = H_row;
                   col_idx(end + 1) = (row + 1) + col*mov_size(1);
                   val(end + 1) = conv_kernel(k, l);
%                    disp([(col + 1) + row*mov_size(2), col, row]);
%                    H(H_row, (row + 1) + col*mov_size(1)) = ...
%                        H(H_row, (row + 1) + col*mov_size(1)) + ...
%                        conv_kernel(k, l);
                end
            end
            H_row = H_row + 1;
        end
    end
    H = sparse(row_idx, col_idx, val);
end
