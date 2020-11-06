close all;
clear;

% Get raw image
video_name = "../data/noise_free/gforeman.avi";
num_frame = 25;
mov_raw = preprocess_video(video_name, num_frame);

% Blur, downsample, and add noise
blur_kernel = fspecial('gaussian', [7 7], 1.5);
dp_factor = 4;
gaussian_noise_std = 1;
mov_lr = gen_lr_video(mov_raw, blur_kernel, dp_factor, gaussian_noise_std);

% Bicubic interpolation
mov_bic = imresize(mov_lr, dp_factor);

% Super resolution optimization tested
rou = 0.0001;
beta = 0.2048;
alpha = 1.2;
[H, mov_lex] = gen_H(mov_bic, blur_kernel);
x = mov_lex;
v = mov_lex;
u = zeros(size(v));
y = lex_transform(mov_lr);
dual_gap = 0;
S = gen_S(mov_bic, dp_factor);
L = (1/(gaussian_noise_std^2)) * transpose(S*H) * (S*H);
iter = 40;
for i = 1:iter
    x_size = size(x);
    x_new = zeros(x_size);
    for j = 1:x_size(2)
        % A*x = b
        A = L + rou*speye(size(L));
        b = (1/(gaussian_noise_std^2)) * transpose(S*H) * y(:, j) + ...
            rou * (v(:, j) - u(:, j));
        x_new(:, j) = pcg(A, b, 1e-6, 40);
    end
    x = x_new;
    x_raw = inverse_lex(x, size(mov_bic));
    u_raw = inverse_lex(u, size(mov_bic));
    v_old = v;
    
    % PPP
    % [~, v_raw] = VBM3D(x_raw + u_raw, sqrt(beta/rou));
    
    % RED
    z = inverse_lex(v, size(mov_bic));
    for k = 1:2
        [~, denoise] = VBM3D(z, sqrt(beta/rou));
        z = (1/(beta + rou)) * (beta*denoise + rou * (x_raw + u_raw));
    end
    v_raw = z;
    
    v = double(lex_transform(v_raw));
    dual_gap_new = norm(v - v_old)^2;
    if dual_gap_new <= dual_gap
        rou_new = alpha * rou
    else
        rou_new = rou;
    end
    dual_gap = dual_gap_new;
    u = (rou/rou_new)*(u + x - v);
    rou = rou_new;
    i
end

new_video_name = '../data/vsr.avi';
new_video = VideoWriter(new_video_name, 'Uncompressed AVI');
open(new_video);
for i = 1:x_size(2)
    writeVideo(new_video, rescale(v_raw(:, :, i)))
end
close(new_video);


function mov_raw = preprocess_video(video_name, num_frame)
    video_class = VideoReader(video_name);
    video = read(video_class);
    mov_raw = zeros([video_class.Height, video_class.Width, ...
        num_frame], 'uint8');
    for cf = 1:num_frame
        mov_raw(:,:,cf) = video(:, :, 1, cf);
    end
end

function mov_lr = gen_lr_video(mov_raw, blur_kernel, dp_factor, noise_std)
    mov_blur = imfilter(mov_raw, blur_kernel, 'same', 'conv');
    mov_dp = mov_blur(1:dp_factor:end, 1:dp_factor:end, :);
    mov_lr = imnoise(rescale(mov_dp), 'gaussian', 0, noise_std/255.0);
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
    for j = 0:mov_size(2)-1
        for i = 0:mov_size(1)-1
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
    H = sparse(row_idx, col_idx, val, ...
        mov_size(1)*mov_size(2), mov_size(1)*mov_size(2));
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
    for j = 1:mov_size(2)
        for i = 1:mov_size(1)
            if X(i, j) == 1 && Y(i, j) == 1
                val_len = size(val);
                row_idx(end + 1) = val_len(2) + 1;
                col_idx(end + 1) = i + (j-1)*mov_size(1);
                val(end + 1) = 1;
            end
        end
    end
    S = sparse(row_idx, col_idx, val, ...
        mov_size(1)*mov_size(2)/dp_factor^2, mov_size(1)*mov_size(2));
end

function mov_lex = lex_transform(mov)
    mov_size = size(mov);
    mov_lex = zeros(mov_size(1) * mov_size(2), mov_size(3));
     for i = 0:mov_size(1)-1
        for j = 0:mov_size(2)-1
            mov_lex((i+1) + j*mov_size(1), :) = mov(i+1, j+1, :);
        end
     end
end

function non_lex = inverse_lex(lex, im_size)
    lex_size = size(lex);
    non_lex = zeros([im_size(1), im_size(2), lex_size(2)], 'double');
    for i = 1:lex_size(1)
        row = rem(i, im_size(1));
        col = fix(i/im_size(1))+1;
        if row == 0
            row = 3;
            col = col - 1;
        end
        non_lex(row, col, :) = lex(i, :);
    end
end


