close all;
clear;

video_name_arr = ["gbicycle"];
num_frame = 25;
iter = 40;
for i = 1:size(video_name_arr, 2)
    optimation_analysis(video_name_arr(i), num_frame, iter);
end


function optimation_analysis(video_name, num_frame, iter)
    [mov_red, mov_bic, mov_raw] = video_sr(video_name, num_frame,...
        iter, 'RED');
    [mov_ppp, ~, ~] = video_sr(video_name, num_frame, iter, 'PPP');

    psnr_red = psnr(double(mov_red(:, :, :)), rescale(mov_raw(:, :, :)));
    psnr_ppp = psnr(double(mov_ppp(:, :, :)), rescale(mov_raw(:, :, :)));
    psnr_bic = psnr(double(mov_bic(:, :, :)), rescale(mov_raw(:, :, :)));
    ssim_red = ssim(double(mov_red(:, :, :)), rescale(mov_raw(:, :, :)));
    ssim_ppp = ssim(double(mov_ppp(:, :, :)), rescale(mov_raw(:, :, :)));
    ssim_bic = ssim(double(mov_bic(:, :, :)), rescale(mov_raw(:, :, :)));

    if ~exist("../data/results/" + video_name + int2str(iter) + "/", 'dir')
       mkdir("../data/results/" + video_name + int2str(iter) + "/")
    end

    save("../data/results/" + video_name + int2str(iter) + "/red.mat",...
        'mov_red');
    save("../data/results/" + video_name + int2str(iter) + "/ppp.mat",...
        'mov_ppp');
    save("../data/results/" + video_name + int2str(iter) + "/bic.mat",...
        'mov_bic');
    save("../data/results/" + video_name + int2str(iter) + "/raw.mat",...
        'mov_raw');

    frame_idx = 1:num_frame;
    psnr_bic_arr = zeros(1, num_frame);
    psnr_red_arr = zeros(1, num_frame);
    psnr_ppp_arr = zeros(1, num_frame);
    ssim_bic_arr = zeros(1, num_frame);
    ssim_red_arr = zeros(1, num_frame);
    ssim_ppp_arr = zeros(1, num_frame);
    for i = 1:num_frame
        psnr_red_arr(i) = psnr(double(mov_red(:, :, i)),...
            rescale(mov_raw(:, :, i)));
        psnr_ppp_arr(i) = psnr(double(mov_ppp(:, :, i)),...
            rescale(mov_raw(:, :, i)));
        psnr_bic_arr(i) = psnr(double(mov_bic(:, :, i)),...
            rescale(mov_raw(:, :, i)));
        ssim_red_arr(i) = ssim(double(mov_red(:, :, i)),...
            rescale(mov_raw(:, :, i)));
        ssim_ppp_arr(i) = ssim(double(mov_ppp(:, :, i)),...
            rescale(mov_raw(:, :, i)));
        ssim_bic_arr(i) = ssim(double(mov_bic(:, :, i)),...
            rescale(mov_raw(:, :, i)));
    end

    fig_psnr = figure(1);
    plot(frame_idx, psnr_red_arr, 'r-o', ...
        frame_idx, psnr_ppp_arr, 'g-*', ...
        frame_idx, psnr_bic_arr, 'b-^');
    xlabel("Frame index")
    ylabel("PSNR")
    legend('RED', 'PPP', 'Bicubic');
    title("Bicycle PSNR per frame");
    saveas(fig_psnr, "../data/results/" + ...
        video_name + int2str(iter) + "/psnr_plot.png")
    savefig(fig_psnr, "../data/results/" + ...
        video_name + int2str(iter) + "/psnr_plot.fig")

    fig_ssim = figure(2);
    plot(frame_idx, ssim_red_arr, 'r-o', ...
        frame_idx, ssim_ppp_arr, 'g-*', ...
        frame_idx, ssim_bic_arr, 'b-^');
    xlabel("Frame index")
    ylabel("SSIM")
    legend('RED', 'PPP', 'Bicubic');
    title("Bicycle SSIM per frame");
    saveas(fig_ssim, "../data/results/" + ...
        video_name + int2str(iter) + "/ssim_plot.png")
    savefig(fig_ssim, "../data/results/" + ...
        video_name + int2str(iter) + "/ssim_plot.fig")

    performance = [psnr_red psnr_ppp psnr_bic; ssim_red ssim_ppp ssim_bic];
    writematrix(performance, ...
        "../data/results/" + video_name + int2str(iter) + "/performance.csv")

    new_video_name = "../data/results/" + ...
        video_name + int2str(iter) + "/joint_mov" + ".avi";
    new_video = VideoWriter(new_video_name, 'Uncompressed AVI');
    open(new_video);
    for i = 1:num_frame
    writeVideo(new_video, [rescale(mov_raw(:, :, i)),...
        rescale(mov_bic(:, :, i)); rescale(double(mov_red(:, :, i))),...
        rescale(double(mov_ppp(:, :, i)))])
    end
    close(new_video);
%     close(fig_psnr);
%     close(fig_ssim);
end


function [v_raw, mov_bic, mov_raw] = video_sr(v_name, num_frame, iter, mth)
    % Get raw image
    video_name = "../data/noise_free/" + v_name + ".avi";
    mov_raw = preprocess_video(video_name, num_frame);

    % Blur, downsample, and add noise
    blur_kernel = fspecial('gaussian', [7 7], 1.5);
    dp_factor = 4;
    gaussian_noise_std = 1.0;
    mov_lr = gen_lr_video(mov_raw, blur_kernel, dp_factor, gaussian_noise_std);

    % Bicubic interpolation
    mov_bic = imresize(mov_lr, dp_factor, 'bicubic');

    % Super resolution optimization tested
    rou = 0.0001;
    beta = 0.004;
    alpha = 1.2;
    [H, mov_lex] = gen_H(mov_bic, blur_kernel);
    x = mov_lex;
    v = mov_lex;
    u = zeros(size(v));
    y = lex_transform(mov_lr);
    dual_gap = 0;
    S = gen_S(mov_bic, dp_factor);
    L = (1/(gaussian_noise_std^2)) * transpose(S*H) * (S*H);
    for i = 1:iter
        x_size = size(x);
        x_new = zeros(x_size);
        for j = 1:x_size(2)
            % A*x = b
            A = L + rou*speye(size(L));
            b = (1/(gaussian_noise_std^2)) * transpose(S*H) * y(:, j) + ...
                rou * (v(:, j) - u(:, j));
            x_new(:, j) = pcg(A, b, 1e-6, 30);
        end
        x = x_new;
        x_raw = inverse_lex(x, size(mov_bic));
        u_raw = inverse_lex(u, size(mov_bic));
        v_old = v;
        
        if strcmp(mth, 'PPP')
            % PPP
            [~, v_raw] = VBM3D(x_raw + u_raw, sqrt(beta/rou));
        else
            % RED
            z = inverse_lex(v, size(mov_bic));
            for k = 1:2
                [~, denoise] = VBM3D(z, sqrt(beta/rou));
                z = (1/(beta + rou)) * (beta*denoise + rou * (x_raw + u_raw));
            end
            v_raw = z;
        end

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
    
end

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
    mov_lr = imnoise(rescale(mov_dp), 'gaussian', 0, (noise_std/255)^2);
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


