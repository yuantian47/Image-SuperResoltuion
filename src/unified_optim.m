close all;
clear;

video_name = "../data/foreman_qcif.y4m";
[mov_color_struct, mov_info] = yuv4mpeg2mov(video_name);
num_frame = 30;
mov_raw = preprocess_video(mov_color_struct, num_frame);


function mov_raw = preprocess_video(mov_color_struct, num_frame)
    video_size = size(mov_color_struct(1).cdata);
    mov_raw = zeros(video_size(1), video_size(2), num_frame);
    for i = 1:num_frame
        mov_raw(:, :, i) = rgb2gray(mov_color_struct(i).cdata);
    end
end
