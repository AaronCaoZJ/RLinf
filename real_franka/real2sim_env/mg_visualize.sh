cd /workspace1/zhijun/RLinf/real_franka/real2sim_env

python mg_render_video.py \
    --hdf5 mg_dataset/blockpap_multi_gpu_gen.hdf5 \
    --video mg_dataset/blockpap_multi_gpu_demos_success.mp4 \
    --video-skip 5 \
    --num-render 20 \
    --seed 10

# python mg_render_video.py \
#     --hdf5 mg_dataset/blockpap_gen_failed.hdf5 \
#     --video mg_dataset/blockpap_demos_failed.mp4 \
#     --video-skip 2 \
#     --num-render 50 \
#     --seed 42

# python mg_check_camera_view.py \
#     --hdf5 mg_dataset/blockpap_multi_gpu_gen.hdf5 \
#     --step 10 \
#     --cols 5 \
#     --cam external_cam