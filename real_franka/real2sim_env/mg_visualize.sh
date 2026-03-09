cd /workspace1/zhijun/RLinf

# python real_franka/real2sim_env/mg_render_video.py \
#     --hdf5 /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_gen.hdf5 \
#     --video /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_demos_success.mp4 \
#     --video-skip 5 \
#     --num-render 50

# python real_franka/real2sim_env/mg_render_video.py \
#     --hdf5 /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_gen_failed.hdf5 \
#     --video /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_demos_failed.mp4 \
#     --video-skip 2 \
#     --num-render 50

python real_franka/real2sim_env/mg_check_camera_view.py \
    --hdf5 real_franka/real2sim_env/mg_dataset/blockpap_gen.hdf5 \
    --step 20 \
    --cols 5 \
    --cam external_cam