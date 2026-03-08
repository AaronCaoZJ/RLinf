cd /workspace1/zhijun/RLinf

python real_franka/real2sim_env/mg_generate_blockpap_data.py \
    --src /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_src.hdf5 \
    --output /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_gen.hdf5 \
    --num-demos 10 \
    --max-attempts 100 \
    --use-image-obs \
    --video-success /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_demos_success.mp4 \
    --video-skip 5 \
    --num-render 10