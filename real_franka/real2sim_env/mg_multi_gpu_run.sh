cd /workspace1/zhijun/RLinf

python real_franka/real2sim_env/mg_generate_blockpap_data.py \
    --src /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_cleaned_src.hdf5 \
    --lerobot-dir /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_cleaned_mimicgen \
    --num-workers 5 \
    --gpu-ids 1,2,3,4,5 \
    --num-demos 1000 \
    --max-attempts 10000 \
    --fps 20