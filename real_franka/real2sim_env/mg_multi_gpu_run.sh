cd /workspace1/zhijun/RLinf

python real_franka/real2sim_env/mg_generate_blockpap_data.py \
    --src /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_src.hdf5 \
    --output /workspace1/zhijun/RLinf/real_franka/real2sim_env/mg_dataset/blockpap_multi_gpu_gen.hdf5 \
    --num-workers 6 \
    --gpu-ids 0,1,2,3,4,5 \
    --num-demos 300 \
    --max-attempts 10000 \
    --use-image-obs \
    --cam-name external_cam \
    --render-seed 42