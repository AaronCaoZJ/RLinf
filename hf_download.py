from huggingface_hub import snapshot_download

# download pi0 model
save_dir = "/workspace1/zhijun/hf_download/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
repo_id = "RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT"
repo_type="model"

# download pi0.5 model
save_dir = "/workspace1/zhijun/hf_download/models/RLinf-Pi05-LIBERO-SFT"
repo_id = "RLinf/RLinf-Pi05-LIBERO-SFT"
repo_type="model"

# download libero dataset
# save_dir = "/workspace1/zhijun/hf_download/datasets/physical-intelligence/libero"
# repo_id = "physical-intelligence/libero"
# repo_type="dataset"

cache_dir = save_dir + "/cache"

snapshot_download(
  cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  repo_type=repo_type,
  local_dir_use_symlinks=False,
  resume_download=True,
)