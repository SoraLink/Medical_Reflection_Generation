from huggingface_hub import snapshot_download

# 下载混合集
mix_dir = snapshot_download(
    repo_id="histai/HISTAI-mixed",
    repo_type="dataset",
    local_dir="./HISTAI/HISTAI-mixed",
)

# 下载元数据（包含 diagnosis / conclusion / icd10 等，并给出 case 到图像的映射）
meta_dir = snapshot_download(
    repo_id="histai/HISTAI-metadata",
    repo_type="dataset",
    local_dir="./HISTAI/HISTAI-metadata",
)