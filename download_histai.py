import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
from tqdm import tqdm
import openslide


def download_wsi_from_case_mapping(case_prefix: str, saved_path: str, files, repo_id='histai/HISTAI-mixed') -> Path:

    # 2) 过滤出该 case 目录下的 WSI

    cand = [f for f in files if f.startswith(case_prefix + "/") and f.lower().endswith('.tiff')]
    if not cand:
        raise FileNotFoundError(f"No WSI under {repo_id}/{case_prefix}")

    # 如需更精细选择，可根据命名/后缀/体积排序挑主文件
    cand.sort()
    # 3) 全部下载（或只下一个）
    saved_path = Path(saved_path)
    saved_path = Path("tiff") / saved_path
    saved_path.mkdir(parents=True, exist_ok=True)
    lp = Path(hf_hub_download(
        repo_id=repo_id,
        filename=cand[0],
        local_dir=saved_path,
        repo_type="dataset"
    ))
    return lp


def tiff_to_512(tiff_path: str | Path, mode: str = "fit"):
    """
    把一个 .tiff 病理切片缩小到 512x512 并保存
    mode = "fit"     等比缩放，居中填充白边到正方形
    mode = "stretch" 直接拉伸到512x512
    """
    tiff_path = Path(tiff_path)

    # 用 OpenSlide 打开（更适合 WSI）
    slide = openslide.OpenSlide(str(tiff_path))
    w, h = slide.level_dimensions[0]
    scale = 512 / max(w, h)
    target = (max(1, int(w * scale)), max(1, int(h * scale)))
    thumb = slide.get_thumbnail(target).convert("RGB")
    slide.close()

    if mode == "stretch":
        img512 = thumb.resize((512, 512), Image.BICUBIC)
    else:  # fit
        img512 = ImageOps.pad(
            thumb, (512, 512), method=Image.BICUBIC,
            color=(255, 255, 255), centering=(0.5, 0.5)
        )

    # 删除原始 tiff
    os.remove(tiff_path)
    return img512


def save_sample(img512: Image.Image, text: str, out_root: str | Path,
                case_prefix: str, fmt: str = "jpg", quality: int = 95) -> str:
    """
    把 512x512 图像 + 文本 存到本地：
      - 图片：out_root/images/{case_prefix}_512.jpg
      - 标注：out_root/metadata.jsonl 追加一行
    返回：相对路径（images/...）
    """
    out_root = Path(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.jsonl"

    img_rel = f"images/{case_prefix}_512.{fmt}"
    img_abs = out_root / img_rel
    if fmt.lower() in ("jpg", "jpeg"):
        img512.save(img_abs, "JPEG", quality=quality, optimize=True)
    elif fmt.lower() == "png":
        img512.save(img_abs, "PNG", optimize=True)
    else:
        img512.save(img_abs)

    rec = {"image": img_rel, "text": str(text), "case_prefix": case_prefix}
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return img_rel

def fix_case_prefix(case_prefix: str) -> str:
    """
    如果 case_xxxx (4位数字)，就在数字前补一个 0
    例: case_0002 -> case_00002
    """
    m = re.match(r"(case_)(\d+)$", case_prefix)
    if m:
        num = m.group(2)
        if len(num) == 4:  # 四位数字
            return f"{m.group(1)}0{num}"
    return case_prefix

def main(args):
    df = pd.read_json(r"/data/sora/Medical_Reflection_Generation/HISTAI/HISTAI-metadata/metadata.json")

    # 强制所有列转成 string
    df = df.astype(str)
    df = df.sort_values(by="case_mapping").reset_index(drop=True)

    # 转成 Hugging Face Dataset
    ds = Dataset.from_pandas(df, preserve_index=False)

    repo_id = 'histai/HISTAI-mixed'
    files = list_repo_files(
        repo_id=repo_id,
        revision="main",
        token=os.getenv("HISTAI_TOKEN"),
        repo_type="dataset"
    )

    for data in tqdm(ds):
        out_root = Path(args.saved_path)
        case_prefix = data["case_mapping"].split("/")[-1]
        case_prefix = fix_case_prefix(case_prefix)
        img_rel = f"images/{case_prefix}_512.jpg"
        img_abs = out_root / img_rel
        if img_abs.exists():
            continue
        tiff_path = download_wsi_from_case_mapping(case_prefix, args.saved_path, files)
        img512 = tiff_to_512(tiff_path)
        text = data['conclusion']
        save_sample(img512, text, args.saved_path, case_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_path', type=str, default='./data_histai/')
    main(parser.parse_args())