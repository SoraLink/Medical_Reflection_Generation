import argparse
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from webdataset.writer import ShardWriter

from evaluation import get_cxr_loader
from models import Diffusion
import webdataset as wds



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='itsanmolgupta/mimic-cxr-dataset')
    parser.add_argument('--model_path', type=str, default='./output')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./data_with_bad')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--shard_maxcount', type=int, default=2000)
    return parser.parse_args()

def pil_to_png_bytes(img: Image.Image) -> bytes:
    # 如需 16bit 保真，可改：img.save(buf, format="PNG", bits=16)
    # 或确保 img.mode in ["L","I;16","RGB"] 按需处理
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = Diffusion(args.model_path, args.device)
    loader = get_cxr_loader(args)

    tar_pattern = f"{args.output_dir}/cxr-%06d.tar"
    now = int(time.time())
    uid = 0
    # 在循环外打开一次 ShardWriter，自动按 maxcount 滚动分片
    with ShardWriter(tar_pattern, maxcount=args.shard_maxcount) as sink:
        for real_pils, prompts in tqdm(loader):
            # 生成
            gen_pils = model(prompts=prompts, num_inference_steps=args.steps)

            # 逐样本写入
            for i, (r, g, p) in enumerate(zip(real_pils, gen_pils, prompts)):
                # 生成稳定唯一键（时间戳 + 递增）
                uid += 1
                key = f"{now}-{uid:012d}"

                sample = {
                    "__key__": key,
                    "real.png": pil_to_png_bytes(r),
                    "gen.png":  pil_to_png_bytes(g),
                    "prompt.txt": str(p).encode("utf-8"),
                }
                sink.write(sample)


if __name__ == "__main__":
    main()