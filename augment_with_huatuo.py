# -*- coding: utf-8 -*-
import os, io, json, argparse, traceback, sys, tempfile
from datetime import datetime
from pathlib import Path

import webdataset as wds
from webdataset.writer import ShardWriter
from PIL import Image
from tqdm import tqdm

# 让脚本能找到 HuatuoGPT-Vision 的 cli.py（里面有 HuatuoChatbot）
# 做法1：把仓库路径加到 PYTHONPATH；做法2：这里 sys.path.append 一下
# 比如：sys.path.append("/path/to/HuatuoGPT-Vision")
sys.path.append("/path/to/HuatuoGPT-Vision")   # ←←← 改成你的实际路径
from cli import HuatuoChatbot  # HuatuoGPT-Vision 官方提供的推理类

def pil_from_bytes(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

def pil_to_jpg_bytes(img: Image.Image, quality=95):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def make_side_by_side(imgL: Image.Image, imgR: Image.Image, pad=16, bg=(0,0,0)):
    h = max(imgL.height, imgR.height)
    def resize_to_h(im, h):
        if im.height == h:
            return im
        w = int(im.width * (h / im.height))
        return im.resize((w, h), Image.Resampling.LANCZOS)

    imgL = resize_to_h(imgL, h)
    imgR = resize_to_h(imgR, h)
    w = imgL.width + pad + imgR.width
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(imgL, (0, 0))
    canvas.paste(imgR, (imgL.width + pad, 0))
    return canvas

SYSTEM_PROMPT = (
    "You are a medical imaging quality control assistant, focusing only on image quality and compliance (non-diagnostic). "
    "The left image is a qualified reference, and the right image is the one to be improved. "
    "Compare them according to: positioning & rotation, field coverage, inspiration level, exposure/contrast, sharpness/motion artifacts, noise/grid artifacts, boundary cutoff, labeling compliance. "
    "Write a detailed reflection in natural language, highlighting key differences, possible root causes, and suggestions for improvement."
)

USER_PROMPT = (
    "Compare the image quality of the two chest X-rays. "
    "Focus only on aspects that need improvement in the right image. "
    "Provide a clear reflection that highlights the differences, explains possible causes, "
    "and suggests practical ways to improve the image quality."
)

def run(args):
    # 初始化模型
    bot = HuatuoChatbot(args.model_path)  # 例如 "FreedomIntelligence/HuatuoGPT-Vision-34B" 或本地路径

    # 读入旧数据分片（保持顺序，不shuffle）
    ds = (
        wds.WebDataset(args.input, shardshuffle=False)
        .to_tuple("real.png", "gen.png", "prompt.txt", "__key__")
    )

    # 写出新数据分片：在原有字段基础上，新增 huatuo.json 与 merged.jpg
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    uid = 0
    os.makedirs(Path(args.output).parent, exist_ok=True)

    with ShardWriter(args.output, maxcount=args.shard_maxcount) as sink:
        for real_bytes, gen_bytes, prompt_bytes, key in tqdm(ds, desc="augment"):

            # 1) 拼图
            real_pil = pil_from_bytes(real_bytes)
            gen_pil  = pil_from_bytes(gen_bytes)
            merged_pil = make_side_by_side(real_pil, gen_pil, pad=args.pad)
            merged_jpg = pil_to_jpg_bytes(merged_pil, quality=args.jpeg_quality)

            # 2) 调用 Huatuo 推理（用临时文件存一份，因 HuatuoChatbot 接口走路径列表）
            with tempfile.TemporaryDirectory() as td:
                mp = os.path.join(td, f"{key}_merged.jpg")
                with open(mp, "wb") as f:
                    f.write(merged_jpg)

                query = f"{SYSTEM_PROMPT}\n{USER_PROMPT}"
                out_text = bot.inference(query, [mp])  # 官方接口：文本 + [单图路径]
                print(out_text)


            # 3) 组织输出样本（保留原字段，新增两个字段）
            uid += 1
            sample = {
                "__key__": key,                         # 复用原key
                "real.png": real_bytes,                 # 原样保存
                "gen.png":  gen_bytes,                  # 原样保存
                "prompt.txt": prompt_bytes,             # 原样保存
                "merged.jpg": merged_jpg,               # 方便复核
                "huatuo.json": out_text.encode("utf-8"),
            }
            sink.write(sample)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
                    default="/data/sora/Medical_Reflection_Generation/data_with_bad/cxr-{000000..000007}.tar")
    ap.add_argument("--output",
                    default= "/data/sora/Medical_Reflection_Generation/out/ds_huatuo-%06d.tar")
    ap.add_argument("--model_path",
                    default='FreedomIntelligence/HuatuoGPT-Vision-34B')
    ap.add_argument("--shard_maxcount", type=int, default=2000)
    ap.add_argument("--pad", type=int, default=16, help="左右拼图的间隔像素")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()
    run(args)
