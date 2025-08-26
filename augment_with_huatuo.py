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
    "You are a medical imaging evaluation assistant. "
    "The left image is a real chest X-ray. "
    "The diagnostic report below was written by a doctor based on this real image. "
    "The right image is a generated chest X-ray created from that diagnostic description. "
    "Your task is to evaluate how well the generated image (right) aligns with the diagnostic description, "
    "using the real X-ray (left) as the ground truth reference. "
    "Then, identify what aspects of the generated image should be improved to: "
    "1) better match the diagnostic description, and "
    "2) look more realistic as a chest X-ray. "
    "Focus on anatomical accuracy, realism, and consistency with the description. "
    "Provide a detailed reflection that highlights discrepancies and gives concrete improvement suggestions."
)

USER_PROMPT = (
    "Diagnostic description (from the real image):\n"
    "{diagnostic_description}\n\n"
    "Task:\n"
    "Provide clear suggestions on how to modify the generated image to better reflect the description and achieve higher realism."
)

def load_done_keys(state_path: Path) -> set:
    """从进度文件加载已完成的 key 集合。"""
    done = set()
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    k = obj.get("key")
                    if k is not None:
                        done.add(k)
                except Exception:
                    # 容忍坏行
                    continue
    return done

def append_done_key(state_path: Path, key: str):
    """把一个完成的 key 追加到进度文件。"""
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key}, ensure_ascii=False) + "\n")

def run(args):
    # 初始化模型
    bot = HuatuoChatbot(args.model_path)  # 例如 "FreedomIntelligence/HuatuoGPT-Vision-34B" 或本地路径

    # 读入旧数据分片（保持顺序，不shuffle）
    ds = (
        wds.WebDataset(args.input, shardshuffle=False)
        .to_tuple("real.png", "gen.png", "prompt.txt", "__key__")
    )

    # 写出新数据分片：在原有字段基础上，新增 huatuo.json 与 merged.jpg
    uid = 0
    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_file) if args.state_file else (
        out_dir / (out_path.stem + ".state.jsonl")
    )

    done_keys = load_done_keys(state_path) if args.resume else set()
    if args.resume and done_keys:
        print(f"[resume] loaded {len(done_keys)} done keys from {state_path}")

    with ShardWriter(args.output, maxcount=args.shard_maxcount) as sink:
        for real_bytes, gen_bytes, prompt_bytes, key in tqdm(ds, desc="augment"):
            if args.resume and key in done_keys:
                continue
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
                diagnostic_description = prompt_bytes.decode("utf-8")
                query = (SYSTEM_PROMPT + "\n" + USER_PROMPT).format(
                    diagnostic_description=diagnostic_description
                )
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
                "huatuo.json": out_text[0].encode("utf-8"),
            }
            sink.write(sample)
            append_done_key(state_path, key)

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
    ap.add_argument("--resume", action="store_true",
                    help="启用断点续跑：跳过进度文件中已完成的 key")
    ap.add_argument("--state_file", default="/data/sora/Medical_Reflection_Generation/out/ds_huatuo.state.jsonl",
                    help="进度文件路径（默认：<output_stem>.state.jsonl）")
    args = ap.parse_args()
    run(args)
