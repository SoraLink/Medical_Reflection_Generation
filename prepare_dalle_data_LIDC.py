# export_mimic_cxr_image_text.py
import hashlib
import io
from pathlib import Path
import re

import torch
from PIL import Image

OUTDIR = Path("mimic_cxr_image_text")
OUTDIR.mkdir(parents=True, exist_ok=True)

def normalize_caption(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return re.sub(r"\s+", " ", s)

pattern = "/data/LIDC/augmented/ds_ct-{000000..000007}.tar"
import torchvision.transforms as T
def _bytes_to_pil(x):
    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("L")
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return T.ToPILImage()(x)
    raise TypeError(f"unexpected image type: {type(x)}")

def load_img(x):
    return _bytes_to_pil(x)

def load_txt(x):
    return x if isinstance(x, str) else x.decode("utf-8")

def split_by_hash(key: str, m: int = 20):
    # 把样本稳定地分到 0..m-1 的桶里；m=20 相当于 5% 验证集
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % m
    return h

import webdataset as wds

base = (
    wds.WebDataset(
        pattern,
        nodesplitter=wds.split_by_node,
        shardshuffle=False
    )   # 显式关掉分片shuffle
    # 不要调用 .decode(...)，避免触发内置的 json 解码器
    .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
    .map_tuple(
        load_img,          # real.png  -> PIL.Image
        load_img,          # gen.png   -> PIL.Image
        load_txt,          # prompt.txt-> str
        load_txt,          # huatuo.json(其实是纯文本)-> str
        lambda x: x        # __key__   -> str
    )
    .with_length(13953)
)
train_ds = base.select(lambda s: split_by_hash(s[-1], 20) != 0)

n_saved = 0
for ex in train_ds:
    real, gen, prompt, huatuo, key = zip(*ex)
    img = real  # 已经是 PIL.Image.Image
    cap = normalize_caption(prompt)
    if not cap:
        continue

    stem = f"{n_saved:07d}"
    img_path = OUTDIR / f"{stem}.jpg"
    txt_path = OUTDIR / f"{stem}.txt"

    img.convert("RGB").save(img_path, format="JPEG", quality=95, subsampling=1)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(cap + "\n")

    n_saved += 1
    if n_saved % 1000 == 0:
        print(f"saved {n_saved} samples")

print(f"Done. total saved: {n_saved}, outdir={OUTDIR.resolve()}")
