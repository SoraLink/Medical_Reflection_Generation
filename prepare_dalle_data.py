# export_mimic_cxr_image_text.py
from pathlib import Path
import re
from datasets import load_dataset

OUTDIR = Path("mimic_cxr_image_text")
OUTDIR.mkdir(parents=True, exist_ok=True)

def normalize_caption(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    return re.sub(r"\s+", " ", s)

print("Loading dataset: itsanmolgupta/mimic-cxr-dataset (train split)")
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train")
ds = ds.train_test_split(test_size=0.1, seed=42)
ds = ds["train"]

n_saved = 0
for ex in ds:
    img = ex["image"]  # 已经是 PIL.Image.Image
    cap = normalize_caption(ex.get("findings", ""))
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
