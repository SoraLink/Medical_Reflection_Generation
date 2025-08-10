import argparse, os, sys, csv, shutil, subprocess
from pathlib import Path

def sh(cmd, env=None):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def export_mimic_to_csv(out_dir: Path, val_ratio: float = 0.05, size: int = 256):
    from datasets import load_dataset
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")

    # 若数据集仅有 train，就从它切分验证集
    split = ds["train"].train_test_split(test_size=val_ratio, seed=42)
    train_ds, test = split["train"], split["test"]
    split = split["train"].train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    def dump(split_ds, img_dir: Path, csv_path: Path):
        img_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "caption"])
            for i, ex in enumerate(split_ds):
                img = ex["image"]              # PIL Image
                cap = (ex.get("findings") or "").strip()
                # 统一到 256x256 RGB
                img = img.convert("RGB").resize((size, size), Image.BICUBIC)
                p = img_dir / f"{i:08d}.png"
                img.save(p)
                w.writerow([str(p.resolve()), cap])

    dump(train_ds, out_dir / "images_train", out_dir / "train.csv")
    dump(val_ds,   out_dir / "images_val",   out_dir / "val.csv")
    dump(test,  out_dir / "images_test",   out_dir / "test.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, default="./dallemini_work", help="工作目录")
    ap.add_argument("--out_dir", type=str, default="./outputs_dallemini_256", help="训练输出")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="验证比例")
    ap.add_argument("--per_device_batch", type=int, default=4, help="每卡 batch")
    ap.add_argument("--accum_steps", type=int, default=2, help="梯度累积步数")
    ap.add_argument("--lr", type=float, default=2e-5, help="学习率")
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bfloat16","fp32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", type=str, default="dalle-mini/dalle-mini", help="dalle-mini/dalle-mega")
    ap.add_argument("--vqgan_id", type=str, default="dalle-mini/vqgan_imagenet_f16_16384")
    args = ap.parse_args()

    work = Path(args.workdir).absolute()
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)

    repo = work / "dalle-mini"

    data_dir = repo / "data"
    data_dir.mkdir(exist_ok=True)
    export_dir = data_dir / "mimic256"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_mimic_to_csv(export_dir, val_ratio=args.val_ratio, size=256)


if __name__ == "__main__":
    main()