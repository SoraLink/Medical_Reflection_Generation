# save as encode_hf_to_parquet.py
import argparse, numpy as np
from datasets import load_dataset, Dataset
from PIL import Image
import jax
import jax.numpy as jnp
from vqgan_jax.modeling_flax_vqgan import VQModel

def img256(pil):
    return np.asarray(pil.convert("RGB").resize((256,256), Image.BICUBIC), dtype=np.float32)/255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--vqgan_id", default="dalle-mini/vqgan_imagenet_f16_16384")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    args = ap.parse_args()

    ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")["train"]
    ds = ds.train_test_split(test_size=0.1, seed=42)
    ds = ds['train'].train_test_split(test_size=args.val_ratio, seed=42)
    vq, params = VQModel.from_pretrained(args.vqgan_id, _do_init=False)

    @jax.jit
    def encode_batch(imgs_f32):  # [B,256,256,3] float32 in [0,1]
        return vq.encode(imgs_f32, params=params)  # -> [B,16,16]

    def to_rgb256(x):
        # 统一成 PIL RGB 256x256
        if x is None:
            raise ValueError("image is None")
        if not isinstance(x, Image.Image):
            x = Image.fromarray(np.asarray(x))
        if x.mode != "RGB":
            x = x.convert("RGB")
        return x.resize((256, 256), Image.BICUBIC)

    def encode_rows(batch, indices=None):
        try:
            pil_list = [to_rgb256(p) for p in batch["image"]]
            imgs = np.stack([np.asarray(p, dtype=np.float32) / 255.0 for p in pil_list], 0)  # [B,256,256,3]
            codes = np.asarray(encode_batch(jnp.asarray(imgs)))  # [B,16,16]
            enc = codes.reshape(codes.shape[0], -1).astype(np.int32)  # [B,256]
            caps = batch.get("findings") or [""] * enc.shape[0]
            caps = [c if isinstance(c, str) and c is not None else "" for c in caps]
            assert len(caps) == enc.shape[0]
            return {"caption": caps, "encoding": enc}
        except Exception as e:
            # 打印本批每张图的真实信息，锁定哪张脏
            dbg = []
            for i, x in enumerate(batch["image"]):
                if isinstance(x, Image.Image):
                    dbg.append((i, "PIL", x.size, x.mode))
                else:
                    arr = np.asarray(x)
                    dbg.append((i, "ND", getattr(arr, "shape", None), getattr(arr, "dtype", None)))
            print("encode_rows failed; indices:", indices, "details:", dbg)
            raise

    train = ds["train"].map(encode_rows, num_proc=1, remove_columns=ds["train"].column_names, batched=True, batch_size=32)
    val   = ds["test" ].map(encode_rows, num_proc=1, remove_columns=ds["test" ].column_names, batched=True, batch_size=32)

    Dataset.from_dict(train).to_parquet(f"{args.out_dir}/train.parquet")
    Dataset.from_dict(val  ).to_parquet(f"{args.out_dir}/val.parquet")
    print("Wrote:", f"{args.out_dir}/train.parquet", f"{args.out_dir}/val.parquet")

if __name__ == "__main__":
    main()
