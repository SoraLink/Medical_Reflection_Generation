# save as encode_hf_to_parquet.py
import argparse, numpy as np
from datasets import load_dataset
from PIL import Image
import jax
import jax.numpy as jnp
from vqgan_jax.modeling_flax_vqgan import VQModel

def img256_norm_m11(pil_img):
    """RGB->[-1,1], 256x256"""
    arr = np.asarray(pil_img.convert("RGB").resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0
    return arr * 2.0 - 1.0  # [-1, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--vqgan_id", default="dalle-mini/vqgan_imagenet_f16_16384")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # 1) 准备数据划分
    ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")["train"]
    ds = ds.train_test_split(test_size=0.1, seed=42)
    ds = ds["train"].train_test_split(test_size=args.val_ratio, seed=42)

    # 2) 加载 VQ 模型
    vq, params = VQModel.from_pretrained(args.vqgan_id, _do_init=False)

    @jax.jit
    def encode_batch(imgs_f32_m11):  # [B,256,256,3], float32 in [-1,1]
        out = vq.encode(imgs_f32_m11, params=params)
        if not isinstance(out, dict) or "encoding_indices" not in out:
            raise ValueError(
                f"VQ encode()返回不含encoding_indices: type={type(out)}, "
                f"keys={list(out.keys()) if isinstance(out, dict) else None}"
            )
        codes = out["encoding_indices"]  # 期望 [B,16,16]
        h, w = int(codes.shape[-2]), int(codes.shape[-1])
        if (h, w) != (16, 16):
            raise ValueError(f"VQ codes 空间尺寸应为 16x16，实际 {h}x{w}。检查 vq 模型或预处理。")
        return codes  # int-like, [B,16,16]

    def encode_rows(batch):
        pil_list = batch["image"]
        imgs = np.stack([img256_norm_m11(p) for p in pil_list], axis=0).astype(np.float32)  # [B,256,256,3], [-1,1]
        codes = encode_batch(jnp.asarray(imgs))                                             # [B,16,16]
        enc = np.asarray(codes).reshape(codes.shape[0], -1).astype(np.int32)               # [B,256]

        # 文本 caption
        caps_raw = batch.get("findings")
        if caps_raw is None:
            caps = [""] * enc.shape[0]
        else:
            caps = [c if isinstance(c, str) and c is not None else "" for c in caps_raw]

        # 1D 解码端 mask（与长度 256 对齐）
        dam = np.ones_like(enc, dtype=np.int32)  # 没有 pad 的场景，直接全 1

        return {
            "caption": caps,
            "encoding": enc,                 # 方便检查
            "labels": enc,                   # 训练常用
            "decoder_input_ids": enc,        # 保险：有些 trainer 直接用这个
            "decoder_attention_mask": dam,   # 1D mask，与长度匹配
        }

    # 3) 批量编码并写 parquet
    train = ds["train"].map(
        encode_rows,
        batched=True,
        batch_size=args.batch_size,
        num_proc=1,  # JAX 与多进程常冲突，这里保持单进程
        remove_columns=ds["train"].column_names,
    )
    val = ds["test"].map(
        encode_rows,
        batched=True,
        batch_size=args.batch_size,
        num_proc=1,
        remove_columns=ds["test"].column_names,
    )

    train.to_parquet(f"{args.out_dir}/train.parquet")
    val.to_parquet(f"{args.out_dir}/val.parquet")
    print("Wrote:", f"{args.out_dir}/train.parquet", f"{args.out_dir}/val.parquet")

if __name__ == "__main__":
    main()