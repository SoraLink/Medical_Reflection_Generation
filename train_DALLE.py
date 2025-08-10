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
    def encode_batch(imgs):
        out = vq.encode(imgs, params=params)

        # 优先：dict 路径（最稳）
        if isinstance(out, dict):
            codes = out.get("encoding_indices") or out.get("indices")
            if codes is None:
                raise ValueError(f"encode()未找到indices，keys={list(out.keys())}")
        elif isinstance(out, (tuple, list)):
            # 常见顺序：(z_q, indices, ...)
            # 先试第2个；不行就扫一遍挑“像索引”的那个
            cand = out[1] if len(out) > 1 else out[0]
            xa = jnp.asarray(cand)
            if not (xa.ndim >= 2 and xa.shape[-2:] == (16, 16) and xa.dtype in (jnp.int32, jnp.int16, jnp.int64)):
                # 扫描所有元素，挑 dtype 像整数、尾部是16x16的
                picked = None
                for x in out:
                    x = jnp.asarray(x)
                    if x.ndim >= 2 and x.shape[-2:] == (16, 16) and x.dtype in (jnp.int32, jnp.int16, jnp.int64):
                        picked = x
                        break
                if picked is None:
                    # 仍然没有整数索引；可能只有 z_q (B,16,16,D)，不能直接当索引用
                    shapes = [tuple(jnp.asarray(x).shape) for x in out]
                    raise ValueError(
                        f"encode()没有返回离散索引（只看到了嵌入/其它）：{shapes}。请升级 vqgan_jax 或改为 dict['encoding_indices']。")
                codes = picked
            else:
                codes = xa
        else:
            # 稀有：单对象输出，尽量兜底
            xa = jnp.asarray(out)
            if xa.ndim == 2 and xa.shape == (16, 16):
                codes = xa
            else:
                raise ValueError(f"无法识别的 encode() 返回类型/形状: {type(out)}, {tuple(xa.shape)}")

        # 统一到 (B,16,16)
        codes = jnp.asarray(codes)
        if codes.ndim == 2:  # (16,16) → (1,16,16)
            codes = codes[None, ...]
        elif codes.ndim == 4 and codes.shape[-3:-1] == (16, 16):
            # 这是 z_q 的形状 (B,16,16,D)，不能直接当索引
            raise ValueError(f"拿到的是嵌入 (B,16,16,D={codes.shape[-1]})，不是索引。需要 indices。")

        assert codes.shape[-2:] == (16, 16), f"期望(16,16)，实际{codes.shape}"
        return codes  # (B,16,16)

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