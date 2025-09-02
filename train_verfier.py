# train_bt_verifier_qwenvl.py
import io
import os, json, math, argparse, random
import shutil
from collections import deque

from PIL import Image
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup, Qwen2_5_VLForConditionalGeneration
)
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import webdataset as wds
import random, numpy as np

def _bytes_to_pil(x):
    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return T.ToPILImage()(x)
    raise TypeError(f"unexpected image type: {type(x)}")

def load_img(x):
    return _bytes_to_pil(x)

def load_txt(x):
    return x if isinstance(x, str) else x.decode("utf-8")

# ---------------- Reward head ----------------
class ScalarHead(nn.Module):
    """把最后一层 hidden state 的 pooled 表示 -> 标量分数 r(x,y)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, pooled):  # [B, H]
        return self.mlp(pooled).squeeze(-1)  # [B]

# ---------------- QwenVL Wrapper ----------------
class QwenVLRewardModel(nn.Module):
    """
    用 Qwen2.5-VL-3B 做骨干，在前向里做多模态融合；
    取最后一层 hidden_state 的“最后非 PAD token”向量作为 pooled，再接一个 MLP -> 标量。
    """
    def __init__(self, name="Qwen/Qwen2.5-VL-3B-Instruct", lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 lora_on_lang=True, unfreeze_vision=True, bf16=True):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            low_cpu_mem_usage=True
        )
        hidden = self.model.config.hidden_size
        self.head = ScalarHead(hidden)

        # 视觉塔是否全量训练（贴论文）
        if not unfreeze_vision:
            for n, p in self.model.named_parameters():
                if "vision_tower" in n or "visual" in n:
                    p.requires_grad = False

        # 语言侧 LoRA（贴论文）
        if lora_on_lang:
            target = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
            peft_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                  bias="none", task_type="CAUSAL_LM", target_modules=target)
            self.model = get_peft_model(self.model, peft_cfg)

    def forward_score(self, images, texts, device):
        """
        返回 r(x,y) 标量分数；不开启生成，仅前向拿 hidden_states。
        """
        # QwenVL 的 processor 会自动把多模态拼好
        batch = self.processor(text=texts, images=images, return_tensors="pt").to(device)
        out = self.model(**batch, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]                # [B, T, H]
        B, T, H = hs.shape

        attn = batch.get("attention_mask", None)
        if attn is not None:
            # sum 可能是浮点，做索引要 long
            idx = attn.to(torch.int64).sum(dim=-1) - 1
        elif "input_ids" in batch and getattr(self.model.config, "pad_token_id", None) is not None:
            pad_id = self.model.config.pad_token_id
            idx = (batch["input_ids"] != pad_id).to(torch.int64).sum(dim=-1) - 1
        else:
            # 最保险的兜底：直接取最后一位
            idx = torch.full((B,), T - 1, device=hs.device, dtype=torch.long)
        idx = idx.clamp(min=0, max=T - 1)  # 防越界
        pooled = hs[torch.arange(B, device=hs.device), idx]  # 用 hs.device 更稳
        score = self.head(pooled)                 # [B]
        return score

# ---------------- Loss / Metrics ----------------
def bt_loss(sw, sl):
    # L = -log σ(Δ)  的数值稳定形式
    return F.softplus(-(sw - sl)).mean()

@torch.no_grad()
def pairwise_acc(sw, sl):
    return (sw > sl).float().mean().item()

# ---------------- Train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)          # QwenVL 显存更大，先保守
    ap.add_argument("--lr", type=float, default=2e-6)              # 论文同量级
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--save_dir", default="./bt_verifier_qwenvl")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--unfreeze_vision", action="store_true")      # 贴论文：打开视觉侧全量更新
    ap.add_argument("--no_lora", action="store_true")              # 若不想 LoRA，传此开关
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    pattern = "/data/sora/Medical_Reflection_Generation/out/ds_huatuo-{000000..000007}.tar"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def keys_stream():
        # 只读 key，不解码图片；不要 nodesplitter，这样能看到全量
        return wds.WebDataset(pattern, nodesplitter=None, shardshuffle=False).to_tuple("__key__")

    def base_stream(nodesplitter=None, do_shuffle=False):
        ds = wds.WebDataset(
            pattern,
            nodesplitter=nodesplitter,  # 训练放开；验证用 None
            shardshuffle=False
        )
        if do_shuffle:
            ds = ds.shuffle(1000)  # 训练期打乱放在 WebDataset
        return (ds
                .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
                .map_tuple(load_img, load_img, load_txt, load_txt, lambda x: x))

    # 1) 统计数据规模 + 拿到“尾部 100 个” val keys（只看 key，快速且可复现）
    from collections import deque
    def tail_keys_and_count(k=100):
        dq = deque(maxlen=k)
        total = 0
        for (key,) in keys_stream():
            dq.append(key)
            total += 1
        return set(dq), total

    val_keys, total_samples = tail_keys_and_count(100)

    # 2) 构建互斥的 train / val dataset（train 排除 val_keys；val 只保留 val_keys）
    def to_pairwise(sample):
        real_img, gen_img, prompt, _huatuo_txt, _key = sample
        return prompt, real_img, gen_img  # 默认 real 赢；需要可再按 json 改向

    def make_loader(split, batch_size=32, num_workers=8):
        is_train = (split == "train")
        ds = base_stream(
            nodesplitter=(wds.split_by_node if is_train else None),
            do_shuffle=is_train
        ).select(lambda s: (s[-1] not in val_keys) if is_train else (s[-1] in val_keys)
                 ).map(to_pairwise)

        def collate(batch):
            p, w, l = zip(*batch)
            return list(p), list(w), list(l)

        # IterableDataset 下 DataLoader 的 shuffle 必须是 False
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, collate_fn=collate)

    train_loader = make_loader("train", batch_size=args.batch_size)
    val_loader = make_loader("val", batch_size=args.batch_size)

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QwenVLRewardModel(
        lora_on_lang=not args.no_lora,
        unfreeze_vision=args.unfreeze_vision,
        bf16=args.bf16
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    # 可选：如果用了 PEFT，打印更详细的信息（LoRA 层数等）
    if hasattr(model, "model") and hasattr(model.model, "print_trainable_parameters"):
        model.model.print_trainable_parameters()

    # 只优化需要训练的参数（LoRA + 视觉/头部）
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    train_samples = max(0, total_samples - len(val_keys))
    steps_per_epoch = math.ceil(train_samples / max(1, args.batch_size))
    total_steps = args.epochs * steps_per_epoch
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    best_val = -1.0

    def save_raw(model_wrapper, save_root, tag):
        ckpt = os.path.join(save_root, tag)
        os.makedirs(ckpt, exist_ok=True)
        model_wrapper.model.save_pretrained(os.path.join(ckpt, "adapter"))
        base = model_wrapper.model.get_base_model()
        (base.model if hasattr(base, "model") else base).save_pretrained(os.path.join(ckpt, "base"))
        torch.save(model_wrapper.head.state_dict(), os.path.join(ckpt, "head.pt"))
        model_wrapper.processor.save_pretrained(os.path.join(ckpt, "processor"))

    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epochs}")
        for prompts, wins, loss in pbar:

            sw = model.forward_score(wins, prompts, device)
            sl = model.forward_score(loss, prompts, device)
            loss_bt = bt_loss(sw, sl)

            optim.zero_grad(set_to_none=True)
            loss_bt.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            pbar.set_postfix(loss=f"{loss_bt.item():.4f}", acc=f"{pairwise_acc(sw, sl):.3f}",
                             lr=f"{optim.param_groups[0]['lr']:.2e}")

        if val_loader:
            model.eval()
            vs, va = [], []
            with torch.no_grad():
                for prompts, wins, loss in tqdm(val_loader, desc="Valid"):
                    sw = model.forward_score(wins, prompts, device)
                    sl = model.forward_score(loss, prompts, device)
                    vs.append(bt_loss(sw, sl).item())
                    va.append(pairwise_acc(sw, sl))
            mv, ma = sum(vs)/len(vs), sum(va)/len(va)
            print(f"[Val] loss={mv:.4f} acc={ma:.4f}")
            if ma > best_val:
                best_val = ma
                save_raw(model, args.save_dir, tag=f"best_raw_{best_val:.4f}")

    def merge_from_disk(save_root, src_tag, dst_tag="best_merged"):
        src = os.path.join(save_root, src_tag)
        dst = os.path.join(save_root, dst_tag)
        os.makedirs(dst, exist_ok=True)
        base = AutoModelForCausalLM.from_pretrained(os.path.join(src, "base"), trust_remote_code=True,
                                                    torch_dtype=torch.float32)
        peft = PeftModel.from_pretrained(base, os.path.join(src, "adapter"))
        merged = peft.merge_and_unload()
        merged.save_pretrained(os.path.join(dst, "base"))
        shutil.copy2(os.path.join(src, "head.pt"), os.path.join(dst, "head.pt"))
        AutoProcessor.from_pretrained(os.path.join(src, "processor"), trust_remote_code=True).save_pretrained(
            os.path.join(dst, "processor"))

    # 训练结束时调用
    merge_from_disk(args.save_dir, src_tag=f"best_raw_{best_val:.4f}", dst_tag=f"best_merged_{best_val:.4f}")
    print("Saved last.")

if __name__ == "__main__":
    main()
