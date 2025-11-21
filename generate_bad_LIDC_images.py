# -*- coding: utf-8 -*-
"""
LIDC 成对样本生成：清晰图 vs 降质图（支持 mask 引导）
依赖：
  pip install numpy opencv-python pydicom SimpleITK scikit-image torch
"""
import argparse
import io
import os
import json
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import cv2
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm
import webdataset as wds
from webdataset.writer import ShardWriter
import requests

import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline

try:
    import SimpleITK as sitk
except Exception:
    sitk = None
    warnings.warn("SimpleITK 未安装，若需要读取 NIfTI 请先安装：pip install SimpleITK")

from skimage.morphology import disk, binary_erosion, binary_dilation

class Huatuo:
    def __init__(self):
        self.server = "http://127.0.0.1:6006/inference"

    def inference(self, prompt: str, pil_image) -> str:
        if pil_image is None:
            raise ValueError("Pil image cannot be None")

        buf = io.BytesIO()
        pil_image.save(buf, "JPEG", quality=95)
        buf.seek(0)
        files = {'image': ("img.jpg", buf.getvalue(), "image/jpeg")}
        resp = requests.post(self.server, data={"prompt": prompt}, files=files, timeout=180)

        resp.raise_for_status()
        j = resp.json()
        if not j.get("ok", False):
            raise RuntimeError(j.get("error"))
        return j['text']

def pil_to_png_bytes(img: Image.Image) -> bytes:
    # 如需 16bit 保真，可改：img.save(buf, format="PNG", bits=16)
    # 或确保 img.mode in ["L","I;16","RGB"] 按需处理
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def append_done_key(state_path: Path, key: str):
    """把一个完成的 key 追加到进度文件。"""
    with open(state_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key}, ensure_ascii=False) + "\n")

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


def _soft_alpha(mask01, blur_frac=0.02):
    """把二值mask变成软边alpha: 先轻度膨胀/腐蚀，再高斯模糊做羽化"""
    m = (mask01 > 0).astype(np.float32)
    H, W = m.shape
    k = max(3, int(round(min(H, W) * blur_frac)) | 1)  # 奇数核
    # 可选：先把边缘稍微膨胀一下再模糊，减少“锯齿/方块感”
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    m = np.clip(m, 0, 1)
    return m

def _random_affine(img, mask, max_rot=10, scale_range=(0.95, 1.05)):
    """对 patch 和对应mask做同样的轻微仿射，增加自然度"""
    H, W = img.shape
    c = (W/2.0, H/2.0)
    ang = np.random.uniform(-max_rot, max_rot)
    sc  = np.random.uniform(*scale_range)
    M = cv2.getRotationMatrix2D(c, ang, sc)
    img2  = cv2.warpAffine(img,  M, (W, H), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT)
    mask2 = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return img2, mask2

def mask_shift_paste_smooth(img01, mask01, max_shift=12, alpha_strength=0.8,
                            use_poisson_prob=0.4):
    """
    更自然的“随机挪位贴图”：
      - 软边alpha羽化
      - 贴图前对patch做轻微仿射
      - 随机使用 Poisson 融合（cv2.seamlessClone）
    """
    ys, xs = np.where(mask01 > 0)
    if len(ys) == 0:
        return img01.copy()

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch_raw = img01[y0:y1, x0:x1].copy()
    m_raw     = mask01[y0:y1, x0:x1].astype(np.uint8)

    # 软边 alpha（羽化）
    alpha_soft = _soft_alpha(m_raw, blur_frac=0.02) * alpha_strength

    # 轻微仿射增强
    patch_warp, alpha_warp = _random_affine(patch_raw, alpha_soft,
                                            max_rot=10, scale_range=(0.95, 1.05))

    H, W = img01.shape
    hh, ww = y1 - y0, x1 - x0
    dy = np.random.randint(-max_shift, max_shift + 1)
    dx = np.random.randint(-max_shift, max_shift + 1)
    yy0 = int(np.clip(y0 + dy, 0, H - hh))
    xx0 = int(np.clip(x0 + dx, 0, W - ww))

    out = img01.copy()

    # 方式A：Poisson seamlessClone（更自然，代价稍高）
    if np.random.rand() < use_poisson_prob:
        # 需要3通道uint8
        src = (np.clip(patch_warp*255, 0, 255)).astype(np.uint8)
        dst = (np.clip(out*255, 0, 255)).astype(np.uint8)
        src3 = cv2.merge([src, src, src])
        dst3 = cv2.merge([dst, dst, dst])

        # 构造Poisson的mask（硬mask也行，这里用软alpha>阈值）
        pm = (alpha_warp > 0.1).astype(np.uint8) * 255
        center = (xx0 + ww//2, yy0 + hh//2)

        try:
            mixed = cv2.seamlessClone(src3, dst3, pm, center, cv2.MIXED_CLONE)
            out = mixed[...,0].astype(np.float32)/255.0
        except cv2.error:
            # 失败则回退到alpha融合
            roi = out[yy0:yy0+hh, xx0:xx0+ww]
            out[yy0:yy0+hh, xx0:xx0+ww] = roi*(1-alpha_warp) + patch_warp*alpha_warp
    else:
        # 方式B：软边 alpha 融合（快速稳定）
        roi = out[yy0:yy0+hh, xx0:xx0+ww]
        out[yy0:yy0+hh, xx0:xx0+ww] = roi*(1-alpha_warp) + patch_warp*alpha_warp

    return np.clip(out, 0, 1).astype(np.float32)


# -----------------------------
# 基础：DICOM / NIfTI 读写与 HU/窗宽窗位
# -----------------------------
def load_dicom_series(folder: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    从一个 LIDC 病人/序列文件夹读取 DICOM 切片，并按 InstanceNumber 排序。
    返回:
      volume_hu: (D,H,W) float32, HU 值
      spacing: (dz, dy, dx)
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".dcm")]
    if not files:
        raise FileNotFoundError(f"No DICOM found in {folder}")

    slices = []
    for fp in files:
        d = pydicom.dcmread(fp)
        if hasattr(d, "ImagePositionPatient") and hasattr(d, "PixelData"):
            slices.append(d)
    # 按 InstanceNumber 排序（如果没有就按 SliceLocation 或文件名兜底）
    slices.sort(key=lambda x: getattr(x, "InstanceNumber", 1e9))

    # spacing
    dz = float(getattr(slices[1], "ImagePositionPatient", [0, 0, 1])[2] -
               getattr(slices[0], "ImagePositionPatient", [0, 0, 0])[2]) if len(slices) > 1 else 1.0
    dy, dx = map(float, getattr(slices[0], "PixelSpacing", [1.0, 1.0]))

    # 叠加为 HU
    vol = []
    for s in slices:
        img = s.pixel_array.astype(np.int16)
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        hu = img * slope + intercept
        vol.append(hu)
    volume_hu = np.stack(vol).astype(np.float32)  # (D,H,W)
    return volume_hu, (abs(dz), dy, dx)


def load_nifti(path: str) -> np.ndarray:
    """
    读取 NIfTI（.nii / .nii.gz），返回 numpy (D,H,W) float32 或 uint8（二值 mask）
    """
    if sitk is None:
        raise ImportError("需要 SimpleITK 读取 NIfTI：pip install SimpleITK")
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr


def apply_window(hu_img: np.ndarray, w: int = 1500, l: int = -600) -> np.ndarray:
    """
    HU -> 窗宽窗位 -> 归一化到 [0,1]
    """
    lo, hi = l - w // 2, l + w // 2
    x = np.clip(hu_img, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return x.astype(np.float32)


# -----------------------------
# 掩膜相关（腐蚀/膨胀）
# -----------------------------
def morph_mask(mask: np.ndarray, op: str = "dilate", radius: int = 2) -> np.ndarray:
    """
    2D 掩膜形态学
    """
    mask = (mask > 0).astype(np.uint8)
    se = disk(radius)
    if op == "erode":
        out = binary_erosion(mask.astype(bool), se)
    else:
        out = binary_dilation(mask.astype(bool), se)
    return out.astype(np.uint8)


# -----------------------------
# 降质算子
# -----------------------------
def random_window_from_hu(hu_img: np.ndarray,
                          w_range=(1200, 1800),
                          l_range=(-750, -400)) -> Tuple[np.ndarray, Dict]:
    w = random.randint(*w_range)
    l = random.randint(*l_range)
    img = apply_window(hu_img, w=w, l=l)
    return img, {"type": "window", "W": w, "L": l}


def add_gaussian_noise(img01: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img01.shape).astype(np.float32)
    return np.clip(img01 + noise, 0, 1)


def add_multiplicative_noise(img01: np.ndarray, sigma: float) -> np.ndarray:
    noise = 1.0 + np.random.normal(0, sigma, img01.shape).astype(np.float32)
    return np.clip(img01 * noise, 0, 1)


def motion_blur(img01: np.ndarray, ksize: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    s = k.sum()
    if s > 0:
        k /= s
    out = cv2.filter2D(img01, -1, k, borderType=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 1)


def down_up_sampling(img01: np.ndarray, min_scale: float = 0.4, max_scale: float = 0.8) -> np.ndarray:
    """
    先下采样再上采样，模拟分辨率下降导致的模糊
    """
    h, w = img01.shape
    scale = random.uniform(min_scale, max_scale)
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(img01, (nw, nh), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


def jpeg_artifact(img01: np.ndarray, quality: int) -> np.ndarray:
    """
    利用 OpenCV JPEG 编码产生压缩伪影；输入[0,1]灰度，先转 0-255 单通道
    """
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    enc = cv2.imencode(".jpg", img8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return dec


def adjust_gamma(img01: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    return np.power(np.clip(img01, 0, 1), inv).astype(np.float32)


def mask_shift_paste(img01: np.ndarray, mask01: np.ndarray,
                     max_shift: int = 12, alpha: float = 0.7) -> np.ndarray:
    """
    将掩膜区域裁出，随机平移后再贴回，alpha 融合（不改标签）
    """
    ys, xs = np.where(mask01 > 0)
    if len(ys) == 0:
        return img01.copy()
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch = img01[y0:y1, x0:x1].copy()
    pmask = mask01[y0:y1, x0:x1].astype(np.float32)

    h, w = img01.shape
    dy = random.randint(-max_shift, max_shift)
    dx = random.randint(-max_shift, max_shift)
    hh, ww = y1 - y0, x1 - x0
    yy0 = np.clip(y0 + dy, 0, h - hh)
    xx0 = np.clip(x0 + dx, 0, w - ww)

    out = img01.copy()
    # 可选：原位轻度淡化，避免双影
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - pmask) + out[y0:y1, x0:x1] * pmask * 0.5

    roi = out[yy0:yy0 + hh, xx0:xx0 + ww]
    out[yy0:yy0 + hh, xx0:xx0 + ww] = roi * (1 - pmask * alpha) + patch * (pmask * alpha)
    return np.clip(out, 0, 1)


# -----------------------------
# 降质流水线（随机组合）
# -----------------------------
def make_degraded(
    img01: np.ndarray,
    mask01: Optional[np.ndarray] = None,
    hu_img: Optional[np.ndarray] = None,
    change_mask_size: bool = False,
    delta_mask_radius_range: Tuple[int, int] = (1, 3),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    生成降质图像（与清晰图同尺寸），默认不改掩膜；若 change_mask_size=True，会对 mask 做随机腐蚀/膨胀
    返回:
      degraded01, mask_out, params
    """
    params = {}

    # 1) 可选：随机窗宽窗位（如果提供了 HU）
    work = img01.copy()
    if hu_img is not None and random.random() < 0.8:
        work, p = random_window_from_hu(hu_img,
                                        w_range=(1200, 1800),
                                        l_range=(-750, -400))
        params["window"] = p

    # 2) 掩膜区域随机挪位贴回（制造位置扰动）
    if mask01 is not None and random.random() < 0.7:
        max_shift = random.randint(6, 16)
        alpha = random.uniform(0.5, 0.9)
        work = mask_shift_paste_smooth(work, mask01,
                                       max_shift=np.random.randint(6, 16),
                                       alpha_strength=np.random.uniform(0.6, 0.9),
                                       use_poisson_prob=0.4)
        params["mask_shift_paste"] = {"max_shift": max_shift, "alpha": round(alpha, 3)}

    # 3) 运动模糊
    if random.random() < 0.6:
        k = random.choice([7, 9, 11, 13])
        ang = random.uniform(-20, 20)
        work = motion_blur(work, ksize=k, angle_deg=ang)
        params["motion_blur"] = {"ksize": k, "angle": round(ang, 2)}

    # 4) 降分辨率再上采样
    if random.random() < 0.6:
        work = down_up_sampling(work, 0.4, 0.8)
        params["down_up_sampling"] = True

    # 5) 噪声
    if random.random() < 0.8:
        if random.random() < 0.5:
            sigma = random.uniform(0.01, 0.06)
            work = add_gaussian_noise(work, sigma=sigma)
            params["gaussian_noise"] = {"sigma": round(sigma, 3)}
        else:
            sigma = random.uniform(0.02, 0.08)
            work = add_multiplicative_noise(work, sigma=sigma)
            params["multiplicative_noise"] = {"sigma": round(sigma, 3)}

    # # 6) Gamma/对比度
    # if random.random() < 0.5:
    #     gamma = random.uniform(0.8, 1.4)
    #     work = adjust_gamma(work, gamma=gamma)
    #     params["gamma"] = {"gamma": round(gamma, 3)}

    # 7) JPEG 伪影
    if random.random() < 0.6:
        q = random.randint(20, 60)  # 质量越低伪影越重
        work = jpeg_artifact(work, quality=q)
        params["jpeg"] = {"quality": q}

    # 8) 可选：改变病灶大小（对 mask 做腐蚀/膨胀）
    mask_out = mask01.copy() if mask01 is not None else None
    # if change_mask_size and (mask_out is not None) and random.random() < 0.7:
    #     r = random.randint(*delta_mask_radius_range)
    #     op = random.choice(["dilate", "erode"])
    #     mask_out = morph_mask(mask_out, op=op, radius=r)
    #     params["mask_morph"] = {"op": op, "radius": r}
    #
    #     # 可选：同步对图像边缘做轻微模糊以贴合标签变化（减少 label/边界 mismatch）
    #     if random.random() < 0.5:
    #         work = motion_blur(work, ksize=7, angle_deg=random.uniform(-10, 10))
    #         params["edge_soften"] = True

    return np.clip(work, 0, 1).astype(np.float32), mask_out, params


# -----------------------------
# PyTorch 数据集（2D 切片）
# -----------------------------
class LIDCSlicePairDataset(Dataset):
    """
    输入：一组 (hu_slice, mask_slice) 或者 (dicom_path, mask_path)
    为了通用，这里提供两种构造方式：
      1) 你已经有 numpy 切片列表：list_of_slices, list_of_masks
      2) 你有 DICOM 序列路径 + NIfTI 掩膜路径，先在外面切成 2D 再传进来
    这里采用方式 (1)：直接接收数组。
    """
    def __init__(
        self,
        hu_slices: List[np.ndarray],        # 每个元素 (H,W) HU
        mask_slices: List[np.ndarray],      # 每个元素 (H,W) {0,1}
        window_w: int = 1500,
        window_l: int = -600,
        change_mask_size: bool = False,
        out_size: Optional[Tuple[int, int]] = None,
    ):
        assert len(hu_slices) == len(mask_slices)
        self.hu_slices = hu_slices
        self.mask_slices = [(m > 0).astype(np.uint8) for m in mask_slices]
        self.window_w = window_w
        self.window_l = window_l
        self.change_mask_size = change_mask_size
        self.out_size = out_size

    def __len__(self):
        return len(self.hu_slices)

    def _resize_pair(self, img01: np.ndarray, mask01: np.ndarray, hu: np.ndarray, size):
        """
        将图像/掩膜（以及可选 HU）同时 resize 到同一尺寸。
        - 图像/hu 用 LINEAR
        - 掩膜用 NEAREST
        返回： (img_r, mask_r) 或 (img_r, mask_r, hu_r)
        """
        H, W = size
        # 图像（[0,1]）线性插值
        img_r = cv2.resize(img01, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # 掩膜（0/1）最近邻
        mask_r = cv2.resize(mask01.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_r = (mask_r > 0).astype(np.uint8)
        # HU 同样线性插值
        hu_r = cv2.resize(hu.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return img_r, mask_r, hu_r

    def __getitem__(self, idx):
        hu = self.hu_slices[idx].astype(np.float32)
        mask = self.mask_slices[idx].astype(np.uint8)

        # 清晰图（标准窗宽窗位）
        clean01 = apply_window(hu, w=self.window_w, l=self.window_l)
        #
        if self.out_size is not None:
            clean01, mask, hu = self._resize_pair(clean01, mask, hu, self.out_size,)
        #
        # degraded01, mask_out, params = make_degraded(
        #     img01=clean01,
        #     mask01=mask,
        #     hu_img=hu,  # 提供 HU 以便随机窗宽窗位
        #     change_mask_size=self.change_mask_size
        # )
        # mask_out = mask if mask_out is None else mask_out

        sample = {
            "image_clean": torch.from_numpy(clean01).unsqueeze(0).float(),    # (1,H,W)
            # "image_degraded": torch.from_numpy(degraded01).unsqueeze(0).float(),
            # "mask": torch.from_numpy(mask_out.astype(np.float32)).unsqueeze(0),  # (1,H,W)
            # "params": params
        }
        return sample


# -----------------------------
# 示例：把 DICOM 序列 + NIfTI 掩膜 -> 切片列表 -> Dataset
# -----------------------------
def series_to_slices(
    dicom_folder: str,
    mask_nifti_path: Optional[str] = None,
    window_w: int = 1500,
    window_l: int = -600,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    从一个序列文件夹提取所有切片 (HU, mask)
    - 若提供 mask_nifti_path（与 DICOM 对齐的 3D 掩膜），将按同一索引切片
    - 若没有 mask，返回全零掩膜
    """
    vol_hu, _ = load_dicom_series(dicom_folder)     # (D,H,W)
    D, H, W = vol_hu.shape

    if mask_nifti_path is not None:
        mask3d = load_nifti(mask_nifti_path)        # (D,H,W)
        # 保证二值
        mask3d = (mask3d > 0).astype(np.uint8)
        if mask3d.shape != vol_hu.shape:
            raise ValueError(f"Mask shape {mask3d.shape} != volume shape {vol_hu.shape}")
    else:
        mask3d = np.zeros_like(vol_hu, dtype=np.uint8)

    hu_slices = [vol_hu[i] for i in range(D)]
    mask_slices = [mask3d[i] for i in range(D)]
    return hu_slices, mask_slices

class Diffusion:

    def __init__(self,  model_path, device):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(device)

    def __call__(self, prompts, num_inference_steps):
        images = self.pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps
        ).images
        return images


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    """
    示例用法：
      1) 先将 LIDC 一个病人的 DICOM 文件夹与对应 NIfTI 掩膜转成切片列表
      2) 构建 Dataset
      3) 取一个样本，保存成 PNG 观察
    """
    # ======= 修改为你的路径 =======

    ap = argparse.ArgumentParser()
    ap.add_argument("--output",
                    default="/data/LIDC/augmented/ds_ct-%06d.tar")
    ap.add_argument("--shard_maxcount", type=int, default=2000)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--resume", action="store_true",
                    help="启用断点续跑：跳过进度文件中已完成的 key")
    ap.add_argument("--state_file", default="/data/LIDC/augmented/ds_ct.state.jsonl",
                    help="进度文件路径（默认：<output_stem>.state.jsonl）")
    ap.add_argument("--model_path", type=str, required=True)

    args = ap.parse_args()

    REFLECTION_SYSTEM_PROMPT = (
        "You are a medical imaging evaluation assistant. "
        "The left image is a real lung CT, the right image is generated from the diagnostic description. "
        "Using the real CT as the ground truth, evaluate how well the generated image matches the description. "
        "Identify what aspects should be improved to: "
        "1) better match the diagnostic description, and "
        "2) look more realistic as a lung CT. "
        "Focus on anatomical accuracy, realism, and consistency. "
        "Keep the answer concise, no longer than 100 words."
    )

    REFLECTION_USER_PROMPT = (
        "Diagnostic description (from the real image):\n"
        "{diagnostic_description}\n\n"
        "Task:\n"
        "Provide short and clear suggestions to improve the generated image so it better reflects the description and looks more realistic, in under 100 words."
    )

    CT_DESCRIPTION_SYSTEM_PROMPT = (
        "You are a board-certified thoracic radiologist specializing in lung CT interpretation. "
        "You are reviewing a single axial CT slice from the LIDC-IDRI dataset. "
        "This image contains at least one radiologist-annotated pulmonary nodule, whose approximate 3D bounding box is provided. "
        "Your task is to generate a concise, visually grounded radiological description focusing on the nodule’s appearance. "
        "Describe only what is visible: the nodule’s position within the lung, shape, margin definition, internal density (solid, part-solid, or ground-glass), "
        "and relationship to nearby anatomy (bronchi, vessels, pleura). "
        "Do not infer clinical significance or diagnosis. "
        "Keep the language professional, objective, and under 100 words."
    )

    CT_DESCRIPTION_USER_PROMPT = (
        "This axial CT slice of the chest (from the LIDC-IDRI dataset) contains a visible pulmonary nodule. "
        "The annotated nodule region is located within the bounding box coordinates: {bbox_coordinates}. "
        "Describe the nodule in professional radiological terms, including its location within the lung (e.g., right upper lobe, peripheral vs. central), "
        "its size and shape (round, oval, irregular), margin characteristics (smooth, lobulated, spiculated), "
        "and internal attenuation (solid, part-solid, or ground-glass). "
        "Optionally, mention nearby anatomical structures if relevant (e.g., adjacent vessels, pleural surface). "
        "Avoid diagnostic interpretations or non-visual assumptions. Keep the report concise and factual."
    )

    uid = 0
    save_dir = Path("/data/LIDC/merged_jpgs")
    save_dir.mkdir(parents=True, exist_ok=True)

    huatuo_bot = Huatuo()
    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file) if args.state_file else (
        out_dir / (out_path.stem + ".state.jsonl")
    )
    done_keys = load_done_keys(state_path) if args.resume else set()
    scans = pl.query(pl.Scan).all()
    model = Diffusion(args.model_path, "cuda:1")

    with ShardWriter(args.output, maxcount=args.shard_maxcount) as sink:
        for scan in tqdm(scans):
            vol = scan.to_volume().astype(np.float32)  # (D,H,W) HU
            clusters = scan.cluster_annotations()
            for anns in clusters:
                cons_mask, cbbox, _ = consensus(anns, clevel=0.5)
                # 获取这个 annotation 的 bounding box
                x0, x1 = cbbox[0].start, cbbox[0].stop  # inclusive start / stop
                y0, y1 = cbbox[1].start, cbbox[1].stop
                z0, z1 = cbbox[2].start, cbbox[2].stop
                bbox_str = f"(x0={x0}, x1={x1}, y0={y0}, y1={y1})"
                vol_crop = vol[:, :, z0:z1]  # 截取对应体素块
                H, W, Dz = vol_crop.shape
                h2, w2, Dz2 = cons_mask.shape
                mask_full = np.zeros((H, W, Dz), dtype=np.uint8)

                for k in range(Dz):
                    mask_full[x0: x0 + h2, y0: y0 + w2, k] = (cons_mask[:, :, k] > 0).astype(np.uint8)

                non_empty_indices = [i for i in range(Dz) if mask_full[:, :, i].any()]
                hu_slices = [vol_crop[:, :, i] for i in non_empty_indices]
                mask_slices = [mask_full[:, :, i] for i in non_empty_indices]
                print(hu_slices[0].shape, mask_slices[0].shape)

                # 2) 构建 Dataset
                dataset = LIDCSlicePairDataset(
                    hu_slices=hu_slices,
                    mask_slices=mask_slices,
                    window_w=1500, window_l=-600,
                    change_mask_size=False,             # 如需随机改变病灶大小可改为 True
                    out_size=(512, 512),                # 统一输出大小
                )

                for data in dataset:
                    key = f"{uid:09d}"
                    if args.resume and key in done_keys:
                        uid += 1
                        continue
                    ct_description_prompt = (CT_DESCRIPTION_SYSTEM_PROMPT + "\n" + CT_DESCRIPTION_USER_PROMPT).format(bbox_coordinates=bbox_str)
                    real_img01 = data["image_clean"].numpy().squeeze(0)  # [0,1], HxW
                    real_img8 = (real_img01 * 255).clip(0, 255).astype(np.uint8)
                    # degraded_img01 = data["image_degraded"].numpy().squeeze(0)
                    # degraded_img8 = (degraded_img01 * 255).clip(0, 255).astype(np.uint8)
                    diagnostic_description = huatuo_bot.inference(ct_description_prompt, Image.fromarray(real_img8))
                    real_img = Image.fromarray(real_img8)
                    degraded_img = model(prompts=[diagnostic_description], num_inference_steps=50)[0]
                    merged_pil = make_side_by_side(real_img, degraded_img)
                    merged_path = save_dir / f"{key}.jpg"
                    merged_pil.save(merged_path, quality=95)
                    reflection_prompt = (REFLECTION_SYSTEM_PROMPT + "\n" + REFLECTION_USER_PROMPT).format(
                        diagnostic_description=diagnostic_description
                    )
                    reflection = huatuo_bot.inference(reflection_prompt, merged_pil)
                    print(f"Description: {diagnostic_description}")
                    print(f"Reflection: {reflection}")
                    sample = {
                        "__key__": key,  # 复用原key
                        "real.png": pil_to_png_bytes(real_img),  # 原样保存
                        "gen.png": pil_to_png_bytes(degraded_img),  # 原样保存
                        "prompt.txt": diagnostic_description.encode("utf-8"),  # 原样保存
                        "merged.jpg": merged_pil,  # 方便复核
                        "huatuo.json": reflection.encode("utf-8"),
                    }
                    sink.write(sample)
                    append_done_key(state_path, key)
                    uid += 1




