from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ==================== 配置区 (直接在这里修改) ====================

GT_DIR = ""    
GEN_DIR = ""      

N_SAMPLES = 1000
RANDOM_SEED = 42             

DEVICE = "cuda"              
BATCH_SIZE = 32
NUM_WORKERS = 4

# KID 设置: 重复抽样子集数量与每个子集大小
KID_SUBSETS = 50
KID_SUBSET_SIZE = 500        

# InceptionV3 权重路径 (离线时使用; 联网时留 None 自动下载)
INCEPTION_WEIGHTS = None     # 例如: "./inception_v3.pth"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ================================================================


class ImageFolder(Dataset):
    def __init__(self, folder, n_samples=None, seed=42):
        paths = sorted(p for p in Path(folder).rglob("*")
                       if p.suffix.lower() in IMG_EXTS)
        assert len(paths) > 0, f"文件夹内没有图像: {folder}"
        # 固定数量抽样
        if n_samples is not None and len(paths) > n_samples:
            paths = random.Random(seed).sample(paths, n_samples)
        self.paths = paths
        print(f"[{folder}] 使用 {len(self.paths)} 张图像")
        self.tf = transforms.Compose([
            transforms.Resize((299, 299),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


def get_inception(device):
    """InceptionV3 特征提取器 (输出 2048 维 pool 特征)"""
    if INCEPTION_WEIGHTS:
        m = models.inception_v3(transform_input=False)
        m.load_state_dict(torch.load(INCEPTION_WEIGHTS, map_location="cpu"))
    else:
        m = models.inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    m.fc = nn.Identity()     # 去掉分类层, 直接输出 2048 维特征
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def extract_features(folder, model, device):
    dataset = ImageFolder(folder, n_samples=N_SAMPLES, seed=RANDOM_SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    feats = []
    for batch in loader:
        f = model(batch.to(device))            # [B, 2048]
        feats.append(f.reshape(f.size(0), -1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def sqrtm_psd(m):
    """对称半正定矩阵的平方根"""
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def compute_fid(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    cov1 = np.cov(f1, rowvar=False)
    cov2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    s1 = sqrtm_psd(cov1)
    mid = sqrtm_psd(s1 @ cov2 @ s1)   # 注意: 这里是对称矩阵, eigh 合法
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2)
                 - 2 * np.trace(mid))


def poly_kernel(X, Y):
    """多项式核 k(x,y) = (x·y/d + 1)^3, Inception 特征标准做法"""
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** 3


def compute_kid(f1, f2, n_subsets, subset_size, seed=42):
    rng = np.random.default_rng(seed)
    n = min(subset_size, len(f1), len(f2))
    scores = []
    for _ in range(n_subsets):
        X = f1[rng.choice(len(f1), n, replace=False)]
        Y = f2[rng.choice(len(f2), n, replace=False)]
        K_XX, K_YY, K_XY = poly_kernel(X, X), poly_kernel(Y, Y), poly_kernel(X, Y)
        # 无偏 MMD^2: 去掉对角线
        mmd = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) \
            + (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1)) \
            - 2 * K_XY.mean()
        scores.append(mmd)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if N_SAMPLES is not None:
        print(f"固定抽样: 每组 {N_SAMPLES} 张 (seed={RANDOM_SEED})")
    model = get_inception(device)

    f_gt = extract_features(GT_DIR, model, device)
    f_gen = extract_features(GEN_DIR, model, device)

    fid_score = compute_fid(f_gt, f_gen)
    kid_mean, kid_std = compute_kid(f_gt, f_gen, KID_SUBSETS, KID_SUBSET_SIZE)

    print("\n========== 评估结果 ==========")
    print(f"FID : {fid_score:.4f}  (越低越好)")
    print(f"KID : {kid_mean * 100:.4f} +/- {kid_std * 100:.4f}  (x10^-2, 越低越好)")
    print("==============================\n")

    out = Path(GEN_DIR) / "fid_kid_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"GT  : {GT_DIR}\nGen : {GEN_DIR}\n")
        f.write(f"N   : {N_SAMPLES} (seed={RANDOM_SEED})\n")
        f.write(f"FID : {fid_score:.4f}\n")
        f.write(f"KID : {kid_mean * 100:.4f} +/- {kid_std * 100:.4f} (x10^-2)\n")
    print(f"结果已保存到: {out}")


if __name__ == "__main__":
    main()


import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from pytorch_msssim import ms_ssim
import torch
import re
# ============================================================
# Configuration
# ============================================================
REFERENCE_DIR = ""
GENERATED_DIR = ""
OUTPUT_FILE = ""
GEN_START = 0    
GEN_END = 1000      
IMAGE_SIZE = (512, 512)
# ============================================================
# Image loading
# ============================================================
def load_image(path):
    img = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_to_512(image):
    target_h, target_w = IMAGE_SIZE
    h, w = image.shape[:2]
    scale = min(target_w/w, target_h/h)
    nw, nh = int(w*scale), int(h*scale)
    img_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    top = (target_h - nh)//2
    left = (target_w - nw)//2
    canvas[top:top+nh, left:left+nw] = img_resized
    return canvas
# ============================================================
# SSIM
# ============================================================
def calculate_ssim(reference, generated):
    ref_gray = cv2.cvtColor(
        (reference * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )
    gen_gray = cv2.cvtColor(
        (generated * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )
    score = structural_similarity(
        ref_gray,
        gen_gray,
        data_range=255
    )
    return float(score)
# ============================================================
# MS-SSIM
# ============================================================
def calculate_ms_ssim(reference, generated):
    ref_tensor = torch.from_numpy(
        reference.transpose(2, 0, 1)
    ).unsqueeze(0).float()
    gen_tensor = torch.from_numpy(
        generated.transpose(2, 0, 1)
    ).unsqueeze(0).float()
    score = ms_ssim(
        ref_tensor,
        gen_tensor,
        data_range=1.0,
        size_average=True
    )
    return float(score.item())
# ============================================================
# IoU
# ============================================================
def create_foreground_mask(image):
    img_uint8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(
        img_uint8,
        cv2.COLOR_RGB2GRAY
    )
    mask = gray < 245
    return mask
def calculate_iou(reference, generated):
    ref_mask = create_foreground_mask(reference)
    gen_mask = create_foreground_mask(generated)
    intersection = np.logical_and(
        ref_mask,
        gen_mask
    ).sum()
    union = np.logical_or(
        ref_mask,
        gen_mask
    ).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
# ============================================================
# Evaluate one image pair
# ============================================================
def evaluate_pair(reference_path, generated_path):
    reference = load_image(reference_path)
    generated = load_image(generated_path)
    # 两张图都等比例缩放，居中放到512×512白画布
    reference = resize_to_512(reference)
    generated = resize_to_512(generated)
    reference = reference.astype(np.float32) / 255.0
    generated = generated.astype(np.float32) / 255.0
    ssim_score = calculate_ssim(reference, generated)
    ms_ssim_score = calculate_ms_ssim(reference, generated)
    iou_score = calculate_iou(reference, generated)
    return {
        "ref_image": os.path.basename(reference_path),
        "gen_image": os.path.basename(generated_path),
        "SSIM": ssim_score,
        "MS-SSIM": ms_ssim_score,
        "IoU": iou_score
    }
# ============================================================
# Main
# ============================================================
def numeric_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return tuple(int(n) for n in nums)

def main():
    results = []
    ref_filenames = sorted(
        (f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))),
        key=numeric_sort_key
    )
    gen_all = sorted(
        (f for f in os.listdir(GENERATED_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))),
        key=numeric_sort_key
    )
    gen_filenames = gen_all[GEN_START:GEN_END]
    print(f"Reference共{len(ref_filenames)}张全部使用 | Generated选取范围：第{GEN_START+1}~{GEN_END}张")
    for ref_name, gen_name in zip(ref_filenames, gen_filenames):
        ref_path = os.path.join(REFERENCE_DIR, ref_name)
        gen_path = os.path.join(GENERATED_DIR, gen_name)
        result = evaluate_pair(ref_path, gen_path)
        results.append(result)

    if not results:
        print("No pairs.")
        return
    df = pd.DataFrame(results)
    mean_row = {
        "ref_image": "Mean",
        "gen_image": "",
        "SSIM": df["SSIM"].mean(),
        "MS-SSIM": df["MS-SSIM"].mean(),
        "IoU": df["IoU"].mean()
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print("\n==============================")
    print("Average Results")
    print("==============================")
    print(f"SSIM     : {mean_row['SSIM']:.4f}")
    print(f"MS-SSIM  : {mean_row['MS-SSIM']:.4f}")
    print(f"IoU      : {mean_row['IoU']:.4f}")
    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()