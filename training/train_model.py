"""
train_model.py
===============
Tahap training model SegFormer 7-band untuk segmentasi tutupan lahan.

Isi file ini:
  - LandCover7BandDataset : Dataset PyTorch untuk patch .npy (7-band) + mask .npy
  - SegFormer7Band        : Wrapper SegFormer (HuggingFace) dengan input
                             conv pertama diperluas dari 3 channel (RGB) ke
                             Config.NUM_CHANNELS (7-band), bobot RGB asli
                             dipertahankan, channel baru diinisialisasi dari
                             rata-rata bobot RGB
  - FocalLoss / DiceLoss / FocalDiceLoss : loss function untuk class imbalance
                             (Focal & CE memakai class weights)
  - SegmentationMetrics   : akumulasi confusion matrix per epoch -> IoU per
                             kelas, mIoU, frequency-weighted IoU (FWIoU),
                             pixel accuracy
  - SegFormerTrainer      : training loop lengkap (AMP, warmup+cosine LR,
                             checkpointing, early stopping, plot history)

Class weights dihitung OTOMATIS dari mask split train bila
Config.CLASS_WEIGHTS = "auto" (lihat data_qc/calculate_class_weights.py).

SegFormer7Band diimpor ulang oleh inference/run_inference.py untuk memuat
arsitektur yang sama persis saat inference.

Usage:
    python training/train_model.py
"""

import re
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from data_qc.calculate_class_weights import resolve_class_weights
# Versi AMD/ROCm: pemilihan GPU, AMP (bf16/fp16) & GradScaler terpusat di
# gpu_utils.py (torch.amp, PyTorch >= 2.4 — fallback torch.cuda.amp dihapus).
from gpu_utils import (setup_gpu_env, get_device, device_label,
                       amp_autocast, make_grad_scaler)

setup_gpu_env()


# ============================================================
# DATASET
# ============================================================

class LandCover7BandDataset(Dataset):
    """Dataset PyTorch: load patch (H, W, 7) .npy + mask (H, W) .npy,
    lalu terapkan augmentasi albumentations (jika ada)."""

    def __init__(self, patch_paths: list, mask_paths: list, transform=None):
        self.patch_paths = patch_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch = np.load(self.patch_paths[idx]).astype(np.float32)
        mask  = np.load(self.mask_paths[idx])

        if self.transform:
            transformed = self.transform(image=patch, mask=mask)
            patch = transformed['image']
            mask  = transformed['mask'].long()
        else:
            patch = torch.from_numpy(patch).permute(2, 0, 1).float()
            mask  = torch.from_numpy(mask.astype(np.int64))

        return patch, mask


def get_training_augmentation(config=Config):
    """Susun pipeline augmentasi training dari flag-flag di config.py."""
    transforms = []
    if config.USE_AUGMENTATION:
        if config.AUG_HORIZONTAL_FLIP > 0:
            transforms.append(A.HorizontalFlip(p=config.AUG_HORIZONTAL_FLIP))
        if config.AUG_VERTICAL_FLIP > 0:
            transforms.append(A.VerticalFlip(p=config.AUG_VERTICAL_FLIP))
        if config.AUG_ROTATION > 0:
            # area kosong hasil rotasi: image diisi 0, mask diisi IGNORE_INDEX (bukan
            # kelas 0/Background) supaya tidak ikut dihitung di loss & metrik
            transforms.append(A.Rotate(limit=config.AUG_ROTATION_LIMIT, p=config.AUG_ROTATION, border_mode=0,
                                       fill=0, fill_mask=config.IGNORE_INDEX))
        if config.AUG_BRIGHTNESS > 0 or config.AUG_CONTRAST > 0:
            transforms.append(A.RandomBrightnessContrast(brightness_limit=config.AUG_BRIGHTNESS, contrast_limit=config.AUG_CONTRAST, p=0.5))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_validation_augmentation():
    """Validation tidak diaugmentasi, hanya dikonversi ke tensor."""
    return A.Compose([ToTensorV2()])


# ============================================================
# MODEL
# ============================================================

def _first_patch_embedding(segformer_model):
    """Ambil modul patch-embedding stage pertama dari SegformerModel.

    Struktur berbeda antar versi transformers:
      - transformers < 5.x  : segformer.encoder.patch_embeddings[0]
      - transformers >= 5.x : segformer.stages[0].patch_embeddings
    Modul yang dikembalikan selalu punya atribut `.proj` (Conv2d input).
    """
    encoder = getattr(segformer_model, "encoder", None)
    if encoder is not None and hasattr(encoder, "patch_embeddings"):
        return encoder.patch_embeddings[0]
    stages = getattr(segformer_model, "stages", None)
    if stages is not None and hasattr(stages[0], "patch_embeddings"):
        return stages[0].patch_embeddings
    raise AttributeError(
        "Tidak menemukan patch embedding pertama SegFormer — struktur "
        "transformers tidak dikenali. Cek versi transformers yang terinstal."
    )


# Nama layer SegFormer: transformers < 5 (kiri) -> transformers >= 5 (kanan).
# Sama dengan aturan WeightRenaming di transformers/conversion_mapping.py, yang
# hanya dipakai otomatis oleh from_pretrained(), TIDAK oleh load_state_dict().
_SEGFORMER_RENAMES = [
    (r"segformer\.encoder\.patch_embeddings\.(\d+)\.", r"segformer.stages.\1.patch_embeddings."),
    (r"segformer\.encoder\.block\.(\d+)\.",            r"segformer.stages.\1.blocks."),
    (r"segformer\.encoder\.layer_norm\.(\d+)\.",       r"segformer.stages.\1.layer_norm."),
    (r"\.attention\.self\.query\.",                    r".attention.q_proj."),
    (r"\.attention\.self\.key\.",                      r".attention.k_proj."),
    (r"\.attention\.self\.value\.",                    r".attention.v_proj."),
    (r"\.attention\.self\.sr\.",                       r".attention.sequence_reduction.sequence_reduction."),
    (r"\.attention\.self\.layer_norm\.",               r".attention.sequence_reduction.layer_norm."),
    (r"\.attention\.output\.dense\.",                  r".attention.o_proj."),
    (r"\.mlp\.dense1\.",                               r".mlp.fc1."),
    (r"\.mlp\.dense2\.",                               r".mlp.fc2."),
    (r"\.layer_norm_1\.",                              r".layernorm_before."),
    (r"\.layer_norm_2\.",                              r".layernorm_after."),
    (r"decode_head\.linear_c\.",                       r"decode_head.linear_projections."),
]
_SEGFORMER_RENAMES_BACK = [
    (r"segformer\.stages\.(\d+)\.patch_embeddings\.",  r"segformer.encoder.patch_embeddings.\1."),
    (r"segformer\.stages\.(\d+)\.blocks\.",            r"segformer.encoder.block.\1."),
    (r"segformer\.stages\.(\d+)\.layer_norm\.",        r"segformer.encoder.layer_norm.\1."),
    (r"\.attention\.q_proj\.",                         r".attention.self.query."),
    (r"\.attention\.k_proj\.",                         r".attention.self.key."),
    (r"\.attention\.v_proj\.",                         r".attention.self.value."),
    (r"\.attention\.sequence_reduction\.sequence_reduction\.", r".attention.self.sr."),
    (r"\.attention\.sequence_reduction\.layer_norm\.", r".attention.self.layer_norm."),
    (r"\.attention\.o_proj\.",                         r".attention.output.dense."),
    (r"\.mlp\.fc1\.",                                  r".mlp.dense1."),
    (r"\.mlp\.fc2\.",                                  r".mlp.dense2."),
    (r"\.layernorm_before\.",                          r".layer_norm_1."),
    (r"\.layernorm_after\.",                           r".layer_norm_2."),
    (r"decode_head\.linear_projections\.",             r"decode_head.linear_c."),
]


def _rename_key(key, rules):
    for pat, rep in rules:
        key = re.sub(pat, rep, key)
    return key


def adapt_segformer_state_dict(state_dict, model_keys):
    """Sesuaikan nama key checkpoint dengan versi transformers yang terinstal.

    Checkpoint dari transformers < 5 memakai nama layer lama (encoder.block.*,
    attention.self.query, ...), transformers >= 5 memakai nama baru
    (stages.*.blocks.*, attention.q_proj, ...). Bobotnya identik, hanya namanya
    yang berbeda. Diterjemahkan ke arah yang dibutuhkan model saat ini; kalau
    sudah cocok, state_dict dikembalikan apa adanya.
    """
    model_keys = set(model_keys)
    ckpt_keys = set(state_dict)
    if ckpt_keys == model_keys:
        return state_dict
    for rules in (_SEGFORMER_RENAMES, _SEGFORMER_RENAMES_BACK):
        renamed = {_rename_key(k, rules): v for k, v in state_dict.items()}
        if len(set(renamed) & model_keys) > len(ckpt_keys & model_keys):
            return renamed
    return state_dict


class SegFormer7Band(nn.Module):
    """SegFormer (HuggingFace) dengan patch-embedding pertama diperluas dari
    3 channel (RGB pretrained) ke NUM_CHANNELS (7-band)."""

    def __init__(self, config=Config, pretrained=None):
        """pretrained=None -> ikut Config.PRETRAINED. Inference memakai
        pretrained=False: arsitektur dibangun dari config HF saja (tanpa
        download bobot ImageNet/ADE), karena semua bobot langsung ditimpa
        checkpoint hasil training."""
        super().__init__()
        self.cfg = config
        if pretrained is None:
            pretrained = config.PRETRAINED
        # id2label eksplisit (bukan num_labels=...) supaya transformers tidak
        # memperingatkan bentrok dengan 150 label ADE20K bawaan checkpoint
        id2label = {i: config.CLASS_NAMES.get(i, f"class_{i}") for i in range(config.NUM_CLASSES)}
        label2id = {v: k for k, v in id2label.items()}
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                config.MODEL_NAME,
                id2label=id2label, label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            hf_config = SegformerConfig.from_pretrained(config.MODEL_NAME, id2label=id2label, label2id=label2id)
            self.model = SegformerForSemanticSegmentation(hf_config)
        first_embed = _first_patch_embedding(self.model.segformer)
        orig_conv = first_embed.proj
        new_conv = nn.Conv2d(
            in_channels  = config.NUM_CHANNELS,
            out_channels = orig_conv.out_channels,
            kernel_size  = orig_conv.kernel_size,
            stride       = orig_conv.stride,
            padding      = orig_conv.padding,
            bias         = orig_conv.bias is not None,
        )
        with torch.no_grad():
            # 3 channel pertama (R, G, B) pakai bobot pretrained asli
            new_conv.weight[:, :3, :, :] = orig_conv.weight.clone()
            # channel tambahan (ExG, VARI, Edge, Texture) diinisialisasi
            # dari rata-rata bobot RGB, bukan random, supaya training lebih stabil
            avg_weight = orig_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, config.NUM_CHANNELS):
                new_conv.weight[:, i:i+1, :, :] = avg_weight.clone()
            if orig_conv.bias is not None:
                new_conv.bias.copy_(orig_conv.bias)
        first_embed.proj = new_conv
        self.model.config.num_channels = config.NUM_CHANNELS
        if config.FREEZE_ENCODER:
            # self.model.segformer = encoder SegFormer (tanpa decode head)
            for param in self.model.segformer.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # SegFormer output resolusinya lebih rendah dari input (stride decoder)
        # -> upsample kembali ke ukuran input supaya sama dengan mask
        logits = nn.functional.interpolate(outputs.logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss — turunkan bobot sample yang sudah mudah diklasifikasi,
    fokus ke sample sulit (berguna untuk class imbalance).

    Bila `class_weights` diberikan, loss tiap pixel dikali bobot kelas
    ground-truth-nya, lalu dirata-rata dengan normalisasi jumlah bobot
    (sama seperti CrossEntropyLoss(weight=...)). Pixel IGNORE_INDEX tidak
    ikut dihitung di pembilang maupun penyebut."""

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        # register_buffer -> ikut pindah device saat .to(device)
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs.float(), targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        valid = targets != self.ignore_index
        if self.class_weights is not None:
            safe_t = torch.where(valid, targets, torch.zeros_like(targets))
            w = self.class_weights[safe_t] * valid.float()
            return (focal_loss * w).sum() / w.sum().clamp_min(1e-12)
        n_valid = valid.sum()
        if n_valid == 0:
            return focal_loss.sum() * 0.0
        return focal_loss.sum() / n_valid


class DiceLoss(nn.Module):
    """Dice Loss — optimasi langsung terhadap overlap (IoU-like), efektif
    untuk kelas dengan area kecil di citra."""

    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(inputs)
        valid_mask = targets != self.ignore_index
        targets_valid = targets.clone()
        targets_valid[~valid_mask] = 0
        targets_one_hot.scatter_(1, targets_valid.unsqueeze(1), 1)
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(inputs)
        inputs = inputs * valid_mask_expanded.float()
        targets_one_hot = targets_one_hot * valid_mask_expanded.float()
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """Kombinasi Focal + Dice Loss, dibobot via focal_weight/dice_weight."""

    def __init__(self, focal_weight=0.5, dice_weight=0.5, focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index, class_weights=class_weights)
        self.dice = DiceLoss(smooth=dice_smooth, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + self.dice_weight * self.dice(inputs, targets)


def get_loss_function(config=Config, class_weights=None):
    """Bangun loss function sesuai config.LOSS_FUNCTION.

    `class_weights` = list hasil resolve_class_weights() (atau None). Dipakai
    oleh crossentropy, focal, dan komponen focal dari focal_dice."""
    if config.LOSS_FUNCTION == "crossentropy":
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weights, ignore_index=config.IGNORE_INDEX)
        return nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
    elif config.LOSS_FUNCTION == "focal":
        return FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, ignore_index=config.IGNORE_INDEX, class_weights=class_weights)
    elif config.LOSS_FUNCTION == "dice":
        return DiceLoss(smooth=config.DICE_SMOOTH, ignore_index=config.IGNORE_INDEX)
    elif config.LOSS_FUNCTION == "focal_dice":
        return FocalDiceLoss(
            focal_weight=config.FOCAL_WEIGHT,
            dice_weight=config.DICE_WEIGHT,
            focal_alpha=config.FOCAL_ALPHA,
            focal_gamma=config.FOCAL_GAMMA,
            dice_smooth=config.DICE_SMOOTH,
            ignore_index=config.IGNORE_INDEX,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown loss function: {config.LOSS_FUNCTION}")


# ============================================================
# METRICS
# ============================================================

class SegmentationMetrics:
    """Akumulasi confusion matrix sepanjang SATU epoch, lalu hitung metrik.

    Kenapa tidak rata-rata per batch seperti sebelumnya?
      - IoU per batch lalu dirata-rata != IoU seluruh dataset (batch kecil
        dengan sedikit pixel ikut berbobot sama dengan batch besar).
      - Kelas yang tidak muncul di batch dulu dihitung IoU = 0, sehingga
        mIoU jatuh jauh di bawah nilai sebenarnya.

    Di sini semua pixel valid (bukan IGNORE_INDEX) dari seluruh epoch
    dikumpulkan ke confusion matrix `cm[gt, pred]`, lalu:
      IoU_c   = TP_c / (TP_c + FP_c + FN_c)       (NaN bila kelas tidak ada
                                                    di GT maupun prediksi)
      mIoU    = rata-rata IoU_c untuk kelas yang ada (NaN diabaikan)
      FWIoU   = sum_c (n_gt_c / N) * IoU_c         (frequency-weighted IoU)
      OA      = sum_c TP_c / N                     (pixel accuracy)
      Acc_c   = TP_c / n_gt_c                      (recall / producer's acc.)
    """

    def __init__(self, num_classes, ignore_index=255, device="cpu"):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=device)

    def reset(self):
        self.cm.zero_()

    @torch.no_grad()
    def update(self, preds, targets):
        """preds, targets: tensor (B, H, W) berisi class id."""
        preds = preds.reshape(-1).to(self.cm.device)
        targets = targets.reshape(-1).to(self.cm.device)
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        idx = targets[valid].long() * self.num_classes + preds[valid].long()
        self.cm += torch.bincount(idx, minlength=self.num_classes ** 2)

    def confusion_matrix(self):
        return self.cm.view(self.num_classes, self.num_classes).cpu().numpy().astype(np.int64)

    def compute(self):
        cm = self.confusion_matrix().astype(np.float64)
        tp = np.diag(cm)
        gt = cm.sum(axis=1)      # jumlah pixel ground truth per kelas
        pred = cm.sum(axis=0)    # jumlah pixel prediksi per kelas
        union = gt + pred - tp
        total = gt.sum()

        with np.errstate(divide='ignore', invalid='ignore'):
            class_iou = np.where(union > 0, tp / union, np.nan)
            class_acc = np.where(gt > 0, tp / gt, np.nan)

        present = ~np.isnan(class_iou)
        miou = float(np.mean(class_iou[present])) if present.any() else 0.0
        freq = gt / total if total > 0 else np.zeros_like(gt)
        fwiou = float(np.sum(freq * np.nan_to_num(class_iou, nan=0.0)))
        overall_acc = float(tp.sum() / total) if total > 0 else 0.0

        return {
            'miou': miou,
            'fwiou': fwiou,
            'acc': overall_acc,
            'class_iou': [float(x) for x in class_iou],
            'class_acc': [float(x) for x in class_acc],
            'class_freq': [float(x) for x in freq],
        }


def calculate_metrics(predictions, targets, num_classes, ignore_index=255):
    """Kompatibilitas lama: hitung metrik untuk SATU kumpulan prediksi.
    Return (mean_iou, overall_acc, class_iou, class_acc). Untuk metrik per
    epoch gunakan SegmentationMetrics (akumulasi confusion matrix)."""
    m = SegmentationMetrics(num_classes, ignore_index, device=predictions.device)
    m.update(predictions, targets)
    r = m.compute()
    return r['miou'], r['acc'], r['class_iou'], r['class_acc']


def _json_safe(obj):
    """Ganti NaN/Inf dengan None supaya JSON valid (dibaca sebagai null)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _fmt(x):
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


# ============================================================
# LR SCHEDULER DENGAN WARMUP
# ============================================================

class WarmupScheduler:
    """Bungkus scheduler dasar (cosine/polynomial/step) dengan linear warmup
    di awal training, supaya LR tidak melonjak tiba-tiba dari 0."""

    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            alpha = self.current_epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg['lr'] = self.warmup_lr + alpha * (base_lr - self.warmup_lr)
        else:
            self.base_scheduler.step()

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {'current_epoch': self.current_epoch, 'base_scheduler': self.base_scheduler.state_dict()}

    def load_state_dict(self, state):
        self.current_epoch = state['current_epoch']
        self.base_scheduler.load_state_dict(state['base_scheduler'])


# ============================================================
# DATA LOADING
# ============================================================

def load_patch_mask_paths(config=Config):
    """Cari semua pasangan patch/mask .npy yang valid (nama file sama persis)."""
    patches_dir = config.PATCHES_DIR
    masks_dir = config.MASKS_DIR

    if not patches_dir.exists():
        raise FileNotFoundError(f"PATCHES_DIR not found: {patches_dir}")

    all_patch_files = sorted(patches_dir.rglob("*.npy"))

    if not all_patch_files:
        raise FileNotFoundError(f"No .npy files in {patches_dir}")

    patch_paths = []
    mask_paths = []

    for patch_f in all_patch_files:
        relative = patch_f.relative_to(patches_dir)
        mask_f = masks_dir / relative
        if mask_f.exists():
            patch_paths.append(patch_f)
            mask_paths.append(mask_f)

    return patch_paths, mask_paths


def build_dataloaders(config=Config):
    """Split train/val (random_state = Config.RANDOM_SEED, dipakai juga oleh
    data_qc/analyze_class_distribution.py agar split identik) lalu bungkus DataLoader."""
    patch_paths, mask_paths = load_patch_mask_paths(config)

    train_p, val_p, train_m, val_m = train_test_split(
        patch_paths, mask_paths,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    train_ds = LandCover7BandDataset(train_p, train_m, transform=get_training_augmentation(config))
    val_ds = LandCover7BandDataset(val_p, val_m, transform=get_validation_augmentation())

    # persistent_workers: worker tidak dibuat ulang tiap epoch (di Windows
    # spawn worker lambat karena harus import torch dari nol setiap kali)
    persistent = config.NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=False, persistent_workers=persistent)

    return train_loader, val_loader, len(train_ds), len(val_ds), train_m


# ============================================================
# TRAINER
# ============================================================

class SegFormerTrainer:
    """Orkestrasi training loop lengkap: forward/backward per-epoch, AMP,
    validasi, checkpointing (periodik + best), early stopping, plot history."""

    def __init__(self, config=Config, class_weights=None):
        self.config = config
        self.device = get_device(config)

        self.class_weights = class_weights
        self.model = SegFormer7Band(config).to(self.device)
        self.criterion = get_loss_function(config, class_weights=class_weights).to(self.device)
        self.best_metric_name = getattr(config, 'BEST_MODEL_METRIC', 'miou')

        if config.OPTIMIZER == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS, eps=config.EPS)
        else:
            raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")

        if config.LR_SCHEDULER == "polynomial":
            base_sched = optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=config.NUM_EPOCHS, power=config.LR_POWER)
        elif config.LR_SCHEDULER == "cosine":
            base_sched = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.NUM_EPOCHS)
        else:
            base_sched = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        self.scheduler = WarmupScheduler(optimizer=self.optimizer, warmup_epochs=config.WARMUP_EPOCHS, warmup_lr=config.WARMUP_LR, base_scheduler=base_sched)

        # bf16 (default RDNA4) -> scaler None, fp16 -> GradScaler, CPU/AMP off -> None
        self.scaler = make_grad_scaler(self.device, config)

        # train_iou / val_iou = mIoU, train_fwiou / val_fwiou = frequency-weighted IoU
        self.history = {
            'train_loss': [], 'train_iou': [], 'train_fwiou': [], 'train_acc': [],
            'val_loss': [], 'val_iou': [], 'val_fwiou': [], 'val_acc': [],
            'lr': [],
            'train_class_iou': [[] for _ in range(config.NUM_CLASSES)],
            'train_class_acc': [[] for _ in range(config.NUM_CLASSES)],
            'val_class_iou': [[] for _ in range(config.NUM_CLASSES)],
            'val_class_acc': [[] for _ in range(config.NUM_CLASSES)],
        }
        self.best_val_iou = 0.0          # nilai terbaik dari BEST_MODEL_METRIC
        self.best_metrics = None         # ringkasan metrik val di epoch terbaik
        self.patience_counter = 0
        self.start_epoch = 0

    def _train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = len(loader)
        metrics = SegmentationMetrics(self.config.NUM_CLASSES, self.config.IGNORE_INDEX, device=self.device)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]", leave=False)

        for step, (images, masks) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with amp_autocast(self.device, self.config):
                logits = self.model(images)
                loss = self.criterion(logits.float(), masks)

            if self.scaler is not None:          # float16 AMP
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:                                # bfloat16 AMP / fp32
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            metrics.update(logits.detach().argmax(dim=1), masks)
            total_loss += loss.item()

            if step % max(self.config.LOG_FREQUENCY, 1) == 0 or step == n_batches - 1:
                r = metrics.compute()  # running (kumulatif sejak awal epoch)
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'mIoU': f"{r['miou']:.4f}",
                                  'fwIoU': f"{r['fwiou']:.4f}", 'acc': f"{r['acc']:.4f}"})

        r = metrics.compute()
        r['loss'] = total_loss / max(n_batches, 1)
        r['confusion_matrix'] = metrics.confusion_matrix().tolist()
        return r

    def _validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = len(loader)
        metrics = SegmentationMetrics(self.config.NUM_CLASSES, self.config.IGNORE_INDEX, device=self.device)

        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Validating", leave=False):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with amp_autocast(self.device, self.config):
                    logits = self.model(images)
                logits = logits.float()
                loss = self.criterion(logits, masks)
                metrics.update(logits.argmax(dim=1), masks)
                total_loss += loss.item()

        r = metrics.compute()
        r['loss'] = total_loss / max(n_batches, 1)
        r['confusion_matrix'] = metrics.confusion_matrix().tolist()
        return r

    def _print_class_table(self, train_r, val_r):
        """Tabel IoU per kelas + mIoU / FWIoU / OA (train vs val)."""
        names = self.config.CLASS_NAMES
        print(f"    {'Kelas':<22}{'Freq(val)':>10}{'IoU train':>11}{'IoU val':>10}{'Acc val':>10}")
        for i in range(self.config.NUM_CLASSES):
            name = f"{i} {names.get(i, f'Class {i}')}"
            print(f"    {name:<22}{val_r['class_freq'][i]*100:>9.2f}%"
                  f"{_fmt(train_r['class_iou'][i]):>11}{_fmt(val_r['class_iou'][i]):>10}"
                  f"{_fmt(val_r['class_acc'][i]):>10}")
        print(f"    {'mIoU':<32}{_fmt(train_r['miou']):>11}{_fmt(val_r['miou']):>10}")
        print(f"    {'FWIoU (weighted)':<32}{_fmt(train_r['fwiou']):>11}{_fmt(val_r['fwiou']):>10}")
        print(f"    {'Overall Acc':<32}{_fmt(train_r['acc']):>11}{_fmt(val_r['acc']):>10}")

    def _save_checkpoint(self, epoch, is_best=False):
        self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_iou': self.best_val_iou,
            'best_metric_name': self.best_metric_name,
            'best_metrics': self.best_metrics,
            'class_weights': self.class_weights,
            'history': self.history,
        }

        if (epoch + 1) % self.config.SAVE_FREQUENCY == 0:
            ckpt_path = self.config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1:04d}.pth"
            torch.save(ckpt, ckpt_path)
            # Batasi jumlah checkpoint tersimpan (hapus yang paling lama)
            all_ckpts = sorted(self.config.CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth"))
            while len(all_ckpts) > self.config.MAX_CHECKPOINTS:
                all_ckpts[0].unlink()
                all_ckpts = all_ckpts[1:]

        if is_best:
            best_path = self.config.MODEL_DIR / "best_model.pth"
            torch.save(ckpt, best_path)

    def load_checkpoint(self, ckpt_path):
        """Resume training dari checkpoint (model, optimizer, scheduler, scaler, history)."""
        # weights_only=False: checkpoint berisi history/metrics (dict Python),
        # PyTorch >= 2.6 default weights_only=True bisa menolak file ini.
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(adapt_segformer_state_dict(ckpt['model_state_dict'], self.model.state_dict().keys()))
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.scaler and ckpt.get('scaler_state_dict'):
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_iou = ckpt['best_val_iou']
        self.best_metrics = ckpt.get('best_metrics')
        self.history = ckpt.get('history', self.history)
        # checkpoint lama (sebelum ada FWIoU) -> isi key yang belum ada
        n_done = len(self.history.get('train_loss', []))
        for key in ('train_fwiou', 'val_fwiou'):
            self.history.setdefault(key, [float('nan')] * n_done)
        if ckpt.get('best_metric_name', 'miou') != self.best_metric_name:
            print(f"⚠️  Checkpoint memilih best model berdasarkan {ckpt.get('best_metric_name', 'miou')}, "
                  f"sekarang {self.best_metric_name} -> best score direset")
            self.best_val_iou = 0.0
        self.patience_counter = 0

    def train(self, train_loader, val_loader):
        """Loop training utama: train -> (opsional) validate -> checkpoint -> early stopping."""
        nc = self.config.NUM_CLASSES
        empty = {'loss': 0.0, 'miou': 0.0, 'fwiou': 0.0, 'acc': 0.0,
                 'class_iou': [float('nan')] * nc, 'class_acc': [float('nan')] * nc,
                 'class_freq': [0.0] * nc}

        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            current_lr = self.scheduler.get_last_lr()[0]

            tr = self._train_epoch(train_loader, epoch)

            did_validate = (epoch + 1) % self.config.VAL_FREQUENCY == 0
            va = self._validate(val_loader) if did_validate else dict(empty)

            self.scheduler.step()

            self.history['train_loss'].append(tr['loss'])
            self.history['train_iou'].append(tr['miou'])
            self.history['train_fwiou'].append(tr['fwiou'])
            self.history['train_acc'].append(tr['acc'])
            self.history['val_loss'].append(va['loss'])
            self.history['val_iou'].append(va['miou'])
            self.history['val_fwiou'].append(va['fwiou'])
            self.history['val_acc'].append(va['acc'])
            self.history['lr'].append(current_lr)

            for i in range(nc):
                self.history['train_class_iou'][i].append(tr['class_iou'][i])
                self.history['train_class_acc'][i].append(tr['class_acc'][i])
                self.history['val_class_iou'][i].append(va['class_iou'][i])
                self.history['val_class_acc'][i].append(va['class_acc'][i])

            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | LR: {current_lr:.2e} | "
                  f"Train Loss: {tr['loss']:.4f} mIoU: {tr['miou']:.4f} FWIoU: {tr['fwiou']:.4f} Acc: {tr['acc']:.4f} | "
                  f"Val Loss: {va['loss']:.4f} mIoU: {va['miou']:.4f} FWIoU: {va['fwiou']:.4f} Acc: {va['acc']:.4f}")

            score = va[self.best_metric_name]
            is_best = did_validate and score > self.best_val_iou
            if is_best:
                self.best_val_iou = score
                self.patience_counter = 0
                self.best_metrics = {
                    'epoch': epoch + 1,
                    'selected_by': self.best_metric_name,
                    'val': {k: va[k] for k in ('loss', 'miou', 'fwiou', 'acc', 'class_iou', 'class_acc', 'class_freq', 'confusion_matrix')},
                    'train': {k: tr[k] for k in ('loss', 'miou', 'fwiou', 'acc', 'class_iou', 'class_acc')},
                    'class_names': [self.config.CLASS_NAMES.get(i, f'Class {i}') for i in range(nc)],
                    'class_weights': self.class_weights,
                }
                print(f"  ⭐ Best model baru ({self.best_metric_name} = {score:.4f})")
                self._print_class_table(tr, va)
                self._write_json(self.config.LOG_DIR / "best_metrics.json", self.best_metrics)
            elif did_validate:
                self.patience_counter += 1

            self._save_checkpoint(epoch, is_best=is_best)
            self._write_json(self.config.LOG_DIR / "training_history.json", self.history)

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE and epoch >= self.config.WARMUP_EPOCHS:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self._plot_history()
        return self.history

    def _write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, indent=2)

    def _plot_history(self):
        """Simpan grid 3x3 plot: loss, IoU, akurasi (overall & per kelas), learning rate."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        def _arr(v):
            return np.array(v, dtype=float)  # None/NaN -> NaN (garis terputus)

        axes[0, 1].plot(epochs, _arr(self.history['train_iou']), label='Train mIoU')
        axes[0, 1].plot(epochs, _arr(self.history['val_iou']), label='Val mIoU')
        if 'train_fwiou' in self.history:
            axes[0, 1].plot(epochs, _arr(self.history['train_fwiou']), '--', label='Train FWIoU')
            axes[0, 1].plot(epochs, _arr(self.history['val_fwiou']), '--', label='Val FWIoU')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('mIoU & Frequency-Weighted IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(epochs, self.history['train_acc'], label='Train Acc')
        axes[0, 2].plot(epochs, self.history['val_acc'], label='Val Acc')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Overall Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].plot(epochs, self.history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True)

        for i in range(self.config.NUM_CLASSES):
            axes[1, 1].plot(epochs, _arr(self.history['train_class_iou'][i]), label=f'Class {i}', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].set_title('Train IoU per Class')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True)

        for i in range(self.config.NUM_CLASSES):
            axes[1, 2].plot(epochs, _arr(self.history['val_class_iou'][i]), label=f'Class {i}', alpha=0.7)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('IoU')
        axes[1, 2].set_title('Val IoU per Class')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].grid(True)

        for i in range(self.config.NUM_CLASSES):
            axes[2, 0].plot(epochs, _arr(self.history['train_class_acc'][i]), label=f'Class {i}', alpha=0.7)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Accuracy')
        axes[2, 0].set_title('Train Accuracy per Class')
        axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 0].grid(True)

        for i in range(self.config.NUM_CLASSES):
            axes[2, 1].plot(epochs, _arr(self.history['val_class_acc'][i]), label=f'Class {i}', alpha=0.7)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Accuracy')
        axes[2, 1].set_title('Val Accuracy per Class')
        axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 1].grid(True)

        def _last(v):
            x = v[-1] if v else None
            return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else x

        last_train_iou = [_last(self.history['train_class_iou'][i]) for i in range(self.config.NUM_CLASSES)]
        last_val_iou = [_last(self.history['val_class_iou'][i]) for i in range(self.config.NUM_CLASSES)]
        x = np.arange(self.config.NUM_CLASSES)
        width = 0.35
        axes[2, 2].bar(x - width/2, last_train_iou, width, label='Train')
        axes[2, 2].bar(x + width/2, last_val_iou, width, label='Val')
        axes[2, 2].set_xlabel('Class')
        axes[2, 2].set_ylabel('IoU')
        axes[2, 2].set_title('Final IoU per Class')
        axes[2, 2].set_xticks(x)
        axes[2, 2].legend()
        axes[2, 2].grid(True, axis='y')

        plt.tight_layout()
        plot_path = self.config.LOG_DIR / "training_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_training(config=Config, resume_checkpoint=None):
    """Entry point: bangun dataloader -> bangun trainer -> (opsional resume) -> train."""
    start = datetime.now()
    config.make_dirs()
    config.validate_config()

    # Jangan diam-diam training di CPU kalau yang diminta GPU (strict=True -> raise)
    device = get_device(config, strict=True)
    print(f"🖥️  Device: {device_label(device)} | torch {torch.__version__} | "
          f"AMP: {config.AMP_DTYPE if config.MIXED_PRECISION else 'off'}")

    train_loader, val_loader, n_train, n_val, train_mask_paths = build_dataloaders(config)
    print(f"📦 Dataset: {n_train} patch train, {n_val} patch val")

    # Class weights otomatis dari mask TRAIN saja (tidak bocor ke val)
    class_weights, _ = resolve_class_weights(train_mask_paths, config)

    trainer = SegFormerTrainer(config, class_weights=class_weights)

    if resume_checkpoint and Path(resume_checkpoint).exists():
        trainer.load_checkpoint(Path(resume_checkpoint))

    history = trainer.train(train_loader, val_loader)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Best Val {trainer.best_metric_name}: {trainer.best_val_iou:.4f}")
    if trainer.best_metrics:
        bv = trainer.best_metrics['val']
        print(f"  (epoch {trainer.best_metrics['epoch']}: mIoU {bv['miou']:.4f} | "
              f"FWIoU {bv['fwiou']:.4f} | OA {bv['acc']:.4f})")
        print(f"  Detail per kelas: {config.LOG_DIR / 'best_metrics.json'}")

    return history


if __name__ == "__main__":
    run_training(Config)
