"""
calculate_class_weights.py
===========================
Hitung class weights untuk mengatasi class imbalance dari file mask .npy.

Modul ini dipakai dengan DUA cara:

1. OTOMATIS saat training (default — tidak perlu copy-paste lagi)
   training/train_model.py memanggil `resolve_class_weights()` dengan
   daftar mask hasil split TRAIN, selama `Config.CLASS_WEIGHTS = "auto"`.
   Hasilnya di-cache ke `Config.CLASS_WEIGHT_CACHE` (JSON) dan hanya
   dihitung ulang kalau dataset (daftar file / ukuran / waktu modifikasi
   mask), metode, NUM_CLASSES, atau IGNORE_INDEX berubah.

2. MANUAL via CLI (untuk inspeksi / laporan)
   Mencetak distribusi kelas + bobot. Default memakai split train yang
   SAMA dengan training (random_state = Config.RANDOM_SEED), atau seluruh
   mask dengan --all.

Metode (--method / Config.CLASS_WEIGHT_METHOD):
  - balanced  : inverse frequency  N / (C * n_c), dinormalisasi sum = C
  - sqrt      : inverse sqrt frequency 1 / sqrt(n_c), dinormalisasi sum = C
                (lebih "lembut" dari balanced, cocok kalau imbalance ekstrem)
  - effective : "effective number of samples" (Cui et al., 2019),
                dinormalisasi sum = C. Catatan: dengan jumlah pixel jutaan,
                hasilnya hampir seragam untuk semua kelas.
  - sklearn   : formula sklearn compute_class_weight('balanced')
                N / (C_ada * n_c), TANPA normalisasi

Kelas yang tidak punya pixel sama sekali selalu mendapat weight 0.

Usage:
    python data_qc/calculate_class_weights.py                 # split train
    python data_qc/calculate_class_weights.py --all           # semua mask
    python data_qc/calculate_class_weights.py --method sqrt
"""

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config


VALID_METHODS = ("balanced", "sqrt", "effective", "sklearn")


# ============================================================
# HITUNG DISTRIBUSI PIXEL
# ============================================================

def count_class_pixels(mask_paths, config=Config, show_progress=True):
    """
    Hitung jumlah pixel per kelas dari daftar file mask .npy.

    Pixel bernilai IGNORE_INDEX dilewati. Pixel dengan nilai di luar
    [0, NUM_CLASSES) (dan bukan IGNORE_INDEX) dihitung terpisah sebagai
    `n_invalid` supaya bisa dilaporkan.

    Returns:
        (class_counts: np.ndarray[int64] panjang NUM_CLASSES, n_invalid: int)
    """
    num_classes = config.NUM_CLASSES
    class_counts = np.zeros(num_classes, dtype=np.int64)
    n_invalid = 0

    iterator = tqdm(mask_paths, desc="Scanning masks") if show_progress else mask_paths
    for mask_path in iterator:
        mask = np.load(mask_path).ravel()
        mask = mask[mask != config.IGNORE_INDEX].astype(np.int64, copy=False)
        if mask.size == 0:
            continue
        in_range = (mask >= 0) & (mask < num_classes)
        n_invalid += int((~in_range).sum())
        class_counts += np.bincount(mask[in_range], minlength=num_classes)[:num_classes]

    return class_counts, n_invalid


# ============================================================
# HITUNG WEIGHTS DARI DISTRIBUSI
# ============================================================

def compute_class_weights(class_counts, method="balanced", num_classes=None):
    """
    Hitung class weights dari jumlah pixel per kelas.

    Args:
        class_counts: array-like jumlah pixel per kelas
        method      : salah satu VALID_METHODS
        num_classes : default = len(class_counts)

    Returns:
        list[float] dengan panjang num_classes
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    num_classes = num_classes or len(counts)
    present = counts > 0
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float64)

    if total == 0:
        raise ValueError("Semua mask kosong: tidak ada pixel valid untuk menghitung class weights.")

    if method == "balanced":
        weights[present] = total / (num_classes * counts[present])
        normalize = True
    elif method == "sqrt":
        weights[present] = 1.0 / np.sqrt(counts[present])
        normalize = True
    elif method == "effective":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts[present])
        weights[present] = (1.0 - beta) / effective_num
        normalize = True
    elif method == "sklearn":
        weights[present] = total / (present.sum() * counts[present])
        normalize = False
    else:
        raise ValueError(f"Unknown method: {method!r}. Pilih salah satu: {VALID_METHODS}")

    if normalize and weights.sum() > 0:
        weights = weights / weights.sum() * num_classes

    return [float(w) for w in weights]


# ============================================================
# AUTO + CACHE (dipanggil dari training/train_model.py)
# ============================================================

def _dataset_fingerprint(mask_paths, method, config=Config):
    """Hash ringan dari daftar mask (path relatif + ukuran + mtime) dan
    parameter yang mempengaruhi hasil. Tidak membaca isi file -> cepat."""
    h = hashlib.sha1()
    h.update(f"{method}|{config.NUM_CLASSES}|{config.IGNORE_INDEX}".encode())
    masks_dir = Path(config.MASKS_DIR)
    for p in sorted(Path(p) for p in mask_paths):
        try:
            rel = p.relative_to(masks_dir).as_posix()
        except ValueError:
            rel = p.as_posix()
        st = p.stat()
        h.update(f"|{rel}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def resolve_class_weights(mask_paths, config=Config, refresh=False, verbose=True):
    """
    Tentukan class weights yang dipakai training sesuai Config.CLASS_WEIGHTS:

      - None          -> tanpa class weights (return None)
      - list / tuple  -> dipakai apa adanya (override manual)
      - "auto"        -> hitung dari `mask_paths` (sebaiknya mask TRAIN saja)
                         dengan Config.CLASS_WEIGHT_METHOD; pakai cache JSON
                         di Config.CLASS_WEIGHT_CACHE bila dataset tidak berubah

    Returns:
        (weights: list[float] | None, info: dict)
    """
    setting = config.CLASS_WEIGHTS

    if setting is None:
        if verbose:
            print("⚖️  CLASS_WEIGHTS = None -> training tanpa class weights")
        return None, {"source": "disabled"}

    if isinstance(setting, (list, tuple)):
        weights = [float(w) for w in setting]
        if len(weights) != config.NUM_CLASSES:
            raise ValueError(
                f"len(CLASS_WEIGHTS) = {len(weights)} != NUM_CLASSES = {config.NUM_CLASSES}"
            )
        if verbose:
            print("⚖️  CLASS_WEIGHTS manual dari config.py dipakai apa adanya:")
            _print_weights(weights, None, config)
        return weights, {"source": "manual", "weights": weights}

    if not (isinstance(setting, str) and setting.lower() == "auto"):
        raise ValueError(f'CLASS_WEIGHTS harus "auto", None, atau list. Didapat: {setting!r}')

    method = config.CLASS_WEIGHT_METHOD
    cache_path = Path(config.CLASS_WEIGHT_CACHE)
    fingerprint = _dataset_fingerprint(mask_paths, method, config)

    if not refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                if verbose:
                    print(f"⚖️  CLASS_WEIGHTS auto ({method}) -> dari cache "
                          f"{cache_path.name} (dataset tidak berubah):")
                    _print_weights(cached["weights"], cached.get("class_counts"), config)
                return cached["weights"], {"source": "cache", **cached}
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # cache rusak -> hitung ulang

    if verbose:
        print(f"⚖️  CLASS_WEIGHTS auto ({method}) -> menghitung dari {len(mask_paths)} mask train ...")
    class_counts, n_invalid = count_class_pixels(mask_paths, config, show_progress=verbose)
    weights = compute_class_weights(class_counts, method, config.NUM_CLASSES)

    info = {
        "fingerprint": fingerprint,
        "method": method,
        "n_masks": len(mask_paths),
        "class_counts": class_counts.tolist(),
        "n_invalid_pixels": n_invalid,
        "weights": weights,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"⚠️  Gagal menulis cache class weights: {e}")

    if verbose:
        if n_invalid:
            print(f"⚠️  {n_invalid:,} pixel bernilai di luar 0..{config.NUM_CLASSES-1} "
                  f"(bukan IGNORE_INDEX) dilewati — cek data_qc/validate_mask_data.py")
        _print_weights(weights, class_counts, config)
    return weights, {"source": "computed", **info}


def _print_weights(weights, class_counts, config=Config):
    total = float(np.sum(class_counts)) if class_counts is not None else 0.0
    for cls, w in enumerate(weights):
        name = config.CLASS_NAMES.get(cls, f"Class {cls}")
        if class_counts is not None and total > 0:
            pct = class_counts[cls] / total * 100
            print(f"     {cls} {name:<18} weight = {w:>9.4f}   ({pct:6.2f}% pixel)")
        else:
            print(f"     {cls} {name:<18} weight = {w:>9.4f}")


# ============================================================
# CLI
# ============================================================

def _train_mask_paths(config=Config):
    """Mask untuk split TRAIN — identik dengan training/train_model.py
    (pasangan patch/mask yang sama, train_test_split dengan seed yang sama)."""
    from sklearn.model_selection import train_test_split

    patches_dir, masks_dir = config.PATCHES_DIR, config.MASKS_DIR
    patch_paths, mask_paths = [], []
    for patch_f in sorted(patches_dir.rglob("*.npy")):
        mask_f = masks_dir / patch_f.relative_to(patches_dir)
        if mask_f.exists():
            patch_paths.append(patch_f)
            mask_paths.append(mask_f)
    if not patch_paths:
        raise FileNotFoundError(f"Tidak ada pasangan patch/mask .npy di {patches_dir}")

    _, _, train_m, _ = train_test_split(
        patch_paths, mask_paths,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )
    return train_m


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calculate class weights")
    parser.add_argument("--method", type=str, default=None, choices=VALID_METHODS,
                        help="Metode weights (default: Config.CLASS_WEIGHT_METHOD)")
    parser.add_argument("--all", action="store_true",
                        help="Pakai SEMUA mask (default: hanya split train seperti saat training)")
    args = parser.parse_args()

    method = args.method or Config.CLASS_WEIGHT_METHOD

    if args.all:
        mask_paths = sorted(Config.MASKS_DIR.rglob("*.npy"))
        scope = "SEMUA mask"
    else:
        mask_paths = _train_mask_paths(Config)
        scope = "split TRAIN"
    if not mask_paths:
        raise FileNotFoundError(f"Tidak ada file .npy di {Config.MASKS_DIR}")

    class_counts, n_invalid = count_class_pixels(mask_paths, Config)
    total = class_counts.sum()

    print("\n" + "=" * 70)
    print(f"CLASS DISTRIBUTION ({scope}, {len(mask_paths)} mask)")
    print("=" * 70)
    for cls in range(Config.NUM_CLASSES):
        pct = class_counts[cls] / total * 100 if total else 0
        name = Config.CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"  {cls} {name:<18}: {class_counts[cls]:>15,} px ({pct:6.2f}%)")
    print(f"  {'─' * 66}\n  Total               : {total:>15,} px")
    if n_invalid:
        print(f"  ⚠️  {n_invalid:,} pixel di luar rentang kelas dilewati")

    weights = compute_class_weights(class_counts, method, Config.NUM_CLASSES)
    print("\n" + "=" * 70)
    print(f"CLASS WEIGHTS ({method.upper()})")
    print("=" * 70)
    _print_weights(weights, class_counts, Config)

    present = class_counts[class_counts > 0]
    ratio = present.max() / present.min() if present.size else 0
    print(f"\n  Rasio imbalance (kelas terbanyak : tersedikit) = {ratio:,.1f} : 1")
    if ratio >= 100 and method == "balanced":
        print('  💡 Imbalance ekstrem. Kalau training tidak stabil, coba '
              'CLASS_WEIGHT_METHOD = "sqrt" di config.py')

    print('\nℹ️  Tidak perlu copy-paste ke config.py: dengan CLASS_WEIGHTS = "auto" '
          'bobot ini dihitung otomatis saat training.\n')


if __name__ == "__main__":
    main()
