"""
build_patches.py
=================
Tahap KEDUA preprocessing: TIF (idealnya hasil clahe_enhancement.py) + SHP
label -> citra 7-band ternormalisasi -> mask hasil rasterisasi -> patch
.npy siap training.

Untuk setiap site:
  1. Baca TIF RGB (uint8)
  2. Build 7-band image: R, G, B, ExG, VARI, Edge, Texture
  3. Normalisasi (mean/std per channel)
  4. Rasterize SHP -> mask (uint8)
  5. Patch extraction dengan overlap -> simpan .npy ke
     Config.PATCHES_DIR / Config.MASKS_DIR

Entry point: run_preprocessing(config)

Usage:
    python preprocessing/build_patches.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from tqdm import tqdm
import cv2
from skimage.feature import local_binary_pattern

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config


# ============================================================
# VEGETATION INDICES & FEATURE COMPUTATION
# ============================================================

def compute_vegetation_indices(r: np.ndarray, g: np.ndarray,
                                b: np.ndarray) -> tuple:
    """
    Hitung ExG dan VARI dari band RGB uint8.

    Args:
        r, g, b : (H, W) uint8

    Returns:
        (exg, vari) : masing-masing (H, W) float32 dalam [0, 1]
    """
    r_f = r.astype(np.float32) / 255.0
    g_f = g.astype(np.float32) / 255.0
    b_f = b.astype(np.float32) / 255.0

    # Excess Green Index: 2G - R - B, range [-1, 1] → rescale ke [0, 1]
    exg = np.clip(2 * g_f - r_f - b_f, -1.0, 1.0)
    exg = (exg + 1.0) / 2.0

    # Visible Atmospherically Resistant Index
    denom = g_f + r_f - b_f
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    vari  = np.clip((g_f - r_f) / denom, -1.0, 1.0)
    vari  = (vari + 1.0) / 2.0

    return exg.astype(np.float32), vari.astype(np.float32)


def compute_edge(gray: np.ndarray) -> np.ndarray:
    """
    Canny edge detection.

    Args:
        gray : (H, W) uint8

    Returns:
        edge : (H, W) float32 dalam [0, 1]
    """
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return (edges.astype(np.float32) / 255.0)


def compute_texture(gray: np.ndarray) -> np.ndarray:
    """
    Local Binary Pattern (LBP) texture.

    Dinormalisasi dengan rentang TEORITIS tetap (bukan min-max lokal),
    supaya nilai konsisten baik saat dihitung dari citra penuh (training)
    maupun dari crop tile kecil (inference) — menghindari seam antar-tile.
    """
    radius   = 3
    n_points = 8 * radius
    lbp      = local_binary_pattern(gray, n_points, radius, method='uniform')

    # method='uniform' → kode integer pasti 0 .. n_points+1, TIDAK bergantung
    # pada isi citra, sehingga aman dipakai sebagai konstanta normalisasi.
    max_val = n_points + 1
    return (lbp.astype(np.float32) / max_val)


def create_7band_image(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Buat 7-band image dari RGB uint8.

    Args:
        rgb_uint8 : (H, W, 3) uint8

    Returns:
        seven_band : (H, W, 7) float32, semua band dalam [0, 1]
                     Urutan: R, G, B, ExG, VARI, Edge, Texture
    """
    r = rgb_uint8[:, :, 0]
    g = rgb_uint8[:, :, 1]
    b = rgb_uint8[:, :, 2]

    r_f = r.astype(np.float32) / 255.0
    g_f = g.astype(np.float32) / 255.0
    b_f = b.astype(np.float32) / 255.0

    exg, vari = compute_vegetation_indices(r, g, b)

    gray    = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    edge    = compute_edge(gray)
    texture = compute_texture(gray)

    return np.stack([r_f, g_f, b_f, exg, vari, edge, texture], axis=-1)


def normalize_image(image: np.ndarray, config=Config) -> np.ndarray:
    """
    Normalisasi (H, W, 7) float32 dengan mean/std per channel.

    Returns:
        normalized : (H, W, 7) float32
    """
    mean = np.array(config.NORMALIZE_MEAN, dtype=np.float32)
    std  = np.array(config.NORMALIZE_STD,  dtype=np.float32)
    return ((image - mean) / std).astype(np.float32)


# ============================================================
# RASTERIZATION
# ============================================================

def rasterize_shapefile(shp_path: Path, ref_tif_path: Path,
                         out_mask_path: Path, kelas_col: str,
                         ignore_index: int = 255) -> Path:
    """
    Rasterize shapefile ke mask GeoTIFF menggunakan kolom kelas.

    Args:
        shp_path      : Path ke shapefile
        ref_tif_path  : Path ke TIF referensi (untuk extent & transform)
        out_mask_path : Path output mask .tif
        kelas_col     : Nama kolom kelas di shapefile
        ignore_index  : Nilai untuk piksel di luar semua geometri

    Returns:
        Path ke mask yang disimpan
    """
    gdf = gpd.read_file(shp_path)

    with rasterio.open(ref_tif_path) as src:
        # Sinkronisasi CRS
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # Validasi kolom kelas
        if kelas_col not in gdf.columns:
            available = list(gdf.columns)
            raise ValueError(
                f"Kolom '{kelas_col}' tidak ada di shapefile. "
                f"Kolom tersedia: {available}"
            )

        # Pastikan nilai kelas adalah integer
        gdf = gdf.copy()
        gdf[kelas_col] = gdf[kelas_col].astype(int)

        shapes = (
            (geom, int(val))
            for geom, val in zip(gdf.geometry, gdf[kelas_col])
            if geom is not None and not geom.is_empty
        )

        mask_array = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=ignore_index,   # Piksel di luar geometri = ignore_index
            dtype=np.uint8
        )

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "count" : 1,
            "dtype" : "uint8",
            "nodata": ignore_index,
        })

        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_mask_path, "w", **out_meta) as dst:
            dst.write(mask_array, 1)

    return out_mask_path


# ============================================================
# PATCH EXTRACTION
# ============================================================

def extract_patches(image: np.ndarray, mask: np.ndarray,
                    patch_size: int, stride: int,
                    min_valid_ratio: float,
                    ignore_index: int) -> tuple:
    """
    Ekstrak patch dari image dan mask dengan sliding window + overlap.

    Args:
        image          : (H, W, C) float32
        mask           : (H, W) uint8
        patch_size     : Ukuran patch (px)
        stride         : Stride sliding window (px)
        min_valid_ratio: Minimum rasio piksel valid [0, 1]
        ignore_index   : Nilai piksel yang diabaikan di mask

    Returns:
        (patches, masks, positions)
        patches   : list of (patch_size, patch_size, C) float32
        masks     : list of (patch_size, patch_size) uint8
        positions : list of (y, x) int
    """
    H, W       = image.shape[:2]
    patches    = []
    mask_list  = []
    positions  = []
    total_px   = patch_size * patch_size

    def _try_add(y: int, x: int):
        # Pastikan patch tidak keluar batas
        if y + patch_size > H or x + patch_size > W:
            return
        p  = image[y:y + patch_size, x:x + patch_size]
        mp = mask[y:y + patch_size, x:x + patch_size]
        valid_ratio = float(np.sum(mp != ignore_index)) / total_px
        if valid_ratio >= min_valid_ratio:
            patches.append(p)
            mask_list.append(mp)
            positions.append((y, x))

    # Sliding window
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    # Pastikan baris/kolom terakhir tercakup
    if ys and ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs and xs[-1] + patch_size < W:
        xs.append(W - patch_size)

    for y in ys:
        for x in xs:
            _try_add(y, x)

    return patches, mask_list, positions


# ============================================================
# SITE PROCESSOR
# ============================================================

def process_single_site(tif_path: Path, shp_path: Path,
                        site_name: str, config=Config) -> dict:
    """
    Proses satu site: baca TIF → 7-band → normalisasi → patch → simpan .npy

    Args:
        tif_path  : Path ke TIF RGB
        shp_path  : Path ke shapefile
        site_name : Nama site
        config    : Config class

    Returns:
        dict {'site': str, 'n_patches': int, 'skipped': bool}
    """
    print(f"\n  🔄 Site: {site_name}")

    if not tif_path.exists():
        print(f"  ⚠️  TIF tidak ditemukan: {tif_path}, skip.")
        return {"site": site_name, "n_patches": 0, "skipped": True}

    if not shp_path.exists():
        print(f"  ⚠️  SHP tidak ditemukan: {shp_path}, skip.")
        return {"site": site_name, "n_patches": 0, "skipped": True}

    # ── Step 1: Baca TIF RGB ──────────────────────────────────────
    with rasterio.open(tif_path) as src:
        # Baca 3 band RGB, transpose ke (H, W, 3)
        rgb_uint8 = src.read([1, 2, 3]).transpose(1, 2, 0)  # (H, W, 3)
        H, W      = rgb_uint8.shape[:2]

    print(f"    📐 Dimensi: {H} x {W}")

    # ── Step 2: Build 7-band image ────────────────────────────────
    print("    🎨 Building 7-band image...")
    seven_band = create_7band_image(rgb_uint8)   # (H, W, 7) float32 [0,1]

    # ── Step 3: Normalisasi ───────────────────────────────────────
    if config.NORMALIZE:
        print("    📏 Normalizing...")
        seven_band = normalize_image(seven_band, config)

    # ── Step 4: Rasterize mask ────────────────────────────────────
    print("    🗺️  Rasterizing mask...")
    mask_tif_path = config.MASKS_DIR / site_name / "mask.tif"
    mask_tif_path.parent.mkdir(parents=True, exist_ok=True)

    rasterize_shapefile(
        shp_path      = shp_path,
        ref_tif_path  = tif_path,
        out_mask_path = mask_tif_path,
        kelas_col     = config.KELAS_COL,
        ignore_index  = config.IGNORE_INDEX,
    )

    with rasterio.open(mask_tif_path) as msrc:
        mask_array = msrc.read(1)   # (H, W) uint8

    # ── Step 5: Patch extraction ──────────────────────────────────
    print("    ✂️  Extracting patches...")
    patches, masks, positions = extract_patches(
        image           = seven_band,
        mask            = mask_array,
        patch_size      = config.PATCH_SIZE,
        stride          = config.STRIDE,
        min_valid_ratio = config.MIN_VALID_PIXELS,
        ignore_index    = config.IGNORE_INDEX,
    )

    if not patches:
        print(f"    ⚠️  Tidak ada patch valid untuk {site_name}.")
        return {"site": site_name, "n_patches": 0, "skipped": False}

    # ── Step 6: Simpan .npy ───────────────────────────────────────
    patch_out_dir = config.PATCHES_DIR / site_name
    mask_out_dir  = config.MASKS_DIR   / site_name
    patch_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    save_dtype_img  = np.dtype(config.SAVE_DTYPE_IMAGE)
    save_dtype_mask = np.dtype(config.SAVE_DTYPE_MASK)

    for idx, (patch, mask_p, pos) in enumerate(zip(patches, masks, positions)):
        y, x      = pos
        stem      = f"{site_name}_y{y:05d}_x{x:05d}"
        patch_f   = patch_out_dir / f"{stem}.npy"
        mask_f    = mask_out_dir  / f"{stem}.npy"

        np.save(patch_f, patch.astype(save_dtype_img))
        np.save(mask_f,  mask_p.astype(save_dtype_mask))

    print(f"    ✅ {len(patches)} patch disimpan → {patch_out_dir}")
    return {"site": site_name, "n_patches": len(patches), "skipped": False}


# ============================================================
# MAIN PREPROCESSING ENTRY POINT
# ============================================================

def run_preprocessing(config=Config) -> dict:
    """
    Jalankan preprocessing untuk semua site dari direktori TIF dan SHP.

    Args:
        config : Config class

    Returns:
        dict {
            'patches_dir'   : str,
            'masks_dir'     : str,
            'site_results'  : list[dict],
            'total_patches' : int,
        }
    """
    start = datetime.now()

    config.make_dirs()

    # Scan TIF files dari RAW_DATA_DIR
    # (catatan: sebelumnya memakai config.RAW_DIR yang tidak ada di Config —
    #  sudah diperbaiki jadi RAW_DATA_DIR, atribut yang benar-benar ada)
    tif_files = list(config.RAW_DATA_DIR.glob("*.tif")) + \
                list(config.RAW_DATA_DIR.glob("*.TIF"))

    if not tif_files:
        raise ValueError(f"Tidak ada file TIF di {config.RAW_DATA_DIR}")

    print(f"\n{'='*60}")
    print(f"🔧 PREPROCESSING — {len(tif_files)} site")
    print(f"{'='*60}")

    site_results  = []
    total_patches = 0

    for tif_path in tqdm(tif_files, desc="Preprocessing sites"):
        site_name = tif_path.stem

        # Cari shapefile dengan nama yang sama
        shp_path = config.RAW_DATA_DIR / f"{site_name}.shp"

        result = process_single_site(tif_path, shp_path, site_name, config)
        site_results.append(result)
        total_patches += result.get("n_patches", 0)

    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n{'='*60}")
    print(f"✅ Preprocessing selesai dalam {elapsed:.1f} detik")
    print(f"   Total patch : {total_patches}")
    print(f"   Patches dir : {config.PATCHES_DIR}")
    print(f"   Masks dir   : {config.MASKS_DIR}")
    print(f"{'='*60}")

    return {
        "patches_dir"  : str(config.PATCHES_DIR),
        "masks_dir"    : str(config.MASKS_DIR),
        "site_results" : site_results,
        "total_patches": total_patches,
    }


if __name__ == "__main__":
    # Standalone: jalankan preprocessing langsung dari TIF/SHP di RAW_DATA_DIR
    # (idealnya TIF di sini adalah hasil clahe_enhancement.py)
    run_preprocessing(Config)
