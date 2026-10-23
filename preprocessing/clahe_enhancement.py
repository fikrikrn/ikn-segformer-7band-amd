"""
clahe_enhancement.py
=====================
CLAHE Reference-Based Enhancement Pipeline — tahap PERTAMA preprocessing.

Menyamakan tone/kontras seluruh site citra terhadap SATU site referensi,
supaya model tidak belajar perbedaan pencahayaan antar-site alih-alih
perbedaan tutupan lahan. Alurnya:

  1. Pilih site referensi (otomatis atau via Config.CLAHE_REF_SITE).
  2. Reproject & resample referensi ke Config.TARGET_EPSG/TARGET_GSD.
  3. Cari clipLimit CLAHE optimal untuk referensi dengan memaksimalkan
     jarak Jeffries-Matusita (JM) antar confusing pairs (Config.CONFUSING_PAIRS)
     — makin besar jarak JM, makin mudah kelas-kelas itu dipisahkan model.
  4. Untuk setiap site: reproject/resample, CLAHE dengan clip adaptif,
     lalu histogram-matching (tile-wise) ke hasil CLAHE referensi.
  5. Simpan TIF hasil enhancement + salin SHP label ke
     Config.CLAHE_OUTPUT_DIR/<site>/.

Output dari tahap ini menjadi input untuk build_patches.py.

Usage:
    python preprocessing/clahe_enhancement.py
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import features
import geopandas as gpd
import cv2
from scipy.optimize import minimize_scalar
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_crs_match(tiff_path, gdf, target_epsg, reproj_path):
    """Reproject TIF ke target_epsg bila CRS-nya belum sesuai. Mengembalikan
    path TIF yang sudah pasti berada di target_epsg (asli atau hasil reproject)."""
    target_crs = f"EPSG:{target_epsg}"
    with rasterio.open(tiff_path) as src:
        if src.crs and src.crs.to_epsg() == target_epsg:
            return tiff_path
        print(f"    ↻ Reprojecting TIF {src.crs} → {target_crs}")
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({"crs": target_crs, "transform": transform,
                     "width": width, "height": height})
        reproj_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(reproj_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )
    return reproj_path


def reproject_and_resample(src_path, target_epsg, target_gsd, resampling):
    """
    Reproject + resample ke target CRS & GSD.

    Returns: (data, transform, crs_str, H, W, meta)
    """
    target_crs = f"EPSG:{target_epsg}"
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=target_gsd,
        )
        n_bands = src.count
        out = np.zeros((n_bands, height, width), dtype=src.dtypes[0])
        for i in range(n_bands):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=resampling,
            )
        meta = src.meta.copy()
        meta.update({"crs": target_crs, "transform": transform,
                     "width": width, "height": height})
    return out, transform, target_crs, height, width, meta


def to_uint8_band(arr):
    """Rescale satu band float ke uint8 [0, 255] berdasarkan min/max band itu sendiri."""
    a = arr.astype(np.float32)
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    a = (a - mn) / (mx - mn + 1e-9)
    return np.clip(a * 255, 0, 255).astype(np.uint8), mn, mx


def stack_to_uint8(stack):
    """Terapkan to_uint8_band() ke setiap band dalam stack (bands, H, W)."""
    out = np.empty_like(stack, dtype=np.uint8)
    mins, maxs = [], []
    for b in range(stack.shape[0]):
        out[b], mn, mx = to_uint8_band(stack[b])
        mins.append(mn)
        maxs.append(mx)
    return out, mins, maxs


def clean_geodataframe(gdf: gpd.GeoDataFrame,
                        tif_bounds_geom=None) -> gpd.GeoDataFrame:
    """
    Bersihkan GeoDataFrame:
      1. Hapus baris dengan geometri None / empty / invalid
      2. (Opsional) Filter hanya geometri yang bertampalan dengan TIF bounds

    Args:
        gdf            : GeoDataFrame input
        tif_bounds_geom: shapely geometry bounds TIF (opsional, untuk spatial filter)

    Returns:
        GeoDataFrame yang sudah bersih
    """
    n_before = len(gdf)

    # 1. Hapus None / empty
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    # 2. Hapus geometri invalid (self-intersecting, dll)
    gdf = gdf[gdf.geometry.is_valid].copy()

    n_after_clean = len(gdf)
    if n_before - n_after_clean > 0:
        print(f"    🧹 Removed {n_before - n_after_clean} invalid/empty geometries "
              f"({n_before} → {n_after_clean})")

    # 3. Filter bertampalan dengan TIF (opsional)
    if tif_bounds_geom is not None:
        gdf = gdf[gdf.geometry.intersects(tif_bounds_geom)].copy()
        n_after_spatial = len(gdf)
        if n_after_clean - n_after_spatial > 0:
            print(f"    🗺️  Removed {n_after_clean - n_after_spatial} geometries "
                  f"outside TIF bounds ({n_after_clean} → {n_after_spatial})")

    if len(gdf) == 0:
        print("    ⚠️  Tidak ada geometri valid yang tersisa setelah cleaning!")

    return gdf


def build_class_masks_from_array(rgb_u8: np.ndarray,
                                  transform,
                                  gdf: gpd.GeoDataFrame,
                                  kelas_col: str) -> dict:
    """
    Buat boolean mask (H*W,) per kelas langsung dari array yang sudah
    direproject/resample. Transform harus sesuai dengan array rgb_u8.

    PENTING: Fungsi ini menerima transform dari hasil reproject_and_resample,
             bukan dari TIF asli, sehingga ukuran mask selalu cocok dengan
             ukuran array yang akan di-CLAHE.

    Args:
        rgb_u8    : (bands, H, W) uint8 — array hasil reproject_and_resample
        transform : Affine transform yang sesuai dengan rgb_u8
        gdf       : GeoDataFrame (sudah dalam CRS yang sama dengan array)
        kelas_col : Nama kolom kelas

    Returns:
        dict {class_value: ndarray bool (H*W,)}
    """
    _, H, W = rgb_u8.shape
    mask_per_class = {}

    for cls in gdf[kelas_col].unique():
        geoms = gdf[gdf[kelas_col] == cls].geometry.values
        # Filter None/empty sekali lagi untuk keamanan
        geoms = [g for g in geoms if g is not None and not g.is_empty and g.is_valid]
        if not geoms:
            continue

        try:
            raster_mask = features.geometry_mask(
                geoms,
                out_shape=(H, W),
                transform=transform,
                invert=True    # True = piksel DI DALAM geometri
            )
            flat = raster_mask.ravel()   # (H*W,) — ukuran PASTI cocok dengan rgb_u8
            if flat.any():
                mask_per_class[cls] = flat
        except Exception as e:
            print(f"    ⚠️  Skip kelas {cls}: {e}")
            continue

    return mask_per_class


# ============================================================
# JM DISTANCE (Jeffries-Matusita) — ukur keterpisahan antar kelas
# ============================================================

def _bhattacharyya_mv(x1, x2):
    """Jarak Bhattacharyya multivariat antara dua distribusi sampel."""
    m1, m2 = np.mean(x1, axis=0), np.mean(x2, axis=0)
    c1 = np.cov(x1, rowvar=False) + np.eye(x1.shape[1]) * 1e-4
    c2 = np.cov(x2, rowvar=False) + np.eye(x2.shape[1]) * 1e-4
    c  = (c1 + c2) / 2
    diff  = (m1 - m2).reshape(-1, 1)
    term1 = 0.125 * (diff.T @ np.linalg.inv(c) @ diff).item()
    term2 = 0.5 * np.log(
        np.linalg.det(c) /
        (np.sqrt(np.linalg.det(c1) * np.linalg.det(c2)) + 1e-12)
    )
    return term1 + term2


def jm_distance(x1, x2):
    """Jarak Jeffries-Matusita, range [0, 2]. 2 = kelas terpisah sempurna."""
    x1, x2 = np.atleast_2d(x1), np.atleast_2d(x2)
    return float(2 * (1 - np.exp(-_bhattacharyya_mv(x1, x2))))


# ============================================================
# CLAHE OPTIMIZATION
# ============================================================

def _apply_clahe_multiband(stack_u8, clip, tile=(8, 8)):
    """Terapkan CLAHE per-band dengan clipLimit & tileGridSize yang sama."""
    out   = np.empty_like(stack_u8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tile)
    for b in range(stack_u8.shape[0]):
        out[b] = clahe.apply(stack_u8[b])
    return out


def _evaluate_clahe_clip(clip, stack_u8, masks, pairs, tile):
    """
    Objective function untuk optimizer: maximize mean JM distance antar
    confusing pairs (di-negatifkan karena minimize_scalar mencari minimum).

    stack_u8 dan masks harus konsisten ukurannya (sudah dijamin oleh
    build_class_masks_from_array yang memakai transform yang sama).
    """
    enh    = _apply_clahe_multiband(stack_u8, clip, tile)
    # enh shape: (bands, H, W) → reshape ke (bands, H*W)
    enh_flat = enh.reshape(enh.shape[0], -1)   # (bands, H*W)

    scores = []
    for a, b in pairs:
        if a not in masks or b not in masks:
            continue
        # masks[a] shape: (H*W,) bool — dijamin sama dengan enh_flat axis=1
        pa = enh_flat[:, masks[a]].T.astype(np.float32)   # (N_a, bands)
        pb = enh_flat[:, masks[b]].T.astype(np.float32)   # (N_b, bands)
        if len(pa) > 20 and len(pb) > 20:
            try:
                scores.append(jm_distance(pa, pb))
            except Exception:
                continue

    return -np.mean(scores) if scores else 0.0


def find_optimal_clip(stack_u8, masks, pairs, tile,
                      bounds=(1.0, 15.0), xtol=0.2):
    """Cari clipLimit CLAHE optimal via bounded scalar optimization (1D)."""
    res = minimize_scalar(
        _evaluate_clahe_clip,
        bounds=bounds, method="bounded",
        args=(stack_u8, masks, pairs, tile),
        options={"xatol": xtol}
    )
    return float(res.x), float(-res.fun)


# ============================================================
# HISTOGRAM MATCHING (TILE-WISE)
# ============================================================

def _compute_tile_cdf(band_u8, tile_grid, n_bins=256):
    """Hitung CDF histogram per-tile untuk satu band."""
    H, W   = band_u8.shape
    ty, tx = tile_grid
    th, tw = H // ty, W // tx
    cdf_map = np.zeros((ty, tx, n_bins), dtype=np.float32)
    for i in range(ty):
        for j in range(tx):
            y0 = i * th;  y1 = (i+1)*th if i < ty-1 else H
            x0 = j * tw;  x1 = (j+1)*tw if j < tx-1 else W
            tile = band_u8[y0:y1, x0:x1]
            hist, _ = np.histogram(tile, bins=n_bins, range=(0, 256))
            cdf = hist.cumsum().astype(np.float32)
            cdf_map[i, j] = cdf / (cdf[-1] + 1e-9)
    return cdf_map


def _apply_tiled_matching(band_u8, target_cdf_map, tile_grid):
    """Histogram-matching per-tile dengan interpolasi bilinear antar-tile
    (menghindari efek blok/seam di batas tile)."""
    H, W   = band_u8.shape
    ty, tx = tile_grid
    th, tw = H // ty, W // tx
    lut_map = np.zeros((ty, tx, 256), dtype=np.uint8)
    for i in range(ty):
        for j in range(tx):
            y0 = i * th;  y1 = (i+1)*th if i < ty-1 else H
            x0 = j * tw;  x1 = (j+1)*tw if j < tx-1 else W
            tile = band_u8[y0:y1, x0:x1]
            hist, _ = np.histogram(tile, bins=256, range=(0, 256))
            src_cdf = hist.cumsum().astype(np.float32)
            src_cdf /= src_cdf[-1] + 1e-9
            tgt = target_cdf_map[i, j]
            lut_map[i, j] = np.searchsorted(tgt, src_cdf).clip(0, 255).astype(np.uint8)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fy  = (yy + 0.5) / th - 0.5
    fx  = (xx + 0.5) / tw - 0.5
    iy0 = np.clip(np.floor(fy).astype(int), 0, ty - 1)
    iy1 = np.clip(iy0 + 1, 0, ty - 1)
    ix0 = np.clip(np.floor(fx).astype(int), 0, tx - 1)
    ix1 = np.clip(ix0 + 1, 0, tx - 1)
    wy  = np.clip(fy - iy0, 0, 1)
    wx  = np.clip(fx - ix0, 0, 1)
    v00 = lut_map[iy0, ix0, band_u8]
    v01 = lut_map[iy0, ix1, band_u8]
    v10 = lut_map[iy1, ix0, band_u8]
    v11 = lut_map[iy1, ix1, band_u8]
    out = (v00*(1-wy)*(1-wx) + v01*(1-wy)*wx +
           v10*wy*(1-wx)     + v11*wy*wx)
    return np.clip(out, 0, 255).astype(np.uint8)


def match_to_reference(input_u8, reference_u8, tile_grid):
    """Samakan histogram input_u8 ke reference_u8 secara tile-wise, per band."""
    n_bands, H, W = input_u8.shape
    matched = np.zeros_like(input_u8)
    ref = np.zeros((n_bands, H, W), dtype=np.uint8)
    for b in range(n_bands):
        if reference_u8.shape[1:] != (H, W):
            ref[b] = cv2.resize(reference_u8[b], (W, H),
                                interpolation=cv2.INTER_LINEAR)
        else:
            ref[b] = reference_u8[b]
    for b in range(n_bands):
        ref_cdf = _compute_tile_cdf(ref[b], tile_grid)
        matched[b] = _apply_tiled_matching(input_u8[b], ref_cdf, tile_grid)
    return matched


# ============================================================
# HELPER: cari TIF & SHP di dalam folder site
# ============================================================

def _find_tif_shp(site_dir: Path, image_ext: str, shp_ext: str) -> tuple:
    """Cari file TIF & SHP pertama di dalam folder site (langsung atau rekursif)."""
    tif_files = list(site_dir.glob(f"*{image_ext}")) or \
                list(site_dir.rglob(f"*{image_ext}"))
    shp_files = list(site_dir.glob(f"*{shp_ext}")) or \
                list(site_dir.rglob(f"*{shp_ext}"))
    return (tif_files[0] if tif_files else None,
            shp_files[0] if shp_files else None)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_clahe_pipeline(config=Config) -> dict:
    """
    Jalankan seluruh CLAHE pipeline: pilih referensi -> optimasi clipLimit
    (JM distance) -> enhance + histogram-match semua site.

    Returns:
        dict {
            'output_folder' : str,
            'enhanced_files': list[dict],
            'reference_info': dict,
            'processing_time': float
        }
    """
    start_time = datetime.now()
    config.make_dirs()

    kelas_col   = config.KELAS_COL
    target_epsg = config.TARGET_EPSG
    target_gsd  = config.TARGET_GSD
    tile_grid   = config.CLAHE_TILE_GRID
    resampling  = config.RESAMPLING_METHOD
    skip_exist  = config.CLAHE_SKIP_EXISTING

    # ── Temukan semua site ──────────────────────────────────────
    if not config.RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"RAW_DATA_DIR tidak ditemukan: {config.RAW_DATA_DIR}\n"
            f"💡 Jalankan Config.diagnose() untuk cek struktur folder."
        )

    all_sites = sorted([d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()])
    if not all_sites:
        raise FileNotFoundError(
            f"Tidak ada sub-folder site di: {config.RAW_DATA_DIR}"
        )
    print(f"📂 Ditemukan {len(all_sites)} site: {[s.name for s in all_sites]}")

    # ── Tentukan site referensi ─────────────────────────────────
    ref_site_name = config.CLAHE_REF_SITE
    if ref_site_name is None:
        ref_site_dir = None
        for candidate in all_sites:
            tif, shp = _find_tif_shp(candidate, config.IMAGE_EXT, config.SHAPEFILE_EXT)
            if tif and shp:
                ref_site_dir = candidate
                break
        if ref_site_dir is None:
            raise FileNotFoundError("Tidak ada site dengan TIF + SHP sekaligus.")
        print(f"  💡 Auto-select referensi: '{ref_site_dir.name}'")
    else:
        ref_site_dir = config.RAW_DATA_DIR / ref_site_name
        if not ref_site_dir.exists():
            available = [s.name for s in all_sites]
            raise FileNotFoundError(
                f"CLAHE_REF_SITE '{ref_site_name}' tidak ditemukan.\n"
                f"Site tersedia: {available}\n"
                f"→ Set CLAHE_REF_SITE = None untuk otomatis."
            )

    ref_tif, ref_shp = _find_tif_shp(
        ref_site_dir, config.IMAGE_EXT, config.SHAPEFILE_EXT
    )
    if ref_tif is None or ref_shp is None:
        raise FileNotFoundError(
            f"Site referensi '{ref_site_dir.name}' harus berisi .tif dan .shp\n"
            f"  TIF: {ref_tif}\n  SHP: {ref_shp}"
        )

    print(f"\n[1/4] 🔍 Reference site : {ref_site_dir.name}")
    print(f"       TIF : {ref_tif.name}")
    print(f"       SHP : {ref_shp.name}")

    # ── Sinkronisasi CRS referensi ──────────────────────────────
    print("\n[2/4] 🌐 Sinkronisasi CRS referensi...")
    gdf_ref = gpd.read_file(ref_shp)

    if gdf_ref.crs is None:
        raise ValueError(f"SHP referensi tidak memiliki CRS: {ref_shp}")
    if gdf_ref.crs.to_epsg() != target_epsg:
        gdf_ref = gdf_ref.to_crs(f"EPSG:{target_epsg}")

    if kelas_col not in gdf_ref.columns:
        raise ValueError(
            f"Kolom '{kelas_col}' tidak ada di SHP.\n"
            f"Kolom tersedia: {list(gdf_ref.columns)}\n"
            f"→ Perbaiki KELAS_COL di config.py"
        )

    ref_tif_aligned = ensure_crs_match(
        ref_tif, gdf_ref, target_epsg, config.TIFF_REPROJ_REF
    )

    # ── Build reference CLAHE ───────────────────────────────────
    print("\n[3/4] 🔬 Building reference CLAHE (JM optimization)...")

    # Reproject & resample → dapatkan array + transform yang konsisten
    ref_data, ref_transform, ref_crs, ref_H, ref_W, ref_meta = \
        reproject_and_resample(ref_tif_aligned, target_epsg, target_gsd, resampling)

    ref_rgb = ref_data[:3] if ref_data.shape[0] >= 3 else ref_data
    ref_u8, _, _ = stack_to_uint8(ref_rgb)

    print(f"  Array shape setelah resample: {ref_u8.shape}  "
          f"(H={ref_H}, W={ref_W})")

    # Baca bounds TIF untuk filter geometri yang tidak bertampalan
    from shapely.geometry import box as shapely_box

    tif_bounds_geom = shapely_box(
        ref_transform.c,                          # left
        ref_transform.f + ref_H * ref_transform.e,  # bottom
        ref_transform.c + ref_W * ref_transform.a,  # right
        ref_transform.f                           # top
    )

    # Bersihkan GDF: hapus None/empty/invalid + filter spatial
    gdf_ref_clean = clean_geodataframe(gdf_ref, tif_bounds_geom)

    if len(gdf_ref_clean) == 0:
        print("  ⚠️  Tidak ada geometri valid yang bertampalan dengan TIF referensi.")
        print("  → Pakai default clipLimit tanpa JM optimization.")
        best_clip = config.CLAHE_CLIP_DEFAULT
        best_jm   = 0.0
        masks, pairs_valid = {}, []
    else:
        # Build class masks dari array yang sudah direproject
        # → ukuran mask (H*W,) PASTI cocok dengan ref_u8 (bands, H, W)
        masks = build_class_masks_from_array(
            ref_u8, ref_transform, gdf_ref_clean, kelas_col
        )
        print(f"  Kelas ditemukan di SHP: {list(masks.keys())}")

        pairs_valid = [
            p for p in config.CONFUSING_PAIRS
            if p[0] in masks and p[1] in masks
        ]

        if pairs_valid:
            print(f"  Confusing pairs valid: {pairs_valid}")
            best_clip, best_jm = find_optimal_clip(
                ref_u8, masks, pairs_valid, tile_grid,
                bounds=config.CLAHE_CLIP_BOUNDS,
                xtol=config.CLAHE_CLIP_XTOL,
            )
            print(f"  ✅ Optimal clipLimit : {best_clip:.3f}")
            print(f"  ✅ Mean JM distance  : {best_jm:.4f}")
        else:
            best_clip = config.CLAHE_CLIP_DEFAULT
            best_jm   = 0.0
            print(f"  ⚠️  Tidak ada confusing pairs valid "
                  f"(kelas: {list(masks.keys())}, "
                  f"pairs config: {config.CONFUSING_PAIRS})")
            print(f"  → Pakai default clipLimit = {best_clip}")

    reference_clahe_u8 = _apply_clahe_multiband(ref_u8, best_clip, tile_grid)

    reference_info = {
        "site"      : ref_site_dir.name,
        "clip_limit": best_clip,
        "jm_score"  : best_jm,
        "n_classes" : len(masks),
        "pairs_used": pairs_valid,
    }

    # ── Process semua site ──────────────────────────────────────
    print(f"\n[4/4] 🚀 Processing {len(all_sites)} site(s)...")
    enhanced_files = []

    for site_dir in tqdm(all_sites, desc="Sites"):
        site_name = site_dir.name
        tif_path, shp_path = _find_tif_shp(
            site_dir, config.IMAGE_EXT, config.SHAPEFILE_EXT
        )

        if tif_path is None:
            print(f"  ⚠️  {site_name}: Tidak ada TIF, skip.")
            continue

        site_out_dir = config.CLAHE_OUTPUT_DIR / site_name
        site_out_dir.mkdir(parents=True, exist_ok=True)
        out_tif = site_out_dir / tif_path.name

        if skip_exist and out_tif.exists():
            print(f"  ⏭️  {site_name}: Skip (sudah ada).")
            enhanced_files.append({
                "site"    : site_name,
                "tif_path": str(out_tif),
                "shp_path": str(site_out_dir / shp_path.name) if shp_path else None,
            })
            continue

        print(f"\n  📷 Processing: {site_name}")

        try:
            data, transform, crs_str, H, W, meta = reproject_and_resample(
                tif_path, target_epsg, target_gsd, resampling
            )
        except Exception as e:
            print(f"  ❌ {site_name}: Gagal reproject — {e}, skip.")
            continue

        rgb = data[:3] if data.shape[0] >= 3 else data
        rgb_u8, _, _ = stack_to_uint8(rgb)

        # ClipLimit adaptif: makin rendah kontras (std kecil) site, makin
        # besar clipLimit yang dipakai (relatif ke best_clip referensi)
        mean_std = float(np.mean([np.std(rgb_u8[b]) for b in range(rgb_u8.shape[0])]))
        adaptive_clip = float(np.clip(
            np.interp(mean_std, [20, 50], [best_clip * 1.5, best_clip * 0.8]),
            1.0, 15.0
        ))
        rgb_clahe   = _apply_clahe_multiband(rgb_u8, adaptive_clip, tile_grid)
        rgb_matched = match_to_reference(rgb_clahe, reference_clahe_u8, tile_grid)

        out_meta = meta.copy()
        out_meta.update({
            "count": 3, "dtype": "uint8",
            "driver": "GTiff", "compress": "lzw",
        })
        with rasterio.open(out_tif, "w", **out_meta) as dst:
            for b in range(3):
                dst.write(rgb_matched[b], b + 1)

        # Copy SHP (skip jika tidak ada / tidak bertampalan)
        out_shp = None
        if shp_path:
            try:
                gdf_site = gpd.read_file(shp_path)
                if gdf_site.crs is None:
                    raise ValueError("SHP tidak punya CRS")
                if gdf_site.crs.to_epsg() != target_epsg:
                    gdf_site = gdf_site.to_crs(f"EPSG:{target_epsg}")

                # Buat bounds TIF site
                site_bounds_geom = shapely_box(
                    transform.c,
                    transform.f + H * transform.e,
                    transform.c + W * transform.a,
                    transform.f
                )
                gdf_site_clean = clean_geodataframe(gdf_site, site_bounds_geom)

                if len(gdf_site_clean) == 0:
                    print(f"  ⚠️  {site_name}: SHP tidak bertampalan dengan TIF, skip SHP.")
                else:
                    for ext in [".shp", ".dbf", ".prj", ".shx", ".cpg", ".qpj"]:
                        src_f = shp_path.with_suffix(ext)
                        if src_f.exists():
                            shutil.copy2(src_f, site_out_dir / src_f.name)
                    out_shp = str(site_out_dir / shp_path.name)
            except Exception as e:
                print(f"  ⚠️  {site_name}: Gagal proses SHP ({e}), lanjut tanpa SHP.")

        enhanced_files.append({
            "site"    : site_name,
            "tif_path": str(out_tif),
            "shp_path": out_shp,
        })
        print(f"  ✅ Saved → {out_tif}")

    elapsed = (datetime.now() - start_time).total_seconds()
    result  = {
        "output_folder" : str(config.CLAHE_OUTPUT_DIR),
        "enhanced_files": enhanced_files,
        "reference_info": reference_info,
        "processing_time": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"✅ CLAHE Pipeline selesai dalam {elapsed:.1f} detik")
    print(f"   Output : {config.CLAHE_OUTPUT_DIR}")
    print(f"   Total  : {len(enhanced_files)} site diproses")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    Config.diagnose()
    result = run_clahe_pipeline(Config)
