"""
run_inference.py
=================
Pipeline inference lengkap: Resample -> Prediksi model -> Vektorisasi ->
Filter poligon kecil -> Smoothing batas -> Style QGIS (.qml).

Versi ini mengadopsi pendekatan `predict.py`:

  1. Tile dibaca DENGAN MARGIN KONTEKS. Band Edge (Canny) & Texture (LBP)
     bergantung pada piksel tetangga. Saat training, band ini dihitung sekali
     dari citra utuh, lalu baru dipotong jadi patch. Kalau tile diproses
     tanpa tetangga, kedua band itu salah tepat di tepi tile. Karena itu tiap
     tile dibaca dengan tambahan `tile_overlap / 2` px di setiap sisi, fitur
     dihitung, lalu margin itu dibuang lagi.
  2. Blending PROBABILITAS antar tile. Patch 512 px di dalam tile dan tile
     di dalam raster sama-sama di-blend dengan bobot Gaussian di level
     probabilitas softmax. Versi lama mem-blend one-hot argmax antar tile,
     sehingga keyakinan model dibuang sebelum digabung.
  3. Overlap patch default 75% (bukan 25%). Prediksi di dekat tepi patch
     sistematis kurang andal; artefak kotak/tangga baru hilang kalau tiap
     piksel dilihat cukup banyak patch dengan posisi berbeda. Biayanya
     ~10x komputasi dibanding 25%; `--patch-overlap 256` (50%) jadi jalan
     tengah.
  4. RGB dibaca apa adanya sebagai uint8, SAMA dengan build_patches.py saat
     training. Versi lama melakukan stretch min-max per tile, yang mengubah
     warna tiap tile secara berbeda dan tidak pernah dilakukan saat training.
  5. Tanpa CuPy. Akumulasi probabilitas di RAM CPU, GPU hanya untuk model.
     Jadi VRAM tidak lagi dipakai untuk kanvas seukuran seluruh raster.
  6. Model dibangun dari config HF saja (tanpa download bobot pretrained),
     karena semua bobot langsung ditimpa checkpoint.

Bagian yang dipertahankan dari versi lama:
  - resample_tif(): TIF bersistem koordinat geografis (lat/lon) otomatis
    direproject ke UTM supaya --resolution benar-benar dalam meter.
  - fix_tile_misclassification() tetap tersedia lewat --fix-tile-artifacts
    (default MATI: margin konteks + blending probabilitas sudah menangani
    penyebab utamanya).
  - --model default ke outputs/models/best_model.pth.

Usage:
    python inference/run_inference.py --input citra.tif --resolution 0.1
    python inference/run_inference.py --input folder/ --model outputs/models/best_model.pth
    python inference/run_inference.py --input citra.tif --patch-overlap 256   # lebih cepat
"""

import argparse
import logging
import os
import shutil
import sys
import warnings
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# PENTING (Windows): torch harus di-import SEBELUM rasterio/geopandas/GDAL.
# Wheel rasterio/pyogrio membawa salinan msvcp140.dll versi lama; kalau DLL itu
# termuat duluan, torch gagal dengan "WinError 1114 ... c10.dll".
import torch
import torch.nn as nn
import torch.nn.functional as F

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import shape as shapely_shape
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from preprocessing.build_patches import create_7band_image, normalize_image
from training.train_model import SegFormer7Band, adapt_segformer_state_dict
from inference.filter_small_polygons import filter_small_pixels
from inference.fix_tile_artifacts import fix_tile_misclassification
from inference.smooth_polygon_boundaries import smooth_polygons_topojson

# Versi AMD/ROCm: pilih GPU (VRAM terbesar), AMP bf16/fp16 lewat gpu_utils.py
from gpu_utils import setup_gpu_env, get_device, device_label, amp_autocast

setup_gpu_env()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message="Setting the shape on a NumPy array.*")
# Log pihak ketiga (HTTP, GDAL, "Created N records") disenyapkan; yang tampil
# hanya warning/error, print() dan progress bar tqdm.
for _noisy in ("httpx", "huggingface_hub", "urllib3", "filelock", "transformers",
               "pyogrio", "fiona"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.ERROR)


# ============================================================================
# MODEL
# ============================================================================

def load_model(model_path: Path, device: torch.device, config=Config) -> nn.Module:
    """Bangun SegFormer7Band (arsitektur identik dengan training, tanpa
    download bobot pretrained) lalu isi dengan bobot checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {model_path}")

    model = SegFormer7Band(config, pretrained=False).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # checkpoint dari transformers versi lama/baru -> samakan nama layer
    state = adapt_segformer_state_dict(state, model.state_dict().keys())
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            raise RuntimeError(
                f"Checkpoint {model_path.name} tidak cocok dengan config sekarang "
                f"(NUM_CLASSES = {config.NUM_CLASSES}, NUM_CHANNELS = {config.NUM_CHANNELS}). "
                f"Kemungkinan model dilatih dengan skema kelas lain — train ulang atau "
                f"pakai checkpoint yang sesuai.\nDetail: {str(e)[:300]}"
            ) from e
        raise
    model.eval()
    return model


# ============================================================================
# RESAMPLE
# ============================================================================

def get_utm_crs_for_bounds(bounds) -> str:
    """Tentukan EPSG UTM yang sesuai berdasarkan centroid bounding box (lon/lat)."""
    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.top + bounds.bottom) / 2
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def resample_tif(input_tif: Path, output_tif: Path, target_resolution: float,
                 resampling_method: Resampling = Resampling.bilinear) -> dict:
    """Resample TIF ke target_resolution (meter). Raster geografis (lat/lon)
    otomatis direproject ke UTM supaya resolusi benar dalam meter."""
    with rasterio.open(input_tif) as src:
        src_crs = src.crs
        dst_crs = get_utm_crs_for_bounds(src.bounds) if src_crs.is_geographic else src_crs

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=target_resolution,
        )
        if dst_width <= 0 or dst_height <= 0:
            raise ValueError(
                f"Dimensi hasil resample tidak valid ({dst_width}x{dst_height}) "
                f"untuk {input_tif}. Cek CRS & target_resolution."
            )

        out_profile = src.profile.copy()
        out_profile.update({
            "crs": dst_crs, "width": dst_width, "height": dst_height,
            "transform": dst_transform,
            "compress": "lzw", "tiled": True, "blockxsize": 512, "blockysize": 512,
        })

        output_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_tif, "w", **out_profile) as dst:
            for band_idx in range(1, src.count + 1):
                dst_band = np.empty((dst_height, dst_width), dtype=src.dtypes[0])
                reproject(
                    source=src.read(band_idx), destination=dst_band,
                    src_transform=src.transform, src_crs=src_crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    resampling=resampling_method,
                    src_nodata=src.nodata, dst_nodata=src.nodata,
                )
                dst.write(dst_band, band_idx)

    return {"output": str(output_tif), "dimensions": f"{dst_width}x{dst_height}", "crs": str(dst_crs)}


# ============================================================================
# TILED / PATCHED INFERENCE
# ============================================================================

def make_gaussian_weight(size: int, sigma_ratio: float) -> np.ndarray:
    """Peta bobot Gaussian (puncak di tengah) untuk blending area yang overlap."""
    sigma = size / sigma_ratio
    center = size / 2.0 - 0.5
    y, x = np.meshgrid(np.arange(size) - center, np.arange(size) - center, indexing="ij")
    w = np.exp(-(y ** 2 + x ** 2) / (2 * sigma ** 2))
    return (w / (w.max() + 1e-9)).astype(np.float32)


def generate_grid(H: int, W: int, size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Jendela (y, x, h, w) yang saling overlap dan menutup seluruh array H x W."""
    stride = max(size - overlap, 1)
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    if not ys or ys[-1] + size < H:
        ys.append(max(0, H - size))
    if not xs or xs[-1] + size < W:
        xs.append(max(0, W - size))
    cells = []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            cells.append((y, x, min(size, H - y), min(size, W - x)))
    return cells


def predict_patches(seven_band: np.ndarray, model: nn.Module, device: torch.device,
                    n_classes: int, patch_size: int, overlap: int,
                    batch_size: int, sigma_ratio: float, use_amp: bool,
                    config=Config) -> np.ndarray:
    """Probabilitas softmax hasil blending Gaussian antar patch yang overlap.
    Return (n_classes, H, W) float32."""
    H, W, _ = seven_band.shape
    prob_acc = np.zeros((n_classes, H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)
    gauss = make_gaussian_weight(patch_size, sigma_ratio)
    cells = generate_grid(H, W, patch_size, overlap)
    img_chw = seven_band.transpose(2, 0, 1)

    with torch.no_grad():
        for i in range(0, len(cells), batch_size):
            batch = cells[i:i + batch_size]
            tensors = []
            for (y, x, h, w) in batch:
                patch = img_chw[:, y:y + h, x:x + w]
                if h < patch_size or w < patch_size:
                    patch = np.pad(patch, ((0, 0), (0, patch_size - h), (0, patch_size - w)), mode="reflect")
                tensors.append(patch)

            batch_t = torch.from_numpy(np.stack(tensors)).float().to(device, non_blocking=True)
            if use_amp:
                with amp_autocast(device, config):
                    logits = model(batch_t)
            else:
                logits = model(batch_t)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()

            for j, (y, x, h, w) in enumerate(batch):
                gw = gauss[:h, :w]
                prob_acc[:, y:y + h, x:x + w] += probs[j, :, :h, :w] * gw[np.newaxis]
                weight_acc[y:y + h, x:x + w] += gw

    return prob_acc / np.maximum(weight_acc, 1e-9)[np.newaxis]


def tiled_inference(tif_path: Path, model: nn.Module, device: torch.device,
                    tile_size: int, tile_overlap: int, patch_size: int,
                    patch_overlap: int, batch_size: int, sigma_ratio: float,
                    config=Config) -> Tuple[np.ndarray, dict]:
    """Klasifikasi raster penuh: tiap tile dibaca dengan margin konteks
    (tile_overlap/2 px), patch di dalamnya di-blend (predict_patches), margin
    dibuang, lalu probabilitas antar tile di-blend Gaussian sebelum argmax."""
    pad = max(tile_overlap // 2, 16)
    use_amp = bool(config.MIXED_PRECISION) and device.type == "cuda"

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()

        if src.count < 3:
            raise ValueError(f"{tif_path.name}: butuh minimal 3 band (RGB), ditemukan {src.count}.")
        if src.dtypes[0] != "uint8":
            raise ValueError(
                f"{tif_path.name}: dtype band '{src.dtypes[0]}', padahal model dilatih dari RGB uint8 "
                f"(0-255) apa adanya. Konversi dulu ke 8-bit (mis. Translate/Export di QGIS) "
                f"dengan cara yang sama seperti data training."
            )

        canvas_gb = (config.NUM_CLASSES + 1) * H * W * 4 / 1024 ** 3
        print(f"    Raster {W} x {H} px | kanvas probabilitas ~{canvas_gb:.1f} GB RAM")

        prob_canvas = np.zeros((config.NUM_CLASSES, H, W), dtype=np.float32)
        weight_canvas = np.zeros((H, W), dtype=np.float32)
        tile_gauss = make_gaussian_weight(tile_size, sigma_ratio=4.0)

        cells = generate_grid(H, W, tile_size, tile_overlap)
        for (y, x, th, tw) in tqdm(cells, desc=tif_path.stem, unit="tile", leave=False):
            ry0, rx0 = max(0, y - pad), max(0, x - pad)
            ry1, rx1 = min(H, y + th + pad), min(W, x + tw + pad)
            window = rasterio.windows.Window(rx0, ry0, rx1 - rx0, ry1 - ry0)

            rgb = src.read([1, 2, 3], window=window)
            seven_band = create_7band_image(rgb.transpose(1, 2, 0))
            if config.NORMALIZE:
                seven_band = normalize_image(seven_band, config)

            probs = predict_patches(seven_band, model, device, config.NUM_CLASSES,
                                    patch_size, patch_overlap, batch_size, sigma_ratio, use_amp,
                                    config)

            off_y, off_x = y - ry0, x - rx0
            core = probs[:, off_y:off_y + th, off_x:off_x + tw]
            tg = tile_gauss[:th, :tw]
            prob_canvas[:, y:y + th, x:x + tw] += core * tg[np.newaxis]
            weight_canvas[y:y + th, x:x + tw] += tg

    np.maximum(weight_canvas, 1e-9, out=weight_canvas)
    pred_map = (prob_canvas / weight_canvas[np.newaxis]).argmax(axis=0).astype(np.uint8)
    return pred_map, profile


# ============================================================================
# VEKTORISASI & STYLE
# ============================================================================

def raster_to_shp(pred_map: np.ndarray, profile: dict, out_shp_path: Path,
                  class_field: str = "kelas", config=Config) -> Path:
    """Vektorisasi array kelas (H, W) jadi polygon SHP, lalu dissolve+explode
    poligon bersebelahan dengan kelas sama."""
    records = []
    for geom_mapping, value in rasterio_shapes(
            pred_map,
            mask=(pred_map != config.IGNORE_INDEX).astype(np.uint8),
            transform=profile["transform"]):
        if int(value) != config.IGNORE_INDEX:
            records.append({"geometry": shapely_shape(geom_mapping), class_field: int(value)})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=profile["crs"])
    gdf = (gdf.dissolve(by=class_field, as_index=False)
              .explode(index_parts=False)
              .reset_index(drop=True))
    out_shp_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(out_shp_path))
    return out_shp_path


def create_qml_style(class_colors: dict, class_names: dict, class_field: str,
                     output_qml: Path, alpha: int = 200) -> Path:
    """File style QGIS (.qml) categorized-symbol sesuai Config.CLASS_COLORS / CLASS_NAMES."""
    alpha_norm = alpha / 255.0
    parts = [
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>",
        '<qgis version="3.28.0" styleCategories="Symbology">',
        f'  <renderer-v2 type="categorizedSymbol" attr="{class_field}" symbollevels="0" '
        f'forceraster="0" enableorderby="0">',
        "    <categories>",
    ]
    for class_id in sorted(class_colors.keys()):
        name = class_names.get(class_id, f"Class {class_id}")
        parts.append(f'      <category value="{class_id}" label="{name}" render="true" symbol="{class_id}"/>')
    parts.append("    </categories>\n    <symbols>")

    for class_id in sorted(class_colors.keys()):
        r, g, b = class_colors[class_id][:3]
        parts.append(f'''      <symbol type="fill" name="{class_id}" alpha="{alpha_norm}" clip_to_extent="1" force_rhr="0">
        <data_defined_properties>
          <Option type="Map">
            <Option type="QString" name="name" value=""/>
            <Option name="properties"/>
            <Option type="QString" name="type" value="collection"/>
          </Option>
        </data_defined_properties>
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="border_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="color" v="{r},{g},{b},{alpha}"/>
          <prop k="joinstyle" v="bevel"/>
          <prop k="offset" v="0,0"/>
          <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          <prop k="offset_unit" v="MM"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_style" v="solid"/>
          <prop k="outline_width" v="0.26"/>
          <prop k="outline_width_unit" v="MM"/>
          <prop k="style" v="solid"/>
          <data_defined_properties>
            <Option type="Map">
              <Option type="QString" name="name" value=""/>
              <Option name="properties"/>
              <Option type="QString" name="type" value="collection"/>
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>''')

    parts.append("    </symbols>\n    <rotation/>\n    <sizescale/>\n  </renderer-v2>\n</qgis>")
    output_qml.parent.mkdir(parents=True, exist_ok=True)
    output_qml.write_text("\n".join(parts), encoding="utf-8")
    return output_qml


def _quiet_call(fn, *args, **kwargs):
    """Jalankan fn dengan stdout dibuang (tqdm menulis ke stderr, jadi progress bar tetap tampil)."""
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stdout(devnull):
        return fn(*args, **kwargs)


# ============================================================================
# PIPELINE PER FILE
# ============================================================================

def process_single_image(input_tif: Path, job_dir: Path, model: nn.Module, device: torch.device,
                         target_resolution: Optional[float], tile_size: int, tile_overlap: int,
                         patch_size: int, patch_overlap: int, batch_size: int, sigma_ratio: float,
                         class_field: str, sieve_threshold: int, smooth_tolerance: float,
                         fix_tile_artifacts: bool = False,
                         tile_fix_rectangularity: float = 0.85, tile_fix_purity: float = 0.8,
                         config=Config) -> dict:
    """Satu TIF: resample -> tiled_inference -> (opsional) fix_tile_misclassification
    -> raster_to_shp -> filter_small_pixels -> smooth_polygons_topojson -> qml."""
    start = datetime.now()
    stem = input_tif.stem

    # 1. Resample (opsional)
    if target_resolution:
        resampled = job_dir / f"{stem}_resampled.tif"
        resample_tif(input_tif, resampled, target_resolution)
        inference_input = resampled
    else:
        inference_input = input_tif

    # 2. Prediksi
    pred_map, profile = tiled_inference(
        inference_input, model, device, tile_size, tile_overlap,
        patch_size, patch_overlap, batch_size, sigma_ratio, config,
    )

    # 2b. Perbaikan blob kotak (opsional, lihat inference/fix_tile_artifacts.py)
    if fix_tile_artifacts:
        pred_map = fix_tile_misclassification(
            pred_map,
            ignore_index=config.IGNORE_INDEX,
            min_rectangularity=tile_fix_rectangularity,
            min_neighbor_purity=tile_fix_purity,
            max_area_px=patch_size * patch_size * 2,
        )

    # 3. Raster -> SHP
    raw_shp = job_dir / f"{stem}_raw.shp"
    raster_to_shp(pred_map, profile, raw_shp, class_field, config)

    # 4. Filter poligon kecil
    sieve_shp = job_dir / f"{stem}_sieve.shp"
    try:
        _quiet_call(filter_small_pixels, str(raw_shp), str(sieve_shp), class_field,
                    {"default": sieve_threshold}, 0.3, 4, False)
        input_smooth = str(sieve_shp)
    except Exception as e:
        tqdm.write(f"  ⚠️  filter_small_pixels gagal ({e}), lanjut tanpa filter")
        input_smooth = str(raw_shp)

    # 5. Smoothing batas poligon
    final_shp = job_dir / f"{stem}.shp"
    try:
        smooth_polygons_topojson(input_smooth, str(final_shp), smooth_tolerance)
    except Exception as e:
        tqdm.write(f"  ⚠️  smoothing gagal ({e}), SHP disimpan tanpa smoothing")
        for ext in [".shp", ".dbf", ".prj", ".shx", ".cpg"]:
            src_f = Path(input_smooth).with_suffix(ext)
            if src_f.exists():
                shutil.copy2(src_f, final_shp.with_suffix(ext))

    # 6. Style QGIS
    qml_path = final_shp.with_suffix(".qml")
    create_qml_style(config.CLASS_COLORS, config.CLASS_NAMES, class_field, qml_path, alpha=200)

    # Bersihkan file sementara
    for tmp in [raw_shp, sieve_shp]:
        for ext in [".shp", ".dbf", ".prj", ".shx", ".cpg", ".qpj"]:
            f = tmp.with_suffix(ext)
            if f.exists():
                f.unlink()

    elapsed = (datetime.now() - start).total_seconds()
    return {"input": str(input_tif), "output_shp": str(final_shp),
            "output_qml": str(qml_path), "time": f"{elapsed:.1f}s"}


# ============================================================================
# ORKESTRASI
# ============================================================================

def run_pipeline(input_path: str, model_path: Optional[str] = None,
                 resolution: Optional[float] = None,
                 tile_size: Optional[int] = None, tile_overlap: Optional[int] = None,
                 patch_size: Optional[int] = None, patch_overlap: Optional[int] = None,
                 batch_size: Optional[int] = None, sigma_ratio: Optional[float] = None,
                 class_field: str = "kelas", sieve_threshold: int = 2,
                 smooth_tolerance: float = 0.3, fix_tile_artifacts: bool = False,
                 tile_fix_rectangularity: float = 0.85, tile_fix_purity: float = 0.8,
                 config=Config) -> dict:
    """Entry point: cari semua TIF di input_path, load model sekali, proses
    satu per satu, simpan hasil ke folder job baru (outputs/job_<n>/).
    Parameter yang None diambil dari Config.INFERENCE_*."""
    tile_size = tile_size or config.INFERENCE_TILE_SIZE
    tile_overlap = config.INFERENCE_TILE_OVERLAP if tile_overlap is None else tile_overlap
    patch_size = patch_size or config.PATCH_SIZE
    patch_overlap = config.INFERENCE_PATCH_OVERLAP if patch_overlap is None else patch_overlap
    batch_size = batch_size or config.INFERENCE_BATCH_SIZE
    sigma_ratio = sigma_ratio or config.INFERENCE_SIGMA_RATIO
    if patch_overlap >= patch_size:
        raise ValueError(f"patch_overlap ({patch_overlap}) harus < patch_size ({patch_size})")

    input_path = Path(input_path)
    tif_files = [input_path] if input_path.is_file() else sorted(
        list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    if not tif_files:
        raise ValueError(f"Tidak ada file TIF: {input_path}")

    job_idx = 1
    while (config.OUTPUT_DIR / f"job_{job_idx}").exists():
        job_idx += 1
    job_dir = config.OUTPUT_DIR / f"job_{job_idx}"
    job_dir.mkdir(parents=True)

    # GPU AMD tidak terdeteksi -> peringatan lalu fallback CPU (strict=False)
    device = get_device(config)
    dev_label = device_label(device)

    model_path = Path(model_path) if model_path else config.MODEL_DIR / "best_model.pth"
    print(f"Device : {dev_label}")
    print(f"Model  : {model_path}")
    print(f"Tile   : {tile_size}px (overlap {tile_overlap}, margin konteks {max(tile_overlap // 2, 16)})")
    print(f"Patch  : {patch_size}px (overlap {patch_overlap} = {patch_overlap / patch_size:.0%}), batch {batch_size}")
    model = load_model(model_path, device, config)

    results = []
    for tif_file in tqdm(tif_files, desc="Files", unit="file"):
        try:
            result = process_single_image(
                tif_file, job_dir, model, device, resolution,
                tile_size, tile_overlap, patch_size, patch_overlap, batch_size, sigma_ratio,
                class_field, sieve_threshold, smooth_tolerance,
                fix_tile_artifacts=fix_tile_artifacts,
                tile_fix_rectangularity=tile_fix_rectangularity,
                tile_fix_purity=tile_fix_purity,
                config=config,
            )
            tqdm.write(f"  -> {result['output_shp']} ({result['time']})")
            results.append(result)
        except Exception as e:
            tqdm.write(f"  ! gagal: {tif_file.name} ({e})")
            results.append({"input": str(tif_file), "error": str(e)})

    total_time = sum(float(r["time"].rstrip("s")) for r in results if "time" in r)
    return {
        "job_dir": str(job_dir),
        "total": len(tif_files),
        "success": len([r for r in results if "error" not in r]),
        "time": f"{total_time:.1f}s",
        "results": results,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Inference: resample -> klasifikasi -> SHP -> style QGIS")
    p.add_argument("--input", "-i", required=True, help="File TIF atau folder")
    p.add_argument("--model", "-m", default=None, help="Checkpoint .pth (default: outputs/models/best_model.pth)")
    p.add_argument("--resolution", "-r", type=float, default=None, help="Resolusi target dalam meter (opsional)")
    p.add_argument("--tile-size", type=int, default=None, help="Ukuran tile px (default Config.INFERENCE_TILE_SIZE)")
    p.add_argument("--tile-overlap", type=int, default=None,
                   help="Overlap tile px, separuhnya jadi margin konteks (default Config.INFERENCE_TILE_OVERLAP)")
    p.add_argument("--patch-size", type=int, default=None, help="Ukuran patch model px (default Config.PATCH_SIZE)")
    p.add_argument("--patch-overlap", type=int, default=None,
                   help="Overlap patch px (default Config.INFERENCE_PATCH_OVERLAP = 75%%; 256 = 50%% lebih cepat)")
    p.add_argument("--batch-size", "--patch-batch", dest="batch_size", type=int, default=None,
                   help="Jumlah patch per batch (default Config.INFERENCE_BATCH_SIZE)")
    p.add_argument("--sigma-ratio", type=float, default=None, help="Ketajaman blending Gaussian patch")
    p.add_argument("--class-field", default="kelas", help="Nama kolom kelas di SHP output")
    p.add_argument("--sieve", type=int, default=2, help="Ambang filter poligon kecil (px^2)")
    p.add_argument("--smooth", type=float, default=0.3, help="Toleransi smoothing poligon")
    p.add_argument("--fix-tile-artifacts", action="store_true",
                   help="Aktifkan perbaikan blob kotak salah-klasifikasi (inference/fix_tile_artifacts.py)")
    p.add_argument("--tile-fix-rectangularity", type=float, default=0.85,
                   help="Ambang luas_blob/luas_bbox untuk dianggap blob 'kotak'")
    p.add_argument("--tile-fix-purity", type=float, default=0.8,
                   help="Ambang proporsi tetangga yang harus 1 kelas dominan")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"\n{'=' * 70}\n🚀 INFERENCE: Resample → Prediksi → SHP → Style QGIS\n{'=' * 70}")
    print(f"Input  : {args.input}")
    if args.resolution:
        print(f"Resolusi: {args.resolution} m")

    result = run_pipeline(
        input_path=args.input, model_path=args.model, resolution=args.resolution,
        tile_size=args.tile_size, tile_overlap=args.tile_overlap,
        patch_size=args.patch_size, patch_overlap=args.patch_overlap,
        batch_size=args.batch_size, sigma_ratio=args.sigma_ratio,
        class_field=args.class_field, sieve_threshold=args.sieve, smooth_tolerance=args.smooth,
        fix_tile_artifacts=args.fix_tile_artifacts,
        tile_fix_rectangularity=args.tile_fix_rectangularity,
        tile_fix_purity=args.tile_fix_purity,
        config=Config,
    )

    print(f"\n{'=' * 70}\n✅ SELESAI\n{'=' * 70}")
    print(f"Job dir: {result['job_dir']}")
    print(f"File   : {result['success']}/{result['total']}")
    print(f"Waktu  : {result['time']}")
    print("=" * 70 + "\n")
    return result


if __name__ == "__main__":
    main()
