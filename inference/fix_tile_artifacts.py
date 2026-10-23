"""
fix_tile_artifacts.py
======================
Post-process: perbaiki blob hasil segmentasi yang salah kelas SECARA UTUH
karena satu tile/patch inference diprediksi salah oleh model — bukan
kesalahan piksel acak.

Ciri khas artefak ini (yang membedakannya dari batas tutupan lahan asli):
  1. Bentuknya nyaris kotak sempurna & axis-aligned, mengikuti grid
     tile/patch saat inference — bukan kontur alami yang selalu tidak
     beraturan. Diukur lewat **rectangularity** = luas blob / luas
     bounding box-nya (mendekati 1 = solid kotak).
  2. Dikelilingi mayoritas SATU kelas lain yang solid — diukur lewat
     **neighbor purity**: proporsi piksel tetangga (setelah blob didilasi
     beberapa piksel) yang merupakan satu kelas dominan.

Kalau kedua syarat terpenuhi dan kelas dominan tetangga BEDA dari kelas
blob, seluruh blob direlabel ke kelas tetangganya. Ini berbeda dari
filter_small_polygons.py (sieve berbasis luas, tidak peduli bentuk) — di
sini yang jadi kunci adalah BENTUK KOTAK + KONSISTENSI TETANGGA, sehingga
blob besar yang salah kelas pun bisa terdeteksi & diperbaiki, bukan hanya
noise kecil.

Dua cara pakai:
  1. fix_tile_misclassification() dipanggil langsung dari run_inference.py
     di atas array pred_map, SEBELUM divektorisasi ke SHP (lebih murah &
     akurat karena masih di grid piksel asli, sebelum ada distorsi
     rasterize-ulang).
  2. CLI standalone untuk memperbaiki SHP hasil inference yang SUDAH ada
     (rasterize -> fix -> polygonize), tanpa perlu run ulang model.

Usage (standalone):
    python inference/fix_tile_artifacts.py --input hasil_prediksi.shp
    python inference/fix_tile_artifacts.py --input hasil_prediksi.shp \\
        --output hasil_fixed.shp --class-field kelas \\
        --pixel-size 0.3 --max-blob-px 600000 \\
        --rectangularity 0.85 --neighbor-purity 0.8
"""

import argparse
import sys
from typing import Optional

import geopandas as gpd
import numpy as np
from scipy import ndimage

# Reuse helper rasterize/polygonize dari filter_small_polygons.py supaya
# tidak duplikat logic (grid dari bounds, factorize kelas string, dtype
# rasterisasi, polygonize) — keduanya sama-sama pola rasterize->proses->polygonize.
from inference.filter_small_polygons import (
    _compute_grid_from_bounds,
    _determine_dtype_and_nodata,
    _prepare_class_values,
    _polygonize,
)


# ============================================================
# CORE: operasi di array/raster (dipakai run_inference.py & CLI di bawah)
# ============================================================

def fix_tile_misclassification(
    pred_map: np.ndarray,
    ignore_index: int = 255,
    min_rectangularity: float = 0.85,
    min_neighbor_purity: float = 0.8,
    max_area_px: Optional[int] = None,
    border_dilate_px: int = 2,
    connectivity: int = 4,
    require_no_holes: bool = True,
) -> np.ndarray:
    """
    Cari connected-component per kelas di pred_map yang bentuknya kotak
    (rectangularity tinggi) dan dikelilingi mayoritas satu kelas lain
    (neighbor purity tinggi), lalu relabel ke kelas tetangganya.

    Args:
        pred_map           : (H, W) array kelas hasil prediksi (int/uint8)
        ignore_index        : nilai piksel yang diabaikan (di luar area valid)
        min_rectangularity   : ambang luas_blob/luas_bbox untuk dianggap "kotak"
                               (1.0 = mengisi penuh bounding box-nya)
        min_neighbor_purity  : ambang proporsi tetangga yang harus didominasi
                               satu kelas supaya dianggap "salah kelas jelas"
        max_area_px          : cap luas blob (px) yang diperiksa — supaya
                               fitur besar & sengaja kotak (mis. bangunan,
                               kolam) tidak ikut "diperbaiki". None = tanpa cap.
        border_dilate_px     : lebar dilasi (px) saat mengambil sampel kelas
                               tetangga di sekitar blob
        connectivity         : 4 (atas/bawah/kiri/kanan) atau 8 (+ diagonal)
        require_no_holes      : blob HARUS solid tanpa lubang di dalamnya untuk
                               dianggap kandidat artefak. PENTING: tanpa syarat
                               ini, region besar berbentuk "bingkai/donat" yang
                               MENGELILINGI blob lain (mis. kelas latar yang
                               menyentuh semua tepi raster dan punya "lubang"
                               berisi kelas lain di tengahnya) bisa salah
                               terdeteksi sebagai "kotak" juga (bounding box-nya
                               ikut penuh sampai tepi raster) dan malah tertukar
                               posisi dengan blob asli yang mau diperbaiki.

    Returns:
        pred_map_fixed : (H, W) array baru, kelas artefak sudah direlabel
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity harus 4 atau 8")

    struct = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    pred_fixed = pred_map.copy()

    classes = [c for c in np.unique(pred_map) if c != ignore_index]

    n_fixed_blobs = 0
    for cls in classes:
        cls_mask = (pred_map == cls)
        labeled, n_blobs = ndimage.label(cls_mask, structure=struct)
        if n_blobs == 0:
            continue

        objects = ndimage.find_objects(labeled)
        for blob_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue

            blob_local = (labeled[slc] == blob_id)
            area = int(blob_local.sum())
            if max_area_px is not None and area > max_area_px:
                continue

            if require_no_holes:
                filled = ndimage.binary_fill_holes(blob_local)
                if not np.array_equal(filled, blob_local):
                    continue  # ada lubang -> bukan blob solid, skip

            bbox_area = blob_local.shape[0] * blob_local.shape[1]
            rectangularity = area / bbox_area
            if rectangularity < min_rectangularity:
                continue

            # Bangun mask global blob ini saja, lalu dilasi untuk sampling tetangga
            full_mask = np.zeros_like(cls_mask)
            full_mask[slc] = blob_local
            dilated = ndimage.binary_dilation(
                full_mask, structure=struct, iterations=border_dilate_px
            )
            border = dilated & ~full_mask

            border_vals = pred_map[border]
            border_vals = border_vals[border_vals != ignore_index]
            if border_vals.size == 0:
                continue

            vals, counts = np.unique(border_vals, return_counts=True)
            dominant_idx = int(np.argmax(counts))
            dominant_cls = vals[dominant_idx]
            purity = counts[dominant_idx] / counts.sum()

            if dominant_cls != cls and purity >= min_neighbor_purity:
                pred_fixed[full_mask] = dominant_cls
                n_fixed_blobs += 1

    if n_fixed_blobs > 0:
        print(f"    🔧 fix_tile_artifacts: {n_fixed_blobs} blob kotak "
              f"direlabel ke kelas tetangganya")

    return pred_fixed


# ============================================================
# CLI: operasi di SHP yang sudah ada (rasterize -> fix -> polygonize)
# ============================================================

def fix_tile_artifacts_shp(
    shp_path: str,
    output_shp_path: Optional[str] = None,
    class_field: str = "kelas",
    inferred_pixel_size: float = 1.0,
    min_rectangularity: float = 0.85,
    min_neighbor_purity: float = 0.8,
    max_area_px: Optional[int] = None,
    border_dilate_px: int = 2,
    connectivity: int = 4,
) -> str:
    """
    Versi standalone untuk memperbaiki SHP hasil inference yang SUDAH ada
    (tanpa run ulang model): rasterize -> fix_tile_misclassification() ->
    polygonize kembali.

    Parameter sama seperti filter_small_pixels() untuk hal rasterisasi
    (class_field, inferred_pixel_size, connectivity), plus parameter deteksi
    artefak (min_rectangularity, min_neighbor_purity, max_area_px).
    """
    import os

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"Input SHP kosong: {shp_path}")
    if gdf.crs is None:
        raise ValueError(f"Input SHP tidak memiliki CRS: {shp_path}")
    if class_field not in gdf.columns:
        raise ValueError(
            f"Kolom '{class_field}' tidak ditemukan di SHP.\n"
            f"Kolom tersedia: {list(gdf.columns)}"
        )
    if gdf.crs.is_geographic:
        raise ValueError(
            "CRS geografis terdeteksi. Reproject ke projected CRS (meter) terlebih dahulu."
        )

    print(f"    📊 Input: {len(gdf)} polygon, CRS: {gdf.crs.to_epsg()}")

    transform, width, height = _compute_grid_from_bounds(
        gdf.total_bounds, inferred_pixel_size
    )
    print(f"    🗺️  Raster grid: {height}×{width} px @ {inferred_pixel_size}m/px")

    values, label_lookup = _prepare_class_values(gdf, class_field)
    geoms = gdf.geometry.buffer(0)
    shapes_to_burn = [
        (geom, int(val))
        for geom, val in zip(geoms, values)
        if geom is not None and not geom.is_empty
    ]
    if not shapes_to_burn:
        raise ValueError("Tidak ada geometri valid untuk dirasterisasi.")

    dtype, nodata = _determine_dtype_and_nodata(values)

    from rasterio.features import rasterize
    raster = rasterize(
        shapes=shapes_to_burn,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=dtype,
    )

    raster_fixed = fix_tile_misclassification(
        raster,
        ignore_index=nodata,
        min_rectangularity=min_rectangularity,
        min_neighbor_purity=min_neighbor_purity,
        max_area_px=max_area_px,
        border_dilate_px=border_dilate_px,
        connectivity=connectivity,
    )

    records = _polygonize(raster_fixed, transform, nodata, class_field, label_lookup)
    if not records:
        raise ValueError("Tidak ada polygon yang dihasilkan setelah fix.")

    out_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)
    out_gdf = out_gdf.dissolve(by=class_field, as_index=False)
    out_gdf = out_gdf.explode(index_parts=False).reset_index(drop=True)
    print(f"    📦 Output: {len(out_gdf)} polygon")

    if output_shp_path is None:
        in_dir = os.path.dirname(os.path.abspath(shp_path))
        in_basename = os.path.basename(in_dir)
        parent_dir = os.path.dirname(in_dir)
        out_dir = os.path.join(parent_dir, f"{in_basename}_tilefix")
        os.makedirs(out_dir, exist_ok=True)
        in_stem, _ = os.path.splitext(os.path.basename(shp_path))
        output_shp_path = os.path.join(out_dir, f"{in_stem}_tilefix.shp")
    else:
        out_dir = os.path.dirname(output_shp_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    out_gdf.to_file(output_shp_path)
    print(f"    💾 Tile-fix output: {output_shp_path}")

    return output_shp_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Perbaiki blob kotak hasil salah-klasifikasi 1 tile/patch di SHP segmentasi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Path ke SHP input")
    parser.add_argument("--output", "-o", default=None,
                         help="Path ke SHP output. Default: otomatis dengan suffix _tilefix")
    parser.add_argument("--class-field", "-c", default="kelas", help="Nama kolom kelas di SHP")
    parser.add_argument("--pixel-size", "-p", type=float, default=1.0,
                         help="Ukuran pixel untuk rasterisasi (meter, sesuai GSD data)")
    parser.add_argument("--rectangularity", type=float, default=0.85,
                         help="Ambang luas_blob/luas_bbox untuk dianggap 'kotak'")
    parser.add_argument("--neighbor-purity", type=float, default=0.8,
                         help="Ambang proporsi tetangga yang harus 1 kelas dominan")
    parser.add_argument("--max-blob-px", type=int, default=None,
                         help="Cap luas blob (px) yang diperiksa. Kosongkan = tanpa cap")
    parser.add_argument("--border-dilate", type=int, default=2,
                         help="Lebar dilasi (px) saat sampling kelas tetangga")
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8],
                         help="Konektivitas connected-component: 4=4-arah, 8=8-arah")
    return parser.parse_args()


if __name__ == "__main__":
    # Bisa dijalankan berdiri sendiri (tanpa config.py) -> pastikan stdout
    # UTF-8 supaya emoji log tidak crash UnicodeEncodeError di Windows.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = _parse_args()

    fix_tile_artifacts_shp(
        shp_path=args.input,
        output_shp_path=args.output,
        class_field=args.class_field,
        inferred_pixel_size=args.pixel_size,
        min_rectangularity=args.rectangularity,
        min_neighbor_purity=args.neighbor_purity,
        max_area_px=args.max_blob_px,
        border_dilate_px=args.border_dilate,
        connectivity=args.connectivity,
    )
