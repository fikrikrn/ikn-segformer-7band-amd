"""
check_tif_metadata.py
======================
Diagnostik cepat sebuah file GeoTIFF: dimensi, transform, CRS, bounds,
dtype, nodata, resolusi (GSD) sumber, dan faktor skala yang dibutuhkan
untuk resample ke resolusi target tertentu.

Berguna sebelum menjalankan preprocessing/clahe_enhancement.py atau
inference/run_inference.py, untuk memastikan CRS & resolusi citra sesuai
ekspektasi (mis. sebelum resample ke TARGET_GSD di config.py).

Usage:
    python data_qc/check_tif_metadata.py --input path/ke/citra.tif
    python data_qc/check_tif_metadata.py --input citra.tif --target-resolution 0.1
"""

import argparse
from pathlib import Path

import rasterio


def check_tif_metadata(tif_path: Path, target_resolution: float = 0.15) -> None:
    """Cetak metadata raster & hitung faktor skala resample ke target_resolution."""
    with rasterio.open(tif_path) as src:
        print(f"Width x Height  : {src.width} x {src.height}")
        print(f"Transform       : {src.transform}")
        print(f"CRS             : {src.crs}")
        print(f"Bounds          : {src.bounds}")
        print(f"Dtype           : {src.dtypes}")
        print(f"Nodata          : {src.nodata}")

        # Resolusi piksel sumber (rata-rata sumbu x & y transform)
        src_res = (abs(src.transform.a) + abs(src.transform.e)) / 2

    print(f"\nResolusi sumber (src_res) : {src_res}")
    print(f"Resolusi target           : {target_resolution}")
    print(f"Faktor skala (src/target) : "
          f"{src_res / target_resolution if src_res else 'N/A (src_res=0)'}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Cek metadata & resolusi sebuah file GeoTIFF"
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                         help="Path ke file GeoTIFF yang ingin dicek")
    parser.add_argument("--target-resolution", "-r", type=float, default=0.15,
                         help="Resolusi target (meter) untuk hitung faktor skala resample")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    check_tif_metadata(args.input, args.target_resolution)
