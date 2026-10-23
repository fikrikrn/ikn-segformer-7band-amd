"""
final_grid_assessment.py
==========================
Versi simplified dari analisis ketelitian deteksi tanaman (OBIA/AI vs
Ground Truth): dipakai ketika grid sampel REPRESENTATIF SUDAH ADA (hasil
dari grid_sampling_assessment.py sebelumnya, atau grid final dari klien),
sehingga TIDAK perlu mengulang proses sampling/stratifikasi grid.

Alur:
  1. Load titik GT, titik AI, dan SHP grid final.
  2. Hitung metrik deteksi (TP/FP/FN, F1) untuk setiap grid pada SHP grid
     yang diberikan.
  3. Susun tabel rekap (cakupan, agregat, indikator ketelitian per grid).
  4. Simpan grid dengan metrik (SHP/GPKG) + tabel rekap (Excel).

PENTING: Edit bagian KONFIGURASI di bawah (path GT/AI/grid, nama kolom ID
grid) sebelum menjalankan untuk site/job baru.

Usage:
    python accuracy_assessment/final_grid_assessment.py
"""

import os

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ============================================================
# 1. KONFIGURASI — SESUAIKAN DENGAN DATA KAMU
# ============================================================
GT_SHP   = r"W:\2026\02 VALE 2026\00 BAHODOPI SOROWAKO\05 UJI AKURASI\OBJECT DETECTION\CEK LAST\GT_SOROWAKO.shp"
AI_SHP   = r"W:\2026\02 VALE 2026\00 BAHODOPI SOROWAKO\05 UJI AKURASI\OBJECT DETECTION\CEK LAST\REF_SOROWAKO.shp"
GRID_SHP = r"W:\2026\02 VALE 2026\00 BAHODOPI SOROWAKO\05 UJI AKURASI\OBJECT DETECTION\CEK LAST\output_analisis_sorowako\grid_terpilih_layout.shp"

OUTPUT_DIR      = "output_analisis_sorowako_GTTTTTT RIL"
MATCH_RADIUS_M  = 0.1        # radius toleransi pencocokan titik GT-AI (meter)
GRID_ID_FIELD   = "id_grid"  # <- ganti sesuai nama kolom ID unik pada grid kamu

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA & PASTIKAN CRS PROJECTED (METER)
# ============================================================
def load_points(path):
    """Load SHP titik, reproject otomatis ke UTM (meter) bila CRS-nya geografis."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"CRS tidak terdefinisi pada {path}")
    if gdf.crs.is_geographic:
        centroid = gdf.unary_union.centroid
        zone = int((centroid.x + 180) / 6) + 1
        hemisphere = 326 if centroid.y >= 0 else 327
        epsg = f"{hemisphere}{zone:02d}"
        gdf = gdf.to_crs(epsg=int(epsg))
        print(f"[INFO] {path} diproyeksikan ke EPSG:{epsg}")
    return gdf


def load_grid(path, id_field):
    """Load SHP grid final. Jika kolom id_field tidak ada, buat otomatis dari index."""
    grid = gpd.read_file(path)
    if grid.crs is None:
        raise ValueError(f"CRS tidak terdefinisi pada {path}")
    if id_field not in grid.columns:
        print(f"[INFO] Kolom '{id_field}' tidak ditemukan di grid, membuat ID otomatis dari index.")
        grid[id_field] = grid.index
    return grid


# ============================================================
# 3. MATCHING TITIK GT vs AI (PER GRID) -> TP / FP / FN
# ============================================================
def matching_grid(gt_pts, ai_pts, radius):
    """Pencocokan titik terdekat (nearest-neighbor via KD-Tree) dalam
    radius toleransi, greedy 1-ke-1: satu titik AI hanya dipakai sekali."""
    if len(gt_pts) == 0:
        return 0, len(ai_pts), 0
    if len(ai_pts) == 0:
        return 0, 0, len(gt_pts)

    gt_coords = np.array([[p.x, p.y] for p in gt_pts])
    ai_coords = np.array([[p.x, p.y] for p in ai_pts])

    tree = cKDTree(ai_coords)
    dist, idx = tree.query(gt_coords, distance_upper_bound=radius)

    matched_ai = set()
    tp = 0
    for d, i in zip(dist, idx):
        if d != np.inf and i not in matched_ai:
            matched_ai.add(i)
            tp += 1

    fn = len(gt_pts) - tp
    fp = len(ai_pts) - len(matched_ai)
    return tp, fp, fn


# ============================================================
# 4. HITUNG METRIK UNTUK SETIAP GRID
# ============================================================
def hitung_per_grid(grid, gt, ai, radius, id_field):
    """Hitung C (jumlah GT), D (jumlah AI), TP/FP/FN, dan turunannya untuk
    setiap grid, via spatial join titik -> grid lalu matching per grid."""
    # -----------------------------------------------------------
    # Bersihkan kolom yang akan di-generate ulang, supaya tidak
    # bentrok / ter-suffix _x _y saat merge (misal grid input
    # sebenarnya adalah hasil analisis sebelumnya yang dipakai lagi)
    # -----------------------------------------------------------
    kolom_stat = [
        "C_ground_truth", "D_hasil_OBIA", "E_deteksi_tepat", "F_ketepatan",
        "G_salah_deteksi", "H_persen_kesalahan", "I_terlewat",
        "J_persen_terlewat", "K_deviasi_absolut", "L_persen_error", "F1_grid"
    ]
    kolom_bentrok = [c for c in kolom_stat if c in grid.columns]
    if kolom_bentrok:
        print(f"[INFO] Menghapus kolom lama yang bentrok sebelum recalculate: {kolom_bentrok}")
        grid = grid.drop(columns=kolom_bentrok)

    gt_j = gpd.sjoin(gt, grid[[id_field, "geometry"]], predicate="within", how="inner")
    ai_j = gpd.sjoin(ai, grid[[id_field, "geometry"]], predicate="within", how="inner")

    gt_groups = gt_j.groupby(id_field)
    ai_groups = ai_j.groupby(id_field)

    print(f"[CEK] Titik GT yang terjaring ke grid: {len(gt_j)} / {len(gt)}")
    print(f"[CEK] Titik AI yang terjaring ke grid: {len(ai_j)} / {len(ai)}")

    rows = []
    for gid in grid[id_field]:
        gt_sub = gt_groups.get_group(gid).geometry.tolist() if gid in gt_groups.groups else []
        ai_sub = ai_groups.get_group(gid).geometry.tolist() if gid in ai_groups.groups else []

        C = len(gt_sub)
        D = len(ai_sub)
        tp, fp, fn = matching_grid(gt_sub, ai_sub, radius)

        F = tp / C if C > 0 else 0
        H = fp / C if C > 0 else 0
        J = fn / C if C > 0 else 0
        K = abs(C - D)
        L = K / C if C > 0 else 0

        rows.append({
            id_field: gid, "C_ground_truth": C, "D_hasil_OBIA": D,
            "E_deteksi_tepat": tp, "F_ketepatan": F,
            "G_salah_deteksi": fp, "H_persen_kesalahan": H,
            "I_terlewat": fn, "J_persen_terlewat": J,
            "K_deviasi_absolut": K, "L_persen_error": L
        })

    df = pd.DataFrame(rows)
    grid_out = grid.merge(df, on=id_field)
    return grid_out


def hitung_kualitas(df):
    """Hitung F1-score per grid dari TP/FP/FN."""
    df = df.copy()
    denom = 2 * df["E_deteksi_tepat"] + df["G_salah_deteksi"] + df["I_terlewat"]
    df["F1_grid"] = np.where(denom > 0, 2 * df["E_deteksi_tepat"] / denom, 0)
    return df


# ============================================================
# 5. TABEL REKAP
# ============================================================
def buat_tabel_rekap(df):
    """Susun 3 tabel rekap: cakupan, agregat TP/FP/FN & metrik, indikator
    ketelitian per grid."""
    jumlah_grid = len(df)
    total_gt = df["C_ground_truth"].sum()
    total_ai = df["D_hasil_OBIA"].sum()
    total_tp = df["E_deteksi_tepat"].sum()
    total_fp = df["G_salah_deteksi"].sum()
    total_fn = df["I_terlewat"].sum()

    recall = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / total_ai if total_ai > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    tabel1 = pd.DataFrame({
        "Indikator": ["Jumlah grid dianalisis", "Jumlah sampel tanaman merica (GT)"],
        "Nilai": [jumlah_grid, total_gt]
    })

    tabel2 = pd.DataFrame({
        "Indikator": ["Jumlah tanaman berdasar ground truth",
                      "Jumlah tanaman hasil deteksi berbasis OBIA",
                      "Deteksi tepat (True Positive)",
                      "Salah deteksi (False Positive)",
                      "Deteksi terlewat (False Negative)",
                      "Recall", "Precision", "F1-score"],
        "Nilai": [total_gt, total_ai, total_tp, total_fp, total_fn,
                  f"{recall*100:.1f}%", f"{precision*100:.1f}%", f"{f1*100:.1f}%"]
    })

    tabel3 = pd.DataFrame({
        "Indikator": ["Rata-rata persentase ketepatan deteksi",
                      "Standar deviasi persentase ketepatan deteksi",
                      "Rata-rata kesalahan deteksi (tanaman/grid)",
                      "Rata-rata keterlewatan deteksi (tanaman/grid)",
                      "MAPE", "MAE (tanaman/grid)"],
        "Nilai": [f"{df['F_ketepatan'].mean()*100:.1f}%",
                  f"{df['F_ketepatan'].std()*100:.1f}%",
                  f"{df['G_salah_deteksi'].mean():.0f}",
                  f"{df['I_terlewat'].mean():.0f}",
                  f"{df['L_persen_error'].mean()*100:.1f}%",
                  f"{df['K_deviasi_absolut'].mean():.0f}"]
    })

    return tabel1, tabel2, tabel3


# ============================================================
# 6. MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    print("=== 1. Memuat data titik & grid ===")
    gt = load_points(GT_SHP)
    ai = load_points(AI_SHP)
    if ai.crs != gt.crs:
        ai = ai.to_crs(gt.crs)

    grid = load_grid(GRID_SHP, GRID_ID_FIELD)
    if grid.crs != gt.crs:
        grid = grid.to_crs(gt.crs)

    print(f"Jumlah titik GT : {len(gt)} | Jumlah titik AI: {len(ai)} | Jumlah grid: {len(grid)}")

    print("\n=== 2. Menghitung metrik per grid ===")
    grid_hasil = hitung_per_grid(grid, gt, ai, MATCH_RADIUS_M, GRID_ID_FIELD)
    grid_hasil = hitung_kualitas(grid_hasil)

    print("\n=== 3. Menyusun tabel rekap ===")
    tabel1, tabel2, tabel3 = buat_tabel_rekap(grid_hasil)

    print("\n--- TABEL 1: Cakupan ---")
    print(tabel1.to_string(index=False))
    print("\n--- TABEL 2: Rekapitulasi Agregat ---")
    print(tabel2.to_string(index=False))
    print("\n--- TABEL 3: Indikator Ketelitian per Grid ---")
    print(tabel3.to_string(index=False))

    print("\n=== 4. Menyimpan output ===")
    grid_hasil.to_file(os.path.join(OUTPUT_DIR, "grid_hasil_analisis.gpkg"), driver="GPKG")
    grid_hasil.to_file(os.path.join(OUTPUT_DIR, "grid_hasil_analisis.shp"))

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "rekap_analisis.xlsx")) as writer:
        tabel1.to_excel(writer, sheet_name="Cakupan", index=False)
        tabel2.to_excel(writer, sheet_name="Rekap Agregat", index=False)
        tabel3.to_excel(writer, sheet_name="Ketelitian per Grid", index=False)
        grid_hasil.drop(columns="geometry").to_excel(writer, sheet_name="Detail Grid", index=False)

    print(f"[SELESAI] Semua output tersimpan di folder: {OUTPUT_DIR}")
