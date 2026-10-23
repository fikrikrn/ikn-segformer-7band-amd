"""
grid_sampling_assessment.py
=============================
Analisis Ketelitian Deteksi Tanaman (OBIA/AI vs Ground Truth), dengan
proses sampling grid LENGKAP dari nol:

  1. Buat grid 1 Ha dari extent titik Ground Truth (GT).
  2. Hitung metrik deteksi (TP/FP/FN, F1) untuk SETIAP grid yang berisi
     tanaman.
  3. Filter grid yang layak (jumlah titik GT >= MIN_POINTS_PER_GRID).
  4. Hitung ukuran sampel grid representatif secara STATISTIK (bukan
     angka manual) via rumus Cochran (1977) + Finite Population
     Correction (FPC) — lihat hitung_ukuran_sampel_cochran().
  5. Stratifikasi grid layak berdasarkan zona spasial (X, Y) x kelas
     kepadatan tanaman, lalu pilih grid representatif per strata
     (kuota proporsional, prioritas skor F1 tertinggi).
  6. Ekspor: grid semua (dengan flag terpilih), grid terpilih (untuk
     layouting peta), titik GT/AI per grid terpilih (SHP terpisah),
     dan tabel rekap (Excel, termasuk justifikasi statistik ukuran sampel).

Referensi metode sampling:
  - Cochran, W.G. (1977). Sampling Techniques, 3rd ed. Wiley.
  - Congalton & Green (2019). Assessing the Accuracy of Remotely Sensed Data.
  - Olofsson et al. (2014). Good practices for estimating area and
    assessing accuracy of land change. Remote Sensing of Environment, 148, 42-57.

PENTING: Edit bagian KONFIGURASI di bawah (path GT/AI, parameter grid,
parameter statistik) sebelum menjalankan untuk site/job baru.

Usage:
    python accuracy_assessment/grid_sampling_assessment.py
"""

import os
import math

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from scipy.spatial import cKDTree
from scipy.stats import norm

# ============================================================
# 1. KONFIGURASI — SESUAIKAN DENGAN DATA KAMU
# ============================================================
GT_SHP = r"C:\\Segformer-7band\\DATA CEK SOROWAKO\\SOROWAKO MERGED ALL.shp"
AI_SHP = r"C:\\Segformer-7band\\DATA CEK SOROWAKO\\SOROWAKO MERGED AI.shp"

OUTPUT_DIR = "output_analisis_sorowako"
CELL_SIZE_M = 100                       # 100m x 100m = 1 Ha
MATCH_RADIUS_M = 0.1                    # radius toleransi pencocokan titik GT-AI (meter)
MIN_GRID = 20                           # minimal jumlah grid sampel (peringatan saja)

MIN_POINTS_PER_GRID = 50                # minimal jumlah titik GT per grid agar grid layak dipilih
N_ZONA_X = 3                            # pembagian zona spasial arah X
N_ZONA_Y = 3                            # pembagian zona spasial arah Y
N_KELAS_DENSITAS = 3                    # jumlah kelas kepadatan (jarang/sedang/padat)

# ------------------------------------------------------------
# PARAMETER UKURAN SAMPEL (COCHRAN FORMULA + FPC)
# ------------------------------------------------------------
# Alih-alih hardcode TARGET_N_GRID, ukuran sampel dihitung otomatis
# berdasarkan jumlah grid layak (N) yang tersedia SETELAH filter
# kepadatan minimum. Ini memastikan jumlah grid representatif secara
# statistik, bukan sekadar persentase tetap dari luas area awal.
#
# Referensi:
# - Cochran, W.G. (1977). Sampling Techniques, 3rd ed. Wiley.
# - Congalton & Green (2019). Assessing the Accuracy of Remotely Sensed Data.
# - Olofsson et al. (2014). Good practices for estimating area and
#   assessing accuracy of land change. RSE.
CONFIDENCE_LEVEL = 0.95                 # tingkat kepercayaan 95%
MARGIN_ERROR = 0.05                     # margin of error 5% (bisa diperketat mis. 0.03)
P_ESTIMATE = 0.5                        # proporsi asumsi (0.5 = variabilitas maksimum/paling konservatif)

# Cap opsional dari kontrak/perjanjian awal (jika ada). Jika hasil hitung
# Cochran > TARGET_N_GRID_KONTRAK, maka yang dipakai tetap hasil Cochran
# (karena itu kebutuhan statistik minimum), TAPI jika ingin membatasi
# maksimum sesuai kontrak, set MAX_N_GRID_KONTRAK. Set None jika tidak
# ada batasan kontrak.
MAX_N_GRID_KONTRAK = None               # contoh: 250 (isi sesuai perjanjian, atau None)

# Jika kamu punya angka populasi total (misal dari luas kebun x kerapatan
# tanam), isi di sini. Jika tidak ada, biarkan None.
JUMLAH_POPULASI_TOTAL = None            # contoh: 2202804

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


# ============================================================
# 3. BUAT GRID 1 HA & FILTER YANG BERISI TANAMAN
# ============================================================
def buat_grid(gdf_gt, cell_size):
    """Buat grid bujursangkar cell_size x cell_size menutupi extent GT,
    lalu buang grid yang tidak berisi titik GT sama sekali."""
    minx, miny, maxx, maxy = gdf_gt.total_bounds
    cells = []
    x = minx
    idx = 0
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append({"id_grid": idx, "geometry": box(x, y, x + cell_size, y + cell_size)})
            idx += 1
            y += cell_size
        x += cell_size
    grid = gpd.GeoDataFrame(cells, crs=gdf_gt.crs)

    join = gpd.sjoin(gdf_gt, grid, predicate="within", how="left")
    grid_ada_tanaman = join["id_grid"].dropna().unique()
    grid_valid = grid[grid["id_grid"].isin(grid_ada_tanaman)].reset_index(drop=True)

    print(f"[INFO] Total grid kandidat : {len(grid)}")
    print(f"[INFO] Grid berisi tanaman : {len(grid_valid)}")
    if len(grid_valid) < MIN_GRID:
        print(f"[WARNING] Hanya {len(grid_valid)} grid valid, kurang dari minimum {MIN_GRID}.")
    return grid_valid


# ============================================================
# 4. MATCHING TITIK GT vs AI (PER GRID) -> TP / FP / FN
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
# 5. HITUNG METRIK UNTUK SEMUA GRID VALID
# ============================================================
def hitung_per_grid(grid, gt, ai, radius):
    """Hitung C (jumlah GT), D (jumlah AI), TP/FP/FN, dan turunannya untuk
    setiap grid, via spatial join titik -> grid lalu matching per grid."""
    gt_j = gpd.sjoin(gt, grid[["id_grid", "geometry"]], predicate="within", how="inner")
    ai_j = gpd.sjoin(ai, grid[["id_grid", "geometry"]], predicate="within", how="inner")

    gt_groups = gt_j.groupby("id_grid")
    ai_groups = ai_j.groupby("id_grid")

    print(f"[CEK] Titik GT yang terjaring ke grid: {len(gt_j)} / {len(gt)}")
    print(f"[CEK] Titik AI yang terjaring ke grid: {len(ai_j)} / {len(ai)}")

    rows = []
    for gid in grid["id_grid"]:
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
            "id_grid": gid, "C_ground_truth": C, "D_hasil_OBIA": D,
            "E_deteksi_tepat": tp, "F_ketepatan": F,
            "G_salah_deteksi": fp, "H_persen_kesalahan": H,
            "I_terlewat": fn, "J_persen_terlewat": J,
            "K_deviasi_absolut": K, "L_persen_error": L
        })

    df = pd.DataFrame(rows)
    grid_out = grid.merge(df, on="id_grid")
    return grid_out


# ============================================================
# 6. SKOR KUALITAS PER GRID (F1 per grid)
# ============================================================
def hitung_kualitas(df):
    """Hitung F1-score per grid dari TP/FP/FN — dipakai sebagai skor prioritas
    saat memilih grid representatif dalam satu strata."""
    df = df.copy()
    denom = 2 * df["E_deteksi_tepat"] + df["G_salah_deteksi"] + df["I_terlewat"]
    df["F1_grid"] = np.where(denom > 0, 2 * df["E_deteksi_tepat"] / denom, 0)
    return df


# ============================================================
# 7. STRATIFIKASI: ZONA SPASIAL & KELAS KEPADATAN
# ============================================================
def buat_label_densitas(bin_index, k_diminta):
    """Beri label kualitatif (Jarang/Sedang/Padat, dst) untuk bin kepadatan hasil qcut."""
    unique_bins = sorted(bin_index.dropna().unique())
    k = len(unique_bins)
    if k <= 1:
        return bin_index.map(lambda x: "Sedang")

    pool = {
        2: ["Jarang", "Padat"],
        3: ["Jarang", "Sedang", "Padat"],
        4: ["Sangat Jarang", "Jarang", "Padat", "Sangat Padat"],
        5: ["Sangat Jarang", "Jarang", "Sedang", "Padat", "Sangat Padat"],
    }
    names = pool.get(k, [f"Kelas {i+1}" for i in range(k)])
    label_map = {b: names[i] for i, b in enumerate(unique_bins)}
    return bin_index.map(label_map)


def buat_strata(df, n_zona_x=3, n_zona_y=3, n_densitas=3):
    """Bagi grid layak ke dalam strata (zona spasial X-Y) x (kelas kepadatan GT),
    supaya seleksi grid representatif proporsional di seluruh area, bukan
    menumpuk di satu sisi."""
    df = df.copy()
    cx = df.geometry.centroid.x
    cy = df.geometry.centroid.y

    bin_x = pd.qcut(cx, q=n_zona_x, labels=False, duplicates="drop")
    bin_y = pd.qcut(cy, q=n_zona_y, labels=False, duplicates="drop")
    df["spatial_zone"] = "Z" + bin_x.astype(str) + "-" + bin_y.astype(str)

    bin_dens = pd.qcut(df["C_ground_truth"], q=n_densitas, labels=False, duplicates="drop")
    df["density_bin"] = buat_label_densitas(bin_dens, n_densitas)

    return df


# ============================================================
# 7b. HITUNG UKURAN SAMPEL OTOMATIS (COCHRAN + FPC, 95% CI)
# ============================================================
def hitung_ukuran_sampel_cochran(N, confidence_level=0.95, margin_error=0.05, p=0.5):
    """
    Menghitung ukuran sampel minimum secara statistik menggunakan
    rumus Cochran (1977) dengan koreksi populasi terbatas (Finite
    Population Correction / FPC).

    n0     = ukuran sampel dasar (asumsi populasi tak terbatas)
    n_fpc  = ukuran sampel terkoreksi untuk populasi terbatas N

    Referensi:
      Cochran, W.G. (1977). Sampling Techniques, 3rd ed. Wiley.
      Congalton & Green (2019). Assessing the Accuracy of Remotely
        Sensed Data: Principles and Practices.
      Olofsson et al. (2014). Good practices for estimating area and
        assessing accuracy of land change. Remote Sensing of
        Environment, 148, 42-57.
    """
    Z = norm.ppf(1 - (1 - confidence_level) / 2)
    n0 = (Z ** 2 * p * (1 - p)) / (margin_error ** 2)
    n_fpc = n0 / (1 + (n0 - 1) / N)
    return {
        "Z": Z,
        "n0_tanpa_fpc": math.ceil(n0),
        "n_dengan_fpc": math.ceil(n_fpc),
        "N_populasi_grid_layak": N,
        "sampling_fraction": math.ceil(n_fpc) / N if N > 0 else 0,
    }


def tentukan_target_n_grid(N_grid_layak, confidence_level=0.95,
                            margin_error=0.05, p=0.5,
                            max_n_kontrak=None):
    """Terapkan hitung_ukuran_sampel_cochran(), lalu (opsional) batasi dengan
    cap kontrak, dan pastikan tidak melebihi jumlah grid layak yang tersedia."""
    hasil = hitung_ukuran_sampel_cochran(N_grid_layak, confidence_level, margin_error, p)
    n_statistik = hasil["n_dengan_fpc"]

    print("\n" + "=" * 60)
    print("PERHITUNGAN UKURAN SAMPEL (COCHRAN FORMULA + FPC)")
    print("=" * 60)
    print(f"Confidence level        : {confidence_level*100:.0f}%  (Z = {hasil['Z']:.4f})")
    print(f"Margin of error         : {margin_error*100:.1f}%")
    print(f"Proporsi asumsi (p)     : {p}")
    print(f"Populasi grid layak (N) : {N_grid_layak}")
    print(f"n0 (tanpa FPC)          : {hasil['n0_tanpa_fpc']}")
    print(f"n  (dengan FPC)         : {n_statistik}")
    print(f"Sampling fraction (n/N) : {hasil['sampling_fraction']*100:.1f}%")

    target_final = n_statistik
    if max_n_kontrak is not None:
        print(f"\n[INFO] Batas kontrak (max)  : {max_n_kontrak} grid")
        if n_statistik > max_n_kontrak:
            print(f"[INFO] n statistik ({n_statistik}) > batas kontrak ({max_n_kontrak})")
            print(f"[INFO] -> Dipakai batas kontrak: {max_n_kontrak} grid")
            target_final = max_n_kontrak
        else:
            print(f"[INFO] n statistik ({n_statistik}) <= batas kontrak ({max_n_kontrak})")
            print(f"[INFO] -> Dipakai hasil statistik: {n_statistik} grid "
                  f"(lebih efisien & tetap valid secara statistik)")

    if target_final > N_grid_layak:
        print(f"[WARNING] Target ({target_final}) > grid layak tersedia ({N_grid_layak}). "
              f"Dipakai semua grid layak: {N_grid_layak}")
        target_final = N_grid_layak

    print(f"\n[HASIL AKHIR] TARGET_N_GRID = {target_final} grid")
    print("=" * 60 + "\n")
    return target_final, hasil


# ============================================================
# 8. SELEKSI GRID REPRESENTATIF (KUOTA PROPORSIONAL PER STRATA)
# ============================================================
def pilih_grid_representatif(df, target_n, quality_col="F1_grid"):
    """Pilih target_n grid dengan kuota proporsional per strata (spatial_zone
    x density_bin), prioritas grid dengan skor F1 tertinggi dalam strata."""
    target_n = min(target_n, len(df))

    strata_sizes = df.groupby(["spatial_zone", "density_bin"]).size()
    total = len(df)
    quotas = (strata_sizes / total * target_n).round().astype(int)

    diff = target_n - quotas.sum()
    idx_sorted = strata_sizes.sort_values(ascending=False).index.tolist()
    i = 0
    while diff != 0 and idx_sorted:
        key = idx_sorted[i % len(idx_sorted)]
        if diff > 0:
            quotas[key] += 1
            diff -= 1
        elif quotas[key] > 0:
            quotas[key] -= 1
            diff += 1
        i += 1
        if i > 10000:
            break

    selected_list = []
    for key, quota in quotas.items():
        zona, dens = key
        sub = df[(df["spatial_zone"] == zona) & (df["density_bin"] == dens)]
        sub_sorted = sub.sort_values(quality_col, ascending=False)
        selected_list.append(sub_sorted.head(quota))

    selected = pd.concat(selected_list) if selected_list else df.iloc[0:0]

    if len(selected) < target_n:
        remaining = df.drop(selected.index)
        need = target_n - len(selected)
        topup = remaining.sort_values(quality_col, ascending=False).head(need)
        selected = pd.concat([selected, topup])
    elif len(selected) > target_n:
        selected = selected.sort_values(quality_col, ascending=False).head(target_n)

    return selected.reset_index(drop=True)


# ============================================================
# 8b. EKSTRAK TITIK GT & AI YANG BERADA DI GRID TERPILIH
# ============================================================
def ekstrak_titik_terpilih(gt, ai, grid_selected):
    """Ambil subset titik GT & AI yang berada di dalam grid_selected."""
    gt_terpilih = gpd.sjoin(
        gt, grid_selected[["id_grid", "geometry"]],
        predicate="within", how="inner"
    ).drop(columns="index_right")

    ai_terpilih = gpd.sjoin(
        ai, grid_selected[["id_grid", "geometry"]],
        predicate="within", how="inner"
    ).drop(columns="index_right")

    print(f"[INFO] Titik GT dalam grid terpilih : {len(gt_terpilih)}")
    print(f"[INFO] Titik AI dalam grid terpilih : {len(ai_terpilih)}")

    return gt_terpilih, ai_terpilih


def simpan_titik_per_grid_terpisah(gt_terpilih, ai_terpilih, output_dir):
    """Simpan titik GT & AI sebagai file SHP terpisah per id_grid, untuk
    keperluan layouting peta per grid."""
    folder_gt = os.path.join(output_dir, "titik_per_grid_GT")
    folder_ai = os.path.join(output_dir, "titik_per_grid_AI")
    os.makedirs(folder_gt, exist_ok=True)
    os.makedirs(folder_ai, exist_ok=True)

    jumlah_gt_file = 0
    for gid, sub in gt_terpilih.groupby("id_grid"):
        sub.to_file(os.path.join(folder_gt, f"GT_grid_{int(gid)}.shp"))
        jumlah_gt_file += 1

    jumlah_ai_file = 0
    for gid, sub in ai_terpilih.groupby("id_grid"):
        sub.to_file(os.path.join(folder_ai, f"AI_grid_{int(gid)}.shp"))
        jumlah_ai_file += 1

    print(f"[INFO] {jumlah_gt_file} file SHP titik GT disimpan di '{folder_gt}'")
    print(f"[INFO] {jumlah_ai_file} file SHP titik AI disimpan di '{folder_ai}'")


# ============================================================
# 9. TABEL REKAP (dihitung dari GRID TERPILIH)
# ============================================================
def buat_tabel_rekap(df, info_sampel=None):
    """Susun 4 tabel rekap: cakupan sampel, agregat TP/FP/FN & metrik,
    indikator ketelitian per Ha, dan justifikasi statistik ukuran sampel."""
    jumlah_grid = len(df)
    jumlah_sampel = df["C_ground_truth"].sum()
    if JUMLAH_POPULASI_TOTAL:
        persen_sampel = jumlah_sampel / JUMLAH_POPULASI_TOTAL * 100
    else:
        persen_sampel = None

    tabel1 = pd.DataFrame({
        "Indikator": ["Jumlah grid 1 Ha", "Jumlah sampel tanaman merica",
                      "Jumlah populasi tanaman merica", "Persentase sampel terhadap populasi"],
        "Nilai": [jumlah_grid, jumlah_sampel,
                  JUMLAH_POPULASI_TOTAL if JUMLAH_POPULASI_TOTAL else "N/A",
                  f"{persen_sampel:.1f}%" if persen_sampel else "N/A"]
    })

    total_gt = df["C_ground_truth"].sum()
    total_ai = df["D_hasil_OBIA"].sum()
    total_tp = df["E_deteksi_tepat"].sum()
    total_fp = df["G_salah_deteksi"].sum()
    total_fn = df["I_terlewat"].sum()

    recall = total_tp / total_gt if total_gt > 0 else 0
    precision = total_tp / total_ai if total_ai > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

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
                      "Rata-rata kesalahan deteksi (tanaman/ha)",
                      "Rata-rata keterlewatan deteksi (tanaman/ha)",
                      "MAPE", "MAE (tanaman/ha)"],
        "Nilai": [f"{df['F_ketepatan'].mean()*100:.1f}%",
                  f"{df['F_ketepatan'].std()*100:.1f}%",
                  f"{df['G_salah_deteksi'].mean():.0f}",
                  f"{df['I_terlewat'].mean():.0f}",
                  f"{df['L_persen_error'].mean()*100:.1f}%",
                  f"{df['K_deviasi_absolut'].mean():.0f}"]
    })

    tabel4 = None
    if info_sampel is not None:
        tabel4 = pd.DataFrame({
            "Indikator": [
                "Metode penentuan ukuran sampel",
                "Confidence level",
                "Margin of error",
                "Proporsi asumsi (p)",
                "Z-score",
                "Populasi grid layak (N)",
                "n0 (tanpa Finite Population Correction)",
                "n (dengan Finite Population Correction)",
                "Sampling fraction (n/N)",
                "Jumlah grid terpilih (aktual dipakai)",
            ],
            "Nilai": [
                "Cochran Formula + Finite Population Correction (FPC)",
                f"{CONFIDENCE_LEVEL*100:.0f}%",
                f"{MARGIN_ERROR*100:.1f}%",
                f"{P_ESTIMATE}",
                f"{info_sampel['Z']:.4f}",
                info_sampel["N_populasi_grid_layak"],
                info_sampel["n0_tanpa_fpc"],
                info_sampel["n_dengan_fpc"],
                f"{info_sampel['sampling_fraction']*100:.1f}%",
                jumlah_grid,
            ]
        })

    return tabel1, tabel2, tabel3, tabel4


# ============================================================
# 10. MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    print("=== 1. Memuat data titik ===")
    gt = load_points(GT_SHP)
    ai = load_points(AI_SHP)
    if ai.crs != gt.crs:
        ai = ai.to_crs(gt.crs)
    print(f"Jumlah titik GT : {len(gt)} | Jumlah titik AI: {len(ai)}")

    print("\n=== 2. Membuat grid 1 Ha ===")
    grid_valid = buat_grid(gt, CELL_SIZE_M)

    print("\n=== 3. Menghitung metrik untuk SEMUA grid valid ===")
    grid_all = hitung_per_grid(grid_valid, gt, ai, MATCH_RADIUS_M)
    grid_all = hitung_kualitas(grid_all)

    print(f"\n=== 3b. Filter grid dengan minimal {MIN_POINTS_PER_GRID} titik GT ===")
    grid_layak = grid_all[grid_all["C_ground_truth"] >= MIN_POINTS_PER_GRID].reset_index(drop=True)
    print(f"[INFO] Grid layak (>= {MIN_POINTS_PER_GRID} titik GT): {len(grid_layak)} dari {len(grid_all)} grid valid")

    # ------------------------------------------------------------
    # PENENTUAN TARGET_N_GRID SECARA STATISTIK (BUKAN HARDCODE)
    # ------------------------------------------------------------
    N_grid_layak = len(grid_layak)
    TARGET_N_GRID, info_sampel = tentukan_target_n_grid(
        N_grid_layak,
        confidence_level=CONFIDENCE_LEVEL,
        margin_error=MARGIN_ERROR,
        p=P_ESTIMATE,
        max_n_kontrak=MAX_N_GRID_KONTRAK
    )

    print("\n=== 4. Stratifikasi (zona spasial x kelas kepadatan) ===")
    grid_layak = buat_strata(grid_layak, N_ZONA_X, N_ZONA_Y, N_KELAS_DENSITAS)
    print(grid_layak.groupby(["spatial_zone", "density_bin"]).size().unstack(fill_value=0))

    print(f"\n=== 5. Memilih {TARGET_N_GRID} grid representatif (hasil Cochran+FPC) ===")
    grid_selected = pilih_grid_representatif(grid_layak, TARGET_N_GRID)
    print(f"[INFO] Total grid terpilih: {len(grid_selected)}")
    print("\n[Distribusi grid terpilih]")
    print(grid_selected.groupby(["spatial_zone", "density_bin"]).size().unstack(fill_value=0))

    # tandai grid mana yang terpilih pada grid_all (untuk referensi/QC)
    grid_all["terpilih"] = grid_all["id_grid"].isin(grid_selected["id_grid"])

    print("\n=== 6. Menyusun tabel rekap (berdasarkan grid terpilih) ===")
    tabel1, tabel2, tabel3, tabel4 = buat_tabel_rekap(grid_selected, info_sampel)

    print("\n--- TABEL 1: Cakupan Sampel ---")
    print(tabel1.to_string(index=False))
    print("\n--- TABEL 2: Rekapitulasi Agregat ---")
    print(tabel2.to_string(index=False))
    print("\n--- TABEL 3: Indikator Ketelitian per Ha ---")
    print(tabel3.to_string(index=False))
    print("\n--- TABEL 4: Justifikasi Statistik Ukuran Sampel (Cochran+FPC) ---")
    print(tabel4.to_string(index=False))

    print("\n=== 6b. Mengekstrak titik GT & AI pada grid terpilih ===")
    gt_terpilih, ai_terpilih = ekstrak_titik_terpilih(gt, ai, grid_selected)

    print("\n=== 7. Menyimpan output ===")
    # grid semua (valid) dengan flag terpilih -> untuk QC/referensi
    grid_all.to_file(os.path.join(OUTPUT_DIR, "grid_semua_valid.gpkg"), driver="GPKG")

    # grid terpilih -> untuk layouting peta
    grid_selected.to_file(os.path.join(OUTPUT_DIR, "grid_terpilih_layout.shp"))
    grid_selected.to_file(os.path.join(OUTPUT_DIR, "grid_terpilih_layout.gpkg"), driver="GPKG")

    # titik GT & AI per grid, disimpan sebagai file SHP terpisah per id_grid
    simpan_titik_per_grid_terpisah(gt_terpilih, ai_terpilih, OUTPUT_DIR)

    ringkasan_strata = (
        grid_selected.groupby(["spatial_zone", "density_bin"])
        .size()
        .reset_index(name="jumlah_grid_terpilih")
    )

    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "rekap_analisis.xlsx")) as writer:
        tabel1.to_excel(writer, sheet_name="Cakupan Sampel", index=False)
        tabel2.to_excel(writer, sheet_name="Rekap Agregat", index=False)
        tabel3.to_excel(writer, sheet_name="Ketelitian per Ha", index=False)
        tabel4.to_excel(writer, sheet_name="Justifikasi Ukuran Sampel", index=False)
        grid_selected.drop(columns="geometry").to_excel(writer, sheet_name="Detail Grid Terpilih", index=False)
        ringkasan_strata.to_excel(writer, sheet_name="Ringkasan Seleksi", index=False)

    print(f"[SELESAI] Semua output tersimpan di folder: {OUTPUT_DIR}")
