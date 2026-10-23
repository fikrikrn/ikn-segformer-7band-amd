"""
check_jpg_integrity.py
=======================
Scan sebuah folder (rekursif) dan cek apakah ada file JPG yang corrupt
(gagal didekode). Berguna untuk QA foto lapangan / hasil capture drone
sebelum dipakai untuk orthophoto/orthomosaic, sebelum masuk ke pipeline
utama SegFormer.

Setiap file diverifikasi 2 lapis:
  1. Image.verify()  -> cek header/struktur file
  2. Image.load()    -> paksa full decode pixel data (paling penting,
                         karena verify() saja sering lolos untuk file
                         yang truncated di tengah data pixel)

Output: CSV berisi status tiap file (OK / CORRUPT + pesan error).

Usage:
    python data_qc/check_jpg_integrity.py <path_folder>
"""

import os
import sys
import csv
from datetime import datetime
from PIL import Image
from tqdm import tqdm


def is_jpg_ok(filepath):
    """
    Cek satu file JPG dengan validasi ketat.

    Returns:
        (True, "")              jika file OK
        (False, error_message)  jika file corrupt
    """
    try:
        if os.path.getsize(filepath) == 0:
            return False, "File 0 byte"

        # Step 1: verify struktur file
        img = Image.open(filepath)
        img.verify()

        # PENTING: harus buka ulang setelah verify(),
        # karena verify() menutup file pointer
        img = Image.open(filepath)
        img.load()  # <-- memaksa full decode, raise error kalau corrupt

        return True, ""

    except Exception as e:
        return False, str(e)


def scan_files(folder_path):
    """Kumpulkan semua path file JPG dulu (untuk total di progress bar)."""
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg')):
                file_list.append(os.path.join(root, fname))
    return file_list


def cek_folder(folder_path):
    """Jalankan pengecekan integritas untuk semua JPG di folder_path, lalu
    simpan hasilnya ke CSV bertimestamp."""
    print(f"\n=== Mengecek folder: {folder_path} ===")
    print("Sedang scan daftar file...")

    file_list = scan_files(folder_path)
    total = len(file_list)

    if total == 0:
        print("Tidak ada file JPG ditemukan.")
        return

    print(f"Ditemukan {total} file JPG. Mulai pengecekan...\n")

    hasil = []
    ok_count, corrupt_count = 0, 0

    for fpath in tqdm(file_list, desc="Mengecek file", unit="file"):
        status, err = is_jpg_ok(fpath)

        if status:
            ok_count += 1
            hasil.append([fpath, "OK", ""])
        else:
            corrupt_count += 1
            hasil.append([fpath, "CORRUPT", err])
            tqdm.write(f"  [CORRUPT] {fpath}  -> {err}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"hasil_cek_jpg_{timestamp}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "status", "error_message"])
        writer.writerows(hasil)

    print("\n=================== RINGKASAN ===================")
    print(f"Total file JPG dicek : {total}")
    print(f"File OK              : {ok_count}")
    print(f"File CORRUPT         : {corrupt_count}")
    print(f"Hasil detail disimpan: {out_csv}")
    print("==================================================\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gunakan: python data_qc/check_jpg_integrity.py <path_folder>")
        sys.exit(1)

    folder_path = sys.argv[1]
    cek_folder(folder_path)
