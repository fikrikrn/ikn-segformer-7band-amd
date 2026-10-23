"""
analyze_class_distribution.py
==============================
Analisis persebaran kelas pada dataset train vs validation (hasil split
dari preprocessing/build_patches.py), untuk:

  1. Memastikan tidak ada kelas yang hilang di validation set.
  2. Mengukur rasio imbalance antar kelas -> rekomendasi CLASS_WEIGHTS.
  3. Mengecek representasi confusing pairs (Config.CONFUSING_PAIRS).
  4. Memvisualisasikan perbandingan train vs val (bar chart, pie chart,
     log-scale chart) dan menyimpannya ke outputs/logs/.

Usage:
    python data_qc/analyze_class_distribution.py
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config


def load_patch_mask_paths(config=Config):
    """Load semua path patch dan mask yang berpasangan (nama file sama)."""
    patches_dir = config.PATCHES_DIR
    masks_dir   = config.MASKS_DIR

    if not patches_dir.exists():
        raise FileNotFoundError(f"PATCHES_DIR tidak ditemukan: {patches_dir}")

    all_patch_files = sorted(patches_dir.rglob("*.npy"))

    if not all_patch_files:
        raise FileNotFoundError(f"Tidak ada file .npy di {patches_dir}")

    patch_paths = []
    mask_paths  = []

    for patch_f in all_patch_files:
        relative  = patch_f.relative_to(patches_dir)
        mask_f    = masks_dir / relative

        if mask_f.exists():
            patch_paths.append(patch_f)
            mask_paths.append(mask_f)

    return patch_paths, mask_paths


def analyze_class_distribution(mask_paths, split_name="Dataset", config=Config):
    """
    Analisis persebaran kelas dari list mask paths.

    Returns:
        dict: {class_id: pixel_count}
        int: total_pixels
    """
    print(f"\n📊 Analyzing {split_name}...")

    class_counts = Counter()
    total_pixels = 0

    for mask_path in tqdm(mask_paths, desc=f"  Processing"):
        mask = np.load(mask_path)

        # Hitung pixel per kelas
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls != config.IGNORE_INDEX:  # Skip ignore_index
                class_counts[cls] += count
                total_pixels += count

    return dict(class_counts), total_pixels


def print_distribution_table(class_counts, total_pixels, split_name, config=Config):
    """Print tabel distribusi kelas."""
    print(f"\n{'='*80}")
    print(f"📊 {split_name.upper()} SET - Class Distribution")
    print(f"{'='*80}")
    print(f"{'Class':<8} {'Name':<20} {'Pixels':<18} {'Percentage':<12} {'Bar'}")
    print(f"{'-'*80}")

    for cls in range(config.NUM_CLASSES):
        count = class_counts.get(cls, 0)
        pct = (count / total_pixels * 100) if total_pixels > 0 else 0
        name = config.CLASS_NAMES.get(cls, f"Class {cls}")

        # Bar visualization (max 30 chars)
        bar_length = int(pct / 100 * 30)
        bar = '█' * bar_length

        print(f"{cls:<8} {name:<20} {count:<18,} {pct:>6.2f}%     {bar}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<8} {'':<20} {total_pixels:<18,} {100.0:>6.2f}%")
    print(f"{'='*80}")


def plot_distribution_comparison(train_counts, val_counts, train_total, val_total, config=Config):
    """Plot perbandingan distribusi train vs val (bar %, pie train, pie val, log-scale)."""
    classes = list(range(config.NUM_CLASSES))
    class_names = [config.CLASS_NAMES.get(c, f"C{c}") for c in classes]

    # Hitung persentase
    train_pct = [(train_counts.get(c, 0) / train_total * 100) if train_total > 0 else 0
                 for c in classes]
    val_pct = [(val_counts.get(c, 0) / val_total * 100) if val_total > 0 else 0
               for c in classes]

    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Train vs Validation - Class Distribution Analysis',
                 fontsize=18, fontweight='bold', y=0.995)

    # 1. Bar chart comparison (percentage)
    ax1 = axes[0, 0]
    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, train_pct, width, label='Train',
                    alpha=0.85, color='steelblue', edgecolor='navy', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, val_pct, width, label='Val',
                    alpha=0.85, color='coral', edgecolor='darkred', linewidth=1.5)

    ax1.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Class Distribution Comparison (Percentage)', fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in classes], fontsize=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(max(train_pct), max(val_pct)) * 1.15)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only show if > 1%
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # 2. Train pie chart
    ax2 = axes[0, 1]
    train_values = [train_counts.get(c, 0) for c in classes]
    colors = [np.array(config.CLASS_COLORS[c])/255.0 for c in classes]

    # Filter out zero values for pie
    non_zero_train = [(v, c, n) for v, c, n in zip(train_values, colors, class_names) if v > 0]
    if non_zero_train:
        values, colors_filtered, names = zip(*non_zero_train)
        wedges, texts, autotexts = ax2.pie(
            values,
            labels=[f"{n}" for n in names],
            autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
            colors=colors_filtered,
            startangle=90,
            textprops={'fontsize': 9, 'weight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
    ax2.set_title('Train Set Distribution', fontweight='bold', fontsize=13)

    # 3. Val pie chart
    ax3 = axes[1, 0]
    val_values = [val_counts.get(c, 0) for c in classes]

    non_zero_val = [(v, c, n) for v, c, n in zip(val_values, colors, class_names) if v > 0]
    if non_zero_val:
        values, colors_filtered, names = zip(*non_zero_val)
        wedges, texts, autotexts = ax3.pie(
            values,
            labels=[f"{n}" for n in names],
            autopct=lambda pct: f'{pct:.1f}%' if pct > 2 else '',
            colors=colors_filtered,
            startangle=90,
            textprops={'fontsize': 9, 'weight': 'bold'}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
    ax3.set_title('Validation Set Distribution', fontweight='bold', fontsize=13)

    # 4. Log-scale comparison (untuk class imbalance)
    ax4 = axes[1, 1]

    train_log = [np.log10(train_counts.get(c, 1)) for c in classes]
    val_log = [np.log10(val_counts.get(c, 1)) for c in classes]

    bars1 = ax4.bar(x - width/2, train_log, width, label='Train',
                    alpha=0.85, color='steelblue', edgecolor='navy', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, val_log, width, label='Val',
                    alpha=0.85, color='coral', edgecolor='darkred', linewidth=1.5)

    ax4.set_xlabel('Class', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Log₁₀(Pixel Count)', fontweight='bold', fontsize=12)
    ax4.set_title('Class Distribution (Log Scale)', fontweight='bold', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"C{c}" for c in classes], fontsize=10)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    output_path = config.LOG_DIR / "class_distribution_comparison.png"
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Plot saved: {output_path}")

    plt.show()


def calculate_class_weights(class_counts, total_pixels, method='inverse', config=Config):
    """
    Hitung class weights untuk loss function dari hasil analisis distribusi.

    Args:
        class_counts: dict {class_id: pixel_count}
        total_pixels: total pixel count
        method: 'inverse' atau 'effective'

    Returns:
        list: class weights (dinormalisasi, max = 1.0)
    """
    weights = []

    for cls in range(config.NUM_CLASSES):
        count = class_counts.get(cls, 0)

        if count == 0:
            weights.append(0.0)
        else:
            if method == 'inverse':
                # Inverse frequency
                weight = total_pixels / (config.NUM_CLASSES * count)
            elif method == 'effective':
                # Effective number of samples (Class-Balanced Loss)
                beta = 0.9999
                effective_num = 1.0 - np.power(beta, count)
                weight = (1.0 - beta) / effective_num
            else:
                weight = 1.0

            weights.append(weight)

    # Normalize weights (max = 1.0)
    max_weight = max(weights) if weights else 1.0
    if max_weight > 0:
        weights = [w / max_weight for w in weights]

    return weights


def print_class_weights(weights, config=Config):
    """Print class weights (informasi saja — training menghitung otomatis
    bila Config.CLASS_WEIGHTS = "auto")."""
    print(f"\n{'='*80}")
    print("⚖️  RECOMMENDED CLASS WEIGHTS (Inverse Frequency, Normalized)")
    print(f"{'='*80}")
    print("\nℹ️  Informasi saja: dengan CLASS_WEIGHTS = \"auto\" di config.py, bobot\n    dihitung otomatis saat training (tidak perlu copy-paste).\n")
    print("CLASS_WEIGHTS = [")
    for cls, weight in enumerate(weights):
        name = config.CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"    {weight:.10f},  # Class {cls}: {name}")
    print("]\n")
    print(f"{'='*80}")


def check_confusing_pairs(train_counts, val_counts, config=Config):
    """Cek apakah confusing pairs (Config.CONFUSING_PAIRS) punya representasi yang cukup."""
    print(f"\n{'='*80}")
    print("🔍 CONFUSING PAIRS ANALYSIS")
    print(f"{'='*80}")

    if not config.CONFUSING_PAIRS:
        print("  ℹ️  No confusing pairs defined in config.CONFUSING_PAIRS")
        print(f"{'='*80}")
        return

    for cls1, cls2 in config.CONFUSING_PAIRS:
        name1 = config.CLASS_NAMES.get(cls1, f"Class {cls1}")
        name2 = config.CLASS_NAMES.get(cls2, f"Class {cls2}")

        train1 = train_counts.get(cls1, 0)
        train2 = train_counts.get(cls2, 0)
        val1 = val_counts.get(cls1, 0)
        val2 = val_counts.get(cls2, 0)

        print(f"\n  📌 Pair: {name1} (C{cls1}) ↔ {name2} (C{cls2})")
        print(f"     Train: {train1:>12,} vs {train2:>12,} pixels")
        print(f"     Val  : {val1:>12,} vs {val2:>12,} pixels")

        # Check imbalance
        if train1 > 0 and train2 > 0:
            ratio = max(train1, train2) / min(train1, train2)
            if ratio > 10:
                print(f"     ⚠️  High imbalance ratio: {ratio:.1f}x (consider oversampling)")
            elif ratio > 5:
                print(f"     ⚠️  Moderate imbalance ratio: {ratio:.1f}x")
            else:
                print(f"     ✅ Balanced (ratio: {ratio:.1f}x)")
        else:
            print(f"     ❌ One or both classes have 0 pixels!")

    print(f"\n{'='*80}")


def print_summary_statistics(train_counts, val_counts, train_total, val_total, config=Config):
    """Print summary statistics train/val: imbalance ratio, kelas hilang, konsistensi distribusi."""
    print(f"\n{'='*80}")
    print("📈 SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Train statistics
    print(f"\n  🚂 TRAIN SET:")
    print(f"     Total pixels       : {train_total:>15,}")

    train_values = [c for c in train_counts.values() if c > 0]
    if train_values:
        max_count = max(train_values)
        min_count = min(train_values)
        mean_count = np.mean(train_values)
        std_count = np.std(train_values)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"     Max class pixels   : {max_count:>15,}")
        print(f"     Min class pixels   : {min_count:>15,}")
        print(f"     Mean class pixels  : {mean_count:>15,.0f}")
        print(f"     Std class pixels   : {std_count:>15,.0f}")
        print(f"     Imbalance ratio    : {imbalance_ratio:>15.1f}x")

        if imbalance_ratio > 100:
            print(f"     ⚠️  SEVERE imbalance! Strongly recommend class weights + Focal Loss")
        elif imbalance_ratio > 10:
            print(f"     ⚠️  Moderate imbalance, class weights recommended")
        else:
            print(f"     ✅ Relatively balanced")

    # Val statistics
    print(f"\n  🔍 VALIDATION SET:")
    print(f"     Total pixels       : {val_total:>15,}")

    # Check if val has all classes
    train_classes = set(train_counts.keys())
    val_classes = set(val_counts.keys())
    missing_in_val = train_classes - val_classes

    if missing_in_val:
        print(f"     ⚠️  Missing classes in val: {sorted(missing_in_val)}")
        print(f"        → Model cannot be validated for these classes!")
    else:
        print(f"     ✅ All train classes present in validation")

    # Check distribution consistency
    print(f"\n  📊 DISTRIBUTION CONSISTENCY (Train vs Val):")
    max_diff = 0
    for cls in range(config.NUM_CLASSES):
        train_pct = (train_counts.get(cls, 0) / train_total * 100) if train_total > 0 else 0
        val_pct = (val_counts.get(cls, 0) / val_total * 100) if val_total > 0 else 0
        diff = abs(train_pct - val_pct)
        max_diff = max(max_diff, diff)

    print(f"     Max percentage diff: {max_diff:>14.2f}%")
    if max_diff > 5:
        print(f"     ⚠️  Large distribution difference between train and val")
        print(f"        → Consider stratified split or check data preprocessing")
    else:
        print(f"     ✅ Train and val distributions are consistent")

    print(f"\n{'='*80}")


def main():
    """Entry point: load data -> split train/val -> analisis -> plot -> rekomendasi."""
    config = Config

    print("\n" + "="*80)
    print("🔍 CLASS DISTRIBUTION CHECKER FOR SEGMENTATION DATASET")
    print("="*80)

    # Load paths
    print("\n📂 Loading patch and mask paths...")
    try:
        patch_paths, mask_paths = load_patch_mask_paths(config)
        print(f"   ✅ Total patches found: {len(patch_paths)}")
    except FileNotFoundError as e:
        print(f"   ❌ Error: {e}")
        return

    if len(patch_paths) == 0:
        print("   ❌ No patches found! Check your PATCHES_DIR in config.py")
        return

    # Split train/val (harus identik dengan split di training/train_model.py,
    # yaitu random_state = config.RANDOM_SEED, supaya analisis ini merefleksikan
    # split yang benar-benar dipakai saat training)
    print(f"\n✂️  Splitting dataset...")
    print(f"   Train ratio: {config.TRAIN_SPLIT:.1%}")
    print(f"   Val ratio  : {config.VAL_SPLIT:.1%}")

    train_p, val_p, train_m, val_m = train_test_split(
        patch_paths, mask_paths,
        test_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    print(f"   ✅ Train patches: {len(train_p)}")
    print(f"   ✅ Val patches  : {len(val_p)}")

    # Analyze train
    train_counts, train_total = analyze_class_distribution(train_m, "TRAIN", config)
    print_distribution_table(train_counts, train_total, "TRAIN", config)

    # Analyze val
    val_counts, val_total = analyze_class_distribution(val_m, "VALIDATION", config)
    print_distribution_table(val_counts, val_total, "VALIDATION", config)

    # Calculate class weights
    print(f"\n⚖️  Calculating class weights from TRAIN set...")
    weights_inverse = calculate_class_weights(train_counts, train_total, method='inverse', config=config)
    print_class_weights(weights_inverse, config)

    # Check confusing pairs
    check_confusing_pairs(train_counts, val_counts, config)

    # Summary statistics
    print_summary_statistics(train_counts, val_counts, train_total, val_total, config)

    # Plot comparison
    print(f"\n📊 Generating visualization plots...")
    try:
        plot_distribution_comparison(train_counts, val_counts, train_total, val_total, config)
        print(f"   ✅ Plots generated successfully")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not generate plots: {e}")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📝 Next steps:")
    print("   1. CLASS_WEIGHTS dihitung otomatis saat training (CLASS_WEIGHTS = \"auto\")")
    print("   2. Check if all classes present in validation set")
    print("   3. Review confusing pairs analysis")
    print("   4. Check distribution consistency between train and val")
    print("   5. Review saved plot: outputs/logs/class_distribution_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
