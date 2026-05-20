"""
LAB6 Project 1: Similarity Measure
===================================
Extract an axial slice from T1w and T2w, then compute similarity metrics:
  - SSD  (Sum of Squared Differences)
  - RIU  (Ratio Image Uniformity)
  - CC   (Correlation Coefficient)
  - MI   (Mutual Information)

All functions are written manually and compared with standard libraries
(except RIU, which has no standard implementation).
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score

# ============================================================
# 0. Load data & extract axial slice
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
t1_img = nib.load(os.path.join(DATA_DIR, 'T1w.nii'))
t2_img = nib.load(os.path.join(DATA_DIR, 'T2w.nii'))

t1_data = t1_img.get_fdata().astype(np.float64)
t2_data = t2_img.get_fdata().astype(np.float64)

# Take the middle axial slice (z-axis is the last axis)
z_idx = t1_data.shape[2] // 2  # 104
t1_slice = t1_data[:, :, z_idx]
t2_slice = t2_data[:, :, z_idx]

print(f"T1w shape: {t1_data.shape}, T2w shape: {t2_data.shape}")
print(f"Using axial slice z={z_idx}")
print(f"T1 slice range: [{t1_slice.min():.2f}, {t1_slice.max():.2f}]")
print(f"T2 slice range: [{t2_slice.min():.2f}, {t2_slice.max():.2f}]")

# ============================================================
# 1. Mask generation (threshold-based, exclude background)
# ============================================================
# Use Otsu-like simple threshold: >1% of max to exclude air/background
t1_thresh = t1_slice.max() * 0.01
t2_thresh = t2_slice.max() * 0.01
mask = (t1_slice > t1_thresh) & (t2_slice > t2_thresh)
print(f"Mask pixels: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.1f}%)")

# Helper to flatten masked pixels
def masked_flat(a, m):
    return a[m].flatten()

# ============================================================
# 2. Similarity metrics (manual implementation)
# ============================================================

def ssd(a, b, mask):
    """Sum of Squared Differences (lower = more similar)."""
    va = masked_flat(a, mask)
    vb = masked_flat(b, mask)
    return np.sum((va - vb) ** 2)

def riu(a, b, mask):
    """Ratio Image Uniformity.
    RIU = 1 - (std(ratio) / mean(ratio)), higher = more similar.
    """
    va = masked_flat(a, mask)
    vb = masked_flat(b, mask)
    eps = 1e-10
    ratio = va / (vb + eps)
    return 1.0 - np.std(ratio) / (np.mean(np.abs(ratio)) + eps)

def cc(a, b, mask):
    """Pearson Correlation Coefficient (range [-1, 1], higher = more similar)."""
    va = masked_flat(a, mask)
    vb = masked_flat(b, mask)
    ma, mb = va.mean(), vb.mean()
    num = np.sum((va - ma) * (vb - mb))
    den = np.sqrt(np.sum((va - ma)**2) * np.sum((vb - mb)**2))
    return num / (den + 1e-10)

def mutual_information(a, b, mask, bins=256):
    """Mutual Information using histogram binning.
    MI = H(A) + H(B) - H(A,B), higher = more similar.
    """
    va = masked_flat(a, mask)
    vb = masked_flat(b, mask)

    # Normalize to [0, bins-1]
    def digitize(v, bins):
        vmin, vmax = v.min(), v.max()
        if vmax == vmin:
            return np.zeros_like(v, dtype=np.int32)
        return np.floor((v - vmin) / (vmax - vmin) * (bins - 1)).astype(np.int32)

    a_d = digitize(va, bins)
    b_d = digitize(vb, bins)

    # Joint histogram
    joint = np.zeros((bins, bins), dtype=np.float64)
    for i in range(len(a_d)):
        joint[a_d[i], b_d[i]] += 1
    joint /= joint.sum()

    # Marginals
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    # Entropies
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    ha = entropy(pa)
    hb = entropy(pb)
    hab = entropy(joint.flatten())
    return ha + hb - hab

# ============================================================
# 3. Standard library wrappers for comparison
# ============================================================

def cc_standard(a, b, mask):
    """Pearson correlation via scipy."""
    va = masked_flat(a, mask)
    vb = masked_flat(b, mask)
    r, _ = pearsonr(va, vb)
    return r

def mi_standard(a, b, mask, bins=256):
    """Mutual information via sklearn (uses joint entropy approach)."""
    va = masked_flat(a, mask)
    vb = masked_flat(b, mask)

    # Digitize
    vmin_a, vmax_a = va.min(), va.max()
    vmin_b, vmax_b = vb.min(), vb.max()
    a_d = np.floor((va - vmin_a) / (vmax_a - vmin_a + 1e-10) * (bins - 1)).astype(int)
    b_d = np.floor((vb - vmin_b) / (vmax_b - vmin_b + 1e-10) * (bins - 1)).astype(int)

    mi = mutual_info_score(a_d, b_d)
    return mi

# ============================================================
# 4. Compute and compare
# ============================================================

print("\n" + "="*60)
print("SIMILARITY METRICS — T1w vs T2w (axial slice)")
print("="*60)

# Normalize slices to [0,1] for better comparability
t1_norm = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-10)
t2_norm = (t2_slice - t2_slice.min()) / (t2_slice.max() - t2_slice.min() + 1e-10)

results = {}

print(f"\n{'Metric':<8} {'Manual':<15} {'Library':<15}")
print("-" * 40)

# SSD
ssd_val = ssd(t1_slice, t2_slice, mask)
results['SSD'] = ssd_val
print(f"{'SSD':<8} {ssd_val:<15.4f} {'N/A':<15}")

# RIU
riu_val = riu(t1_slice, t2_slice, mask)
results['RIU'] = riu_val
print(f"{'RIU':<8} {riu_val:<15.6f} {'N/A':<15}")

# CC
cc_man = cc(t1_slice, t2_slice, mask)
cc_lib = cc_standard(t1_slice, t2_slice, mask)
results['CC (manual)'] = cc_man
results['CC (scipy)'] = cc_lib
print(f"{'CC':<8} {cc_man:<15.6f} {cc_lib:<15.6f}")

# MI
mi_man = mutual_information(t1_norm, t2_norm, mask)
mi_lib = mi_standard(t1_norm, t2_norm, mask)
results['MI (manual)'] = mi_man
results['MI (sklearn)'] = mi_lib
print(f"{'MI':<8} {mi_man:<15.6f} {mi_lib:<15.6f}")

# Also compute T1w vs itself (identity)
print(f"\n{'Metric':<8} {'T1w vs T1w (identity)':<20}")
print("-" * 30)
mask_t1 = t1_slice > t1_slice.max() * 0.01
print(f"{'SSD':<8} {ssd(t1_slice, t1_slice, mask_t1):<20.4f}")
print(f"{'RIU':<8} {riu(t1_slice, t1_slice, mask_t1):<20.6f}")
print(f"{'CC':<8} {cc(t1_slice, t1_slice, mask_t1):<20.6f}")
print(f"{'MI':<8} {mutual_information(t1_slice, t1_slice, mask_t1):<20.6f}")

# ============================================================
# 5. Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(t1_slice.T, cmap='gray', origin='lower')
axes[0, 0].set_title('T1w axial slice')
axes[0, 0].axis('off')

axes[0, 1].imshow(t2_slice.T, cmap='gray', origin='lower')
axes[0, 1].set_title('T2w axial slice')
axes[0, 1].axis('off')

axes[1, 0].imshow(mask.T, cmap='gray', origin='lower')
axes[1, 0].set_title(f'Mask (both >1% max)')
axes[1, 0].axis('off')

# Overlay T1 (red) and T2 (green) to show misalignment
overlay = np.zeros((*t1_slice.shape, 3))
overlay[:, :, 0] = t1_norm  # Red channel
overlay[:, :, 1] = t2_norm  # Green channel
overlay = np.clip(overlay, 0, 1)
axes[1, 1].imshow(overlay, origin='lower')
axes[1, 1].set_title('Overlay (R=T1, G=T2)')
axes[1, 1].axis('off')

plt.suptitle('LAB6 Project 1 — Similarity Measure', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'project1_output.png'), dpi=150)
print("\nDone! Figure saved to project1_output.png")
