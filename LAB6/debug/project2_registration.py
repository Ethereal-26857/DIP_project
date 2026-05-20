"""
LAB6 Project 2: Image Registration (Steepest Descent)
======================================================
1. Apply known rigid transformation (rotation 15 deg, translation (10, -5))
   to T1w and T2w axial slices.
2. Register transformed images back to target T1 using steepest descent.
3. Compare estimated parameters with ground truth (the true mathematical inverse).

Why NOT use MI? MI's histogram binning is non-differentiable, causing unreliable
finite-difference gradients (plateaus / discontinuities).
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform

# ============================================================
# 0. Load data
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
t1_img = nib.load(os.path.join(DATA_DIR, 'T1w.nii'))
t2_img = nib.load(os.path.join(DATA_DIR, 'T2w.nii'))
t1_data = t1_img.get_fdata().astype(np.float64)
t2_data = t2_img.get_fdata().astype(np.float64)
z_idx = t1_data.shape[2] // 2
t1_slice = t1_data[:, :, z_idx].copy()
t2_slice = t2_data[:, :, z_idx].copy()
print(f"Slice shape: {t1_slice.shape}")

def normalize(img):
    vmin, vmax = img.min(), img.max()
    return (img - vmin) / (vmax - vmin + 1e-10)

t1_norm = normalize(t1_slice)
t2_norm = normalize(t2_slice)

# ============================================================
# 1. Similarity metrics (higher-better for CC & RIU, lower-better for SSD)
# ============================================================
def masked(a, m):
    return a[m]

def ssd(a, b, mask):
    va, vb = masked(a, mask), masked(b, mask)
    return np.sum((va - vb) ** 2) / mask.sum()

def riu(a, b, mask):
    va, vb = masked(a, mask), masked(b, mask)
    eps = 1e-8
    ratio = (va + eps) / (vb + eps)
    return 1.0 - np.std(ratio) / (np.abs(np.mean(ratio)) + eps)

def cc(a, b, mask):
    va, vb = masked(a, mask), masked(b, mask)
    ma, mb = va.mean(), vb.mean()
    num = np.sum((va - ma) * (vb - mb))
    den = np.sqrt(np.sum((va - ma)**2) * np.sum((vb - mb)**2))
    return num / (den + 1e-10)

# ============================================================
# 2. Rigid 2D transformation
# ============================================================
def rigid_transform_2d(image, theta_deg, tx, ty):
    """
    Apply rigid transform (rotation about center + translation) to image.

    Forward mapping:  p' = R*(p - c) + c + t
    Inverse mapping used by affine_transform:
      p = R^T * p' + (c - R^T*(c + t))
    where p = (row, col), c = center, t = (ty, tx).
    """
    center = np.array(image.shape, dtype=np.float64) / 2.0
    theta = np.deg2rad(theta_deg)
    c_t, s_t = np.cos(theta), np.sin(theta)

    M = np.array([[c_t, s_t],
                  [-s_t, c_t]], dtype=np.float64)  # R^T
    t_vec = np.array([ty, tx], dtype=np.float64)
    offset = center - M @ (center + t_vec)

    return affine_transform(image.astype(np.float64), M, offset=offset,
                            order=1, mode='constant', cval=0.0)

# ============================================================
# 3. Compute the mathematical inverse of a transform
# ============================================================
def inverse_params(theta_deg, tx, ty):
    """Given forward transform params, return the true inverse params."""
    theta = np.deg2rad(theta_deg)
    c_t, s_t = np.cos(theta), np.sin(theta)
    R_T = np.array([[c_t, s_t], [-s_t, c_t]])
    t = np.array([ty, tx])
    t_inv = -R_T @ t
    return -theta_deg, t_inv[1], t_inv[0]

TRUE_THETA = 15.0
TRUE_TX = 10.0
TRUE_TY = -5.0
INV_THETA, INV_TX, INV_TY = inverse_params(TRUE_THETA, TRUE_TX, TRUE_TY)
print(f"\nForward transform:  theta={TRUE_THETA} deg, tx={TRUE_TX}, ty={TRUE_TY}")
print(f"True inverse:       theta={INV_THETA:.4f} deg, tx={INV_TX:.4f}, ty={INV_TY:.4f}")

# ============================================================
# 4. Gradient via central finite differences
# ============================================================
def compute_gradient(source, target, mask, params, metric_fn):
    """Gradient of metric_fn w.r.t. params = (theta, tx, ty)."""
    theta, tx, ty = params
    grad = np.zeros(3)

    d_theta = 0.5
    d_tx = 0.5
    d_ty = 0.5

    # d/dtheta
    p_plus  = rigid_transform_2d(source, theta + d_theta, tx, ty)
    p_minus = rigid_transform_2d(source, theta - d_theta, tx, ty)
    grad[0] = (metric_fn(p_plus, target, mask) - metric_fn(p_minus, target, mask)) / (2 * d_theta)

    # d/dtx
    p_plus  = rigid_transform_2d(source, theta, tx + d_tx, ty)
    p_minus = rigid_transform_2d(source, theta, tx - d_tx, ty)
    grad[1] = (metric_fn(p_plus, target, mask) - metric_fn(p_minus, target, mask)) / (2 * d_tx)

    # d/dty
    p_plus  = rigid_transform_2d(source, theta, tx, ty + d_ty)
    p_minus = rigid_transform_2d(source, theta, tx, ty - d_ty)
    grad[2] = (metric_fn(p_plus, target, mask) - metric_fn(p_minus, target, mask)) / (2 * d_ty)

    return grad

# ============================================================
# 5. Steepest descent with normalized gradient and line search
# ============================================================
def register_steepest_descent(source, target, mask, metric_fn, metric_name,
                               init_params, step_size=1.0, max_iter=200, tol=1e-8):
    """
    Steepest descent with normalized gradient direction and backtracking line search.
    Gradient normalization ensures consistent step sizes across parameters.
    """
    params = np.array(init_params, dtype=np.float64)
    history_params = [params.copy()]
    history_metric = []

    current = rigid_transform_2d(source, *params)
    init_metric = metric_fn(current, target, mask)
    print(f"  Initial: theta={params[0]:.2f} deg, tx={params[1]:.2f}, ty={params[2]:.2f}, "
          f"{metric_name}={init_metric:.6f}")

    for it in range(max_iter):
        current = rigid_transform_2d(source, *params)
        metric_val = metric_fn(current, target, mask)
        history_metric.append(metric_val)

        grad = compute_gradient(source, target, mask, params, metric_fn)
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            print(f"  Converged at iter {it+1}, |g|={gnorm:.2e}")
            break

        # Normalize gradient to unit length
        direction = grad / (gnorm + 1e-10)

        # Scale directions appropriately
        scaled_dir = direction * np.array([3.0, 2.0, 2.0])

        # Backtracking line search
        step = step_size
        found = False
        for _ in range(30):
            new_params = params - step * scaled_dir
            new_img = rigid_transform_2d(source, *new_params)
            new_metric = metric_fn(new_img, target, mask)
            if new_metric < metric_val:
                found = True
                break
            step *= 0.5

        if not found:
            step = step_size * 1e-4
            new_params = params - step * scaled_dir

        params = new_params
        history_params.append(params.copy())

        if (it + 1) % 50 == 0:
            current_metric = history_metric[-1]
            print(f"  Iter {it+1:4d}: theta={params[0]:.4f} deg, tx={params[1]:.4f}, "
                  f"ty={params[2]:.4f}, {metric_name}={current_metric:.6f}, |g|={gnorm:.2e}")

    final_metric = history_metric[-1] if history_metric else init_metric
    print(f"  Final:  theta={params[0]:.4f} deg, tx={params[1]:.4f}, ty={params[2]:.4f}, "
          f"{metric_name}={final_metric:.6f}")
    return params, history_params, history_metric

# ============================================================
# 6. Apply the known transformation
# ============================================================
t1_transformed = rigid_transform_2d(t1_norm, TRUE_THETA, TRUE_TX, TRUE_TY)
t2_transformed = rigid_transform_2d(t2_norm, TRUE_THETA, TRUE_TX, TRUE_TY)

mask = t1_norm > 0.01
print(f"Mask pixels: {mask.sum()} / {mask.size} ({100*mask.sum()/mask.size:.1f}%)")

# Verify correct inverse
recovered = rigid_transform_2d(t1_transformed, INV_THETA, INV_TX, INV_TY)
ssd_check = ssd(recovered, t1_norm, mask)
cc_check = cc(recovered, t1_norm, mask)
print(f"\nVerification at true inverse ({INV_THETA:.4f} deg, {INV_TX:.4f}, {INV_TY:.4f}):")
print(f"  SSD = {ssd_check:.8f} (should be ~0)")
print(f"  CC  = {cc_check:.8f} (should be ~1)")

# ============================================================
# 7. Registration experiments (all metrics as lower-is-better)
# ============================================================
def ssd_min(a, b, mask):   return ssd(a, b, mask)
def riu_min(a, b, mask):   return -riu(a, b, mask)    # negate for minimization
def cc_min(a, b, mask):    return -cc(a, b, mask)     # negate for minimization

experiments = [
    (t1_transformed, t1_norm, ssd_min, 'SSD',  'Intra-modality (T1->T1)', [0.0, 0.0, 0.0]),
    (t2_transformed, t1_norm, ssd_min, 'SSD',  'Inter-modality (T2->T1)', [0.0, 0.0, 0.0]),
    (t1_transformed, t1_norm, riu_min, '-RIU', 'Intra-modality (T1->T1)', [0.0, 0.0, 0.0]),
    (t2_transformed, t1_norm, riu_min, '-RIU', 'Inter-modality (T2->T1)', [0.0, 0.0, 0.0]),
    (t1_transformed, t1_norm, cc_min,  '-CC',  'Intra-modality (T1->T1)', [0.0, 0.0, 0.0]),
    (t2_transformed, t1_norm, cc_min,  '-CC',  'Inter-modality (T2->T1)', [0.0, 0.0, 0.0]),
]

all_results = {}

for source, target, metric_fn, mname, label, init in experiments:
    print(f"\n--- {label} using {mname} ---")
    opt_params, hist_p, hist_m = register_steepest_descent(
        source, target, mask, metric_fn, mname, init, step_size=2.0
    )
    key = f"{label} | {mname}"
    all_results[key] = {
        'estimated': tuple(opt_params),
        'error': (opt_params[0] - INV_THETA,
                  opt_params[1] - INV_TX,
                  opt_params[2] - INV_TY),
        'history': hist_m,
    }

# ============================================================
# 8. Results summary
# ============================================================
print(f"\n{'='*90}")
print("REGISTRATION RESULTS - Estimated vs True Inverse")
print(f"True inverse: theta={INV_THETA:.4f} deg, tx={INV_TX:.4f}, ty={INV_TY:.4f}")
print(f"{'='*90}")
print(f"{'Experiment':<40} {'theta':>8} {'tx':>8} {'ty':>8}  "
      f"{'theta_err':>9} {'tx_err':>9} {'ty_err':>9}")
print("-" * 95)
for key, res in all_results.items():
    e = res['estimated']
    er = res['error']
    print(f"{key:<40} {e[0]:>8.2f} {e[1]:>8.2f} {e[2]:>8.2f}  "
          f"{er[0]:>9.3f} {er[1]:>9.3f} {er[2]:>9.3f}")

# ============================================================
# 9. Why NOT use MI?
# ============================================================
print(f"\n{'='*60}")
print("WHY NOT USE MI WITH STEEPEST DESCENT?")
print(f"{'='*60}")
print("""
1. MI's histogram binning is non-differentiable - small parameter changes often
   map to the same histogram bins, causing zero gradients (plateau problem).
2. The finite-difference gradient becomes unreliable due to the step-function
   nature of bin assignment.
3. Each MI evaluation costs O(N * b^2), and gradient descent needs 6 evaluations
   per step, making it prohibitively slow.
4. The MI objective surface is rough with many local optima, causing gradient
   methods to get stuck easily.
5. MI is better suited for global optimization methods (Powell, Nelder-Mead,
   evolutionary algorithms) or continuous MI estimators.
""")

# ============================================================
# 10. Visualization
# ============================================================
best_intra_key = 'Intra-modality (T1->T1) | SSD'
best_inter_key = 'Inter-modality (T2->T1) | SSD'
best_intra = all_results[best_intra_key]['estimated']
best_inter = all_results[best_inter_key]['estimated']

t1_reg = rigid_transform_2d(t1_transformed, *best_intra)
t2_reg = rigid_transform_2d(t2_transformed, *best_inter)

fig, axes = plt.subplots(3, 4, figsize=(18, 13))

# Row 1: Original + transformed
axes[0, 0].imshow(t1_slice.T, cmap='gray', origin='lower')
axes[0, 0].set_title('T1w original (target)', fontsize=10); axes[0, 0].axis('off')

axes[0, 1].imshow(t2_slice.T, cmap='gray', origin='lower')
axes[0, 1].set_title('T2w original', fontsize=10); axes[0, 1].axis('off')

axes[0, 2].imshow(t1_transformed.T, cmap='gray', origin='lower')
axes[0, 2].set_title(f'T1 transformed\n(theta=15 deg, tx=10, ty=-5)', fontsize=10)
axes[0, 2].axis('off')

axes[0, 3].imshow(t2_transformed.T, cmap='gray', origin='lower')
axes[0, 3].set_title(f'T2 transformed\n(theta=15 deg, tx=10, ty=-5)', fontsize=10)
axes[0, 3].axis('off')

# Row 2: Registered + difference
axes[1, 0].imshow(t1_reg.T, cmap='gray', origin='lower')
axes[1, 0].set_title(f'Intra SSD Registered\n(theta={best_intra[0]:.2f} deg, '
                     f'tx={best_intra[1]:.2f}, ty={best_intra[2]:.2f})', fontsize=9)
axes[1, 0].axis('off')

axes[1, 1].imshow(t2_reg.T, cmap='gray', origin='lower')
axes[1, 1].set_title(f'Inter SSD Registered\n(theta={best_inter[0]:.2f} deg, '
                     f'tx={best_inter[1]:.2f}, ty={best_inter[2]:.2f})', fontsize=9)
axes[1, 1].axis('off')

axes[1, 2].imshow(np.abs(t1_reg - t1_norm).T, cmap='hot', origin='lower')
axes[1, 2].set_title('|Intra Reg - Target|', fontsize=10); axes[1, 2].axis('off')

axes[1, 3].imshow(np.abs(t2_reg - t1_norm).T, cmap='hot', origin='lower')
axes[1, 3].set_title('|Inter Reg - Target|', fontsize=10); axes[1, 3].axis('off')

# Row 3: Overlays
def make_overlay(ref, mov):
    ov = np.zeros((*ref.shape, 3))
    rn = normalize(ref); mn = normalize(mov)
    ov[:, :, 0] = rn; ov[:, :, 1] = mn
    return np.clip(ov, 0, 1)

axes[2, 0].imshow(make_overlay(t1_norm, t1_transformed), origin='lower')
axes[2, 0].set_title('Before Intra\n(R=Target, G=Transformed)', fontsize=9); axes[2, 0].axis('off')

axes[2, 1].imshow(make_overlay(t1_norm, t1_reg), origin='lower')
axes[2, 1].set_title('After Intra\n(R=Target, G=Registered)', fontsize=9); axes[2, 1].axis('off')

axes[2, 2].imshow(make_overlay(t1_norm, t2_transformed), origin='lower')
axes[2, 2].set_title('Before Inter\n(R=Target, G=Transformed)', fontsize=9); axes[2, 2].axis('off')

axes[2, 3].imshow(make_overlay(t1_norm, t2_reg), origin='lower')
axes[2, 3].set_title('After Inter\n(R=Target, G=Registered)', fontsize=9); axes[2, 3].axis('off')

plt.suptitle('LAB6 Project 2 - Image Registration with Steepest Descent', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'project2_output.png'), dpi=150)
plt.close()
print("\nFigure saved: project2_output.png")

# Convergence plots
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
colors = plt.cm.tab10.colors
for key, res in all_results.items():
    hist = res['history']
    if 'SSD' in key:
        ax = axes2[0]
    elif '-RIU' in key:
        ax = axes2[1]
    else:
        ax = axes2[2]

    # Show the actual metric value (undo negation for display)
    if '-RIU' in key:
        disp = [-h for h in hist]
        ylbl = 'RIU (higher better)'
    elif '-CC' in key:
        disp = [-h for h in hist]
        ylbl = 'CC (higher better)'
    else:
        disp = hist
        ylbl = 'SSD (lower better)'

    ax.plot(disp, alpha=0.8, linewidth=1, label=key)
    ax.set_title(ylbl)
    ax.set_xlabel('Iteration')
    ax.legend(fontsize=5, loc='best')

plt.suptitle('Convergence Curves')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'project2_convergence.png'), dpi=150)
plt.close()
print("Figure saved: project2_convergence.png")

print("\nDone!")
