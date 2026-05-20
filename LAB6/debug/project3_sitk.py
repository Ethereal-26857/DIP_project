"""
LAB6 Project 3: Tool-based Co-registration (SimpleITK)
========================================================
Use SimpleITK to complete Project 2 tasks:
  - Intra-modality registration (T1w_transformed -> T1w)
  - Inter-modality registration (T2w_transformed -> T1w)
  - Try different similarity metrics
  - Compare with steepest descent results from Project 2
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
import SimpleITK as sitk

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

def normalize(img):
    vmin, vmax = img.min(), img.max()
    return (img - vmin) / (vmax - vmin + 1e-10)

t1_norm = normalize(t1_slice)
t2_norm = normalize(t2_slice)

# ============================================================
# 1. NumPy <-> SimpleITK conversion
# ============================================================
def numpy_to_sitk(arr):
    """Convert 2D numpy (rows, cols) to SimpleITK image."""
    return sitk.GetImageFromArray(arr.T)

def sitk_to_numpy(img):
    """Convert SimpleITK image to 2D numpy (rows, cols)."""
    return sitk.GetArrayFromImage(img).T

# ============================================================
# 2. True inverse transform computation
# ============================================================
TRUE_THETA = 15.0
TRUE_TX = 10.0     # SimpleITK: x-direction
TRUE_TY = -5.0     # SimpleITK: y-direction

def inverse_params_sitk(theta_deg, tx, ty):
    """Compute inverse params in SimpleITK (x, y) coordinate convention."""
    theta = np.deg2rad(theta_deg)
    c_t, s_t = np.cos(theta), np.sin(theta)
    # R(-theta)^T = R(theta)
    R_neg = np.array([[c_t, s_t], [-s_t, c_t]])  # R(-theta)
    t = np.array([tx, ty])
    t_inv = -R_neg @ t
    return -theta_deg, t_inv[0], t_inv[1]

INV_THETA_SITK, INV_TX_SITK, INV_TY_SITK = inverse_params_sitk(TRUE_THETA, TRUE_TX, TRUE_TY)
print(f"Forward transform: theta={TRUE_THETA} deg, tx={TRUE_TX}, ty={TRUE_TY} (SimpleITK coords)")
print(f"True inverse:      theta={INV_THETA_SITK:.4f} deg, tx={INV_TX_SITK:.4f}, ty={INV_TY_SITK:.4f}")

# ============================================================
# 3. Apply known transformation using SimpleITK
# ============================================================
def apply_rigid_transform_sitk(image_2d, theta_deg, tx, ty):
    """Apply rigid 2D transform using SimpleITK Euler2DTransform."""
    img = numpy_to_sitk(image_2d)
    img = sitk.Cast(img, sitk.sitkFloat32)

    # Use the SimpleITK image's own size to set the center correctly
    sz = img.GetSize()
    center = [sz[0] / 2.0, sz[1] / 2.0]

    transform = sitk.Euler2DTransform()
    transform.SetCenter(center)
    transform.SetAngle(np.deg2rad(theta_deg))
    transform.SetTranslation((tx, ty))

    transformed = sitk.Resample(img, img, transform, sitk.sitkLinear, 0.0,
                                img.GetPixelID())
    return sitk_to_numpy(transformed)

# Apply forward transform
t1_transformed = apply_rigid_transform_sitk(t1_norm, TRUE_THETA, TRUE_TX, TRUE_TY)
t2_transformed = apply_rigid_transform_sitk(t2_norm, TRUE_THETA, TRUE_TX, TRUE_TY)
print(f"Images prepared. T1 transformed shape: {t1_transformed.shape}")

# ============================================================
# 4. Registration functions using SimpleITK
# ============================================================
def _setup_registration(fixed_np, moving_np, metric_name):
    """Common setup for SimpleITK registration."""
    fixed = sitk.Cast(numpy_to_sitk(fixed_np), sitk.sitkFloat32)
    moving = sitk.Cast(numpy_to_sitk(moving_np), sitk.sitkFloat32)

    sz = fixed.GetSize()
    center = [sz[0] / 2.0, sz[1] / 2.0]

    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetCenter(center)

    registration = sitk.ImageRegistrationMethod()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(initial_transform, inPlace=True)

    if metric_name == 'MeanSquares':
        registration.SetMetricAsMeanSquares()
    elif metric_name == 'Correlation':
        registration.SetMetricAsCorrelation()
    elif metric_name == 'MattesMutualInformation':
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric_name == 'JointHistogramMutualInformation':
        registration.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    return registration, fixed, moving


def register_sitk_lbfgsb(fixed_np, moving_np, metric_name='MeanSquares'):
    """Register using LBFGSB optimizer (quasi-Newton, most reliable)."""
    registration, fixed, moving = _setup_registration(fixed_np, moving_np, metric_name)
    registration.SetOptimizerAsLBFGSB(
        numberOfIterations=200,
        gradientConvergenceTolerance=1e-8,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=5000,
        costFunctionConvergenceFactor=1e-10
    )
    try:
        final_transform = registration.Execute(fixed, moving)
        return {
            'theta': np.rad2deg(final_transform.GetAngle()),
            'tx': final_transform.GetTranslation()[0],
            'ty': final_transform.GetTranslation()[1],
            'metric_value': registration.GetMetricValue(),
            'iterations': registration.GetOptimizerIteration(),
        }, final_transform
    except Exception as e:
        print(f"    LBFGSB failed: {e}")
        return None, None


def register_sitk_gd(fixed_np, moving_np, metric_name='MeanSquares'):
    """Register using standard Gradient Descent."""
    registration, fixed, moving = _setup_registration(fixed_np, moving_np, metric_name)
    registration.SetOptimizerAsGradientDescent(
        learningRate=50.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20
    )
    try:
        final_transform = registration.Execute(fixed, moving)
        return {
            'theta': np.rad2deg(final_transform.GetAngle()),
            'tx': final_transform.GetTranslation()[0],
            'ty': final_transform.GetTranslation()[1],
            'metric_value': registration.GetMetricValue(),
            'iterations': registration.GetOptimizerIteration(),
        }, final_transform
    except Exception as e:
        print(f"    GD failed: {e}")
        return None, None


def register_sitk_rsgd(fixed_np, moving_np, metric_name='MeanSquares'):
    """Register using Regular Step Gradient Descent."""
    registration, fixed, moving = _setup_registration(fixed_np, moving_np, metric_name)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=4.0, minStep=0.001,
        numberOfIterations=200, relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6
    )
    try:
        final_transform = registration.Execute(fixed, moving)
        return {
            'theta': np.rad2deg(final_transform.GetAngle()),
            'tx': final_transform.GetTranslation()[0],
            'ty': final_transform.GetTranslation()[1],
            'metric_value': registration.GetMetricValue(),
            'iterations': registration.GetOptimizerIteration(),
        }, final_transform
    except Exception as e:
        print(f"    RSGD failed: {e}")
        return None, None

# ============================================================
# 5. Run registration experiments
# ============================================================
experiments = [
    ('Intra-modality (T1_trans -> T1)', t1_transformed, t1_norm),
    ('Inter-modality (T2_trans -> T1)', t2_transformed, t1_norm),
]

metrics = ['MeanSquares', 'Correlation', 'MattesMutualInformation']

print("\n" + "="*60)
print("A. LBFGSB Optimizer (quasi-Newton)")
print("="*60)
all_sitk_results = {}

for label, source, target in experiments:
    for metric in metrics:
        exp_key = f"{label} | {metric} [LBFGSB]"
        print(f"\n--- {exp_key} ---")
        result, _ = register_sitk_lbfgsb(target, source, metric_name=metric)
        if result:
            all_sitk_results[exp_key] = result
            print(f"  Estimated: theta={result['theta']:.4f} deg, "
                  f"tx={result['tx']:.4f}, ty={result['ty']:.4f}")
            print(f"  Iterations: {result['iterations']}, "
                  f"Final metric: {result['metric_value']:.6f}")

print("\n" + "="*60)
print("B. Gradient Descent Optimizer (for comparison)")
print("="*60)
for label, source, target in experiments:
    for metric in ['MeanSquares', 'Correlation']:
        exp_key = f"{label} | {metric} [GD]"
        print(f"\n--- {exp_key} ---")
        result, _ = register_sitk_gd(target, source, metric_name=metric)
        if result:
            all_sitk_results[exp_key] = result
            print(f"  Estimated: theta={result['theta']:.4f} deg, "
                  f"tx={result['tx']:.4f}, ty={result['ty']:.4f}")
            print(f"  Iterations: {result['iterations']}, "
                  f"Final metric: {result['metric_value']:.6f}")

# ============================================================
# 6. Compare with ground truth
# ============================================================
print(f"\n{'='*100}")
print("REGISTRATION RESULTS - SimpleITK vs True Inverse")
print(f"True inverse: theta={INV_THETA_SITK:.4f} deg, tx={INV_TX_SITK:.4f}, ty={INV_TY_SITK:.4f}")
print(f"{'='*100}")
print(f"{'Experiment':<55} {'theta':>8} {'tx':>8} {'ty':>8}  "
      f"{'theta_err':>9} {'tx_err':>9} {'ty_err':>9}")
print("-" * 105)
for key, res in all_sitk_results.items():
    e = res
    print(f"{key:<55} {e['theta']:>8.2f} {e['tx']:>8.2f} {e['ty']:>8.2f}  "
          f"{e['theta'] - INV_THETA_SITK:>9.3f} {e['tx'] - INV_TX_SITK:>9.3f} "
          f"{e['ty'] - INV_TY_SITK:>9.3f}")

# ============================================================
# 8. Visualization
# ============================================================
def apply_found_transform_sitk(image_2d, theta_deg, tx, ty):
    img = sitk.Cast(numpy_to_sitk(image_2d), sitk.sitkFloat32)
    sz = img.GetSize()
    center = [sz[0] / 2.0, sz[1] / 2.0]
    transform = sitk.Euler2DTransform()
    transform.SetCenter(center)
    transform.SetAngle(np.deg2rad(theta_deg))
    transform.SetTranslation((tx, ty))
    return sitk_to_numpy(sitk.Resample(img, img, transform, sitk.sitkLinear,
                                        0.0, img.GetPixelID()))

best_intra_key = 'Intra-modality (T1_trans -> T1) | MeanSquares [LBFGSB]'
best_inter_key = 'Inter-modality (T2_trans -> T1) | MeanSquares [LBFGSB]'

if best_intra_key in all_sitk_results and best_inter_key in all_sitk_results:
    best_intra = all_sitk_results[best_intra_key]
    best_inter = all_sitk_results[best_inter_key]

    t1_registered = apply_found_transform_sitk(t1_transformed, best_intra['theta'],
                                                best_intra['tx'], best_intra['ty'])
    t2_registered = apply_found_transform_sitk(t2_transformed, best_inter['theta'],
                                                best_inter['tx'], best_inter['ty'])

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    # Row 1: Originals + transformed
    axes[0, 0].imshow(t1_slice.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('T1w original (target)', fontsize=10); axes[0, 0].axis('off')

    axes[0, 1].imshow(t2_slice.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('T2w original', fontsize=10); axes[0, 1].axis('off')

    axes[0, 2].imshow(t1_transformed.T, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'T1w transformed\n(theta=15 deg, tx=10, ty=-5)', fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(t2_transformed.T, cmap='gray', origin='lower')
    axes[0, 3].set_title(f'T2w transformed\n(theta=15 deg, tx=10, ty=-5)', fontsize=10)
    axes[0, 3].axis('off')

    # Row 2: Registered
    axes[1, 0].imshow(t1_registered.T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'SimpleITK Intra Registered\n'
                         f'(theta={best_intra["theta"]:.2f} deg, '
                         f'tx={best_intra["tx"]:.2f}, ty={best_intra["ty"]:.2f})', fontsize=9)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(t2_registered.T, cmap='gray', origin='lower')
    axes[1, 1].set_title(f'SimpleITK Inter Registered\n'
                         f'(theta={best_inter["theta"]:.2f} deg, '
                         f'tx={best_inter["tx"]:.2f}, ty={best_inter["ty"]:.2f})', fontsize=9)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.abs(t1_registered - t1_norm).T, cmap='hot', origin='lower')
    axes[1, 2].set_title('|Intra Reg - Target|', fontsize=10); axes[1, 2].axis('off')

    axes[1, 3].imshow(np.abs(t2_registered - t1_norm).T, cmap='hot', origin='lower')
    axes[1, 3].set_title('|Inter Reg - Target|', fontsize=10); axes[1, 3].axis('off')

    # Row 3: Overlays
    def make_overlay(ref, mov):
        ov = np.zeros((*ref.shape, 3))
        rn = normalize(ref); mn = normalize(mov)
        ov[:, :, 0] = rn; ov[:, :, 1] = mn
        return np.clip(ov, 0, 1)

    axes[2, 0].imshow(make_overlay(t1_norm, t1_transformed), origin='lower')
    axes[2, 0].set_title('Before Intra\n(R=Target, G=Transformed)', fontsize=9)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(make_overlay(t1_norm, t1_registered), origin='lower')
    axes[2, 1].set_title('After Intra (SimpleITK)\n(R=Target, G=Registered)', fontsize=9)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(make_overlay(t1_norm, t2_transformed), origin='lower')
    axes[2, 2].set_title('Before Inter\n(R=Target, G=Transformed)', fontsize=9)
    axes[2, 2].axis('off')

    axes[2, 3].imshow(make_overlay(t1_norm, t2_registered), origin='lower')
    axes[2, 3].set_title('After Inter (SimpleITK)\n(R=Target, G=Registered)', fontsize=9)
    axes[2, 3].axis('off')

    plt.suptitle('LAB6 Project 3 - SimpleITK Co-registration', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'project3_output.png'), dpi=150)
    plt.close()
    print("\nFigure saved: project3_output.png")
else:
    print("\nSkipping visualization due to failed registration.")

print("\nDone!")
