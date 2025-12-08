#!/usr/bin/env python3
"""
Calibrate polynomial joint mapping between Allegro and LEAP hands.

Uses normalized scaling or Isaac Gym FK to generate paired joint configurations.

Usage:
    python scripts/calibrate_hand_mapping.py --num_samples 50000 --degree 2

Output:
    cache/allegro_to_leap_calibration.npz
    cache/leap_to_allegro_calibration.npz
"""

import os
import sys
import argparse
import numpy as np


# Joint limits from URDFs
ALLEGRO_LOWER = np.array([
    -0.47, -0.196, -0.174, -0.227,  # Index
    0.263, -0.105, -0.189, -0.162,   # Thumb
    -0.47, -0.196, -0.174, -0.227,  # Middle
    -0.47, -0.196, -0.174, -0.227,  # Ring
], dtype=np.float32)

ALLEGRO_UPPER = np.array([
    0.47, 1.61, 1.709, 1.618,   # Index
    1.396, 1.163, 1.644, 1.719,  # Thumb
    0.47, 1.61, 1.709, 1.618,   # Middle
    0.47, 1.61, 1.709, 1.618,   # Ring
], dtype=np.float32)

LEAP_LOWER = np.array([
    -1.047, -0.314, -0.506, -0.366,  # Index
    -0.349, -0.47, -1.20, -1.34,      # Thumb
    -1.047, -0.314, -0.506, -0.366,  # Middle
    -1.047, -0.314, -0.506, -0.366,  # Ring
], dtype=np.float32)

LEAP_UPPER = np.array([
    1.047, 2.23, 1.885, 1.74,   # Index
    2.094, 2.443, 1.90, 1.88,    # Thumb
    1.047, 2.23, 1.885, 1.74,   # Middle
    1.047, 2.23, 1.885, 1.74,   # Ring
], dtype=np.float32)


def calibrate_polynomial_mapping_np(
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    num_samples: int = 50000,
    degree: int = 2,
    regularization: float = 1e-4,
    verbose: bool = True
) -> dict:
    """
    Generate calibration data using normalized scaling as ground truth.

    Args:
        source_lower: Source hand lower joint limits (16,)
        source_upper: Source hand upper joint limits (16,)
        target_lower: Target hand lower joint limits (16,)
        target_upper: Target hand upper joint limits (16,)
        num_samples: Number of random samples for calibration
        degree: Polynomial degree (1 or 2)
        regularization: Ridge regression regularization strength
        verbose: Print progress

    Returns:
        Dictionary with 'forward_weights' and 'inverse_weights' arrays
    """
    n_joints = 16

    # Compute number of polynomial features
    if degree == 1:
        n_features = 1 + n_joints
    elif degree == 2:
        n_features = 1 + n_joints + (n_joints * (n_joints + 1)) // 2
    else:
        raise ValueError(f"Degree {degree} not supported")

    def compute_poly_features(x: np.ndarray) -> np.ndarray:
        """Compute polynomial features for numpy array."""
        batch = x.shape[0]
        features = [np.ones((batch, 1))]  # bias
        features.append(x)  # linear
        if degree >= 2:
            quad = []
            for i in range(n_joints):
                for j in range(i, n_joints):
                    quad.append(x[:, i:i+1] * x[:, j:j+1])
            features.append(np.concatenate(quad, axis=1))
        return np.concatenate(features, axis=1)

    def normalize(joints, lower, upper):
        return 2.0 * (joints - lower) / (upper - lower + 1e-8) - 1.0

    def denormalize(normalized, lower, upper):
        return (normalized + 1.0) * 0.5 * (upper - lower) + lower

    if verbose:
        print(f"Calibrating polynomial mapping (degree={degree}, n_features={n_features})")
        print(f"Sampling {num_samples} joint configurations...")

    # Sample random source joint configurations
    source_samples = np.random.uniform(source_lower, source_upper, size=(num_samples, n_joints))

    # Use normalized scaling as proxy for "ground truth"
    source_norm = normalize(source_samples, source_lower, source_upper)
    target_samples = denormalize(source_norm, target_lower, target_upper)

    if verbose:
        print(f"Collected {len(source_samples)} sample pairs")

    # Normalize all samples
    source_norm = normalize(source_samples, source_lower, source_upper)
    target_norm = normalize(target_samples, target_lower, target_upper)

    # Compute polynomial features
    source_features = compute_poly_features(source_norm)
    target_features = compute_poly_features(target_norm)

    if verbose:
        print(f"Feature matrix shape: {source_features.shape}")
        print("Fitting forward mapping (source -> target)...")

    # Fit forward mapping: target_norm = source_features @ W_forward
    # Ridge regression: W = (X^T X + λI)^-1 X^T Y
    XtX = source_features.T @ source_features
    XtY = source_features.T @ target_norm
    W_forward = np.linalg.solve(
        XtX + regularization * np.eye(n_features),
        XtY
    )

    # Compute forward error
    pred_target = source_features @ W_forward
    forward_mse = np.mean((pred_target - target_norm) ** 2)

    if verbose:
        print(f"  Forward MSE (normalized): {forward_mse:.6f}")
        print("Fitting inverse mapping (target -> source)...")

    # Fit inverse mapping: source_norm = target_features @ W_inverse
    XtX_inv = target_features.T @ target_features
    XtY_inv = target_features.T @ source_norm
    W_inverse = np.linalg.solve(
        XtX_inv + regularization * np.eye(n_features),
        XtY_inv
    )

    # Compute inverse error
    pred_source = target_features @ W_inverse
    inverse_mse = np.mean((pred_source - source_norm) ** 2)

    if verbose:
        print(f"  Inverse MSE (normalized): {inverse_mse:.6f}")
        print("Calibration complete!")

    return {
        'forward_weights': W_forward.astype(np.float32),
        'inverse_weights': W_inverse.astype(np.float32),
        'forward_mse': forward_mse,
        'inverse_mse': inverse_mse,
        'degree': degree,
        'num_samples': num_samples
    }


def calibrate_from_normalized_scale(num_samples: int = 50000, degree: int = 2, verbose: bool = True):
    """
    Calibrate using normalized scale as ground truth.

    This is fast and provides a good baseline. The polynomial fitting can
    capture the relationship between differently-scaled joint spaces.
    """
    if verbose:
        print("=" * 60)
        print("Calibrating Allegro -> LEAP mapping")
        print("=" * 60)

    # Allegro -> LEAP calibration
    result_a2l = calibrate_polynomial_mapping_np(
        source_lower=ALLEGRO_LOWER,
        source_upper=ALLEGRO_UPPER,
        target_lower=LEAP_LOWER,
        target_upper=LEAP_UPPER,
        num_samples=num_samples,
        degree=degree,
        regularization=1e-4,
        verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Calibrating LEAP -> Allegro mapping")
        print("=" * 60)

    # LEAP -> Allegro calibration
    result_l2a = calibrate_polynomial_mapping_np(
        source_lower=LEAP_LOWER,
        source_upper=LEAP_UPPER,
        target_lower=ALLEGRO_LOWER,
        target_upper=ALLEGRO_UPPER,
        num_samples=num_samples,
        degree=degree,
        regularization=1e-4,
        verbose=verbose
    )

    return result_a2l, result_l2a


def main():
    parser = argparse.ArgumentParser(description='Calibrate hand joint mapping')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of calibration samples')
    parser.add_argument('--degree', type=int, default=2, choices=[1, 2], help='Polynomial degree')
    parser.add_argument('--output_dir', type=str, default='cache', help='Output directory for calibration files')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("Using normalized scale as ground truth (fast calibration)")
    result_a2l, result_l2a = calibrate_from_normalized_scale(
        num_samples=args.num_samples,
        degree=args.degree,
        verbose=True
    )

    # Save calibration files
    a2l_path = os.path.join(args.output_dir, 'allegro_to_leap_calibration.npz')
    l2a_path = os.path.join(args.output_dir, 'leap_to_allegro_calibration.npz')

    np.savez(a2l_path,
             forward_weights=result_a2l['forward_weights'],
             inverse_weights=result_a2l['inverse_weights'],
             forward_mse=result_a2l['forward_mse'],
             inverse_mse=result_a2l['inverse_mse'],
             degree=result_a2l['degree'],
             num_samples=result_a2l['num_samples'])

    np.savez(l2a_path,
             forward_weights=result_l2a['forward_weights'],
             inverse_weights=result_l2a['inverse_weights'],
             forward_mse=result_l2a['forward_mse'],
             inverse_mse=result_l2a['inverse_mse'],
             degree=result_l2a['degree'],
             num_samples=result_l2a['num_samples'])

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print(f"  Allegro -> LEAP: {a2l_path}")
    print(f"  LEAP -> Allegro: {l2a_path}")
    print("=" * 60)

    # Print summary
    print("\nSummary:")
    print(f"  Allegro -> LEAP forward MSE: {result_a2l['forward_mse']:.6f}")
    print(f"  Allegro -> LEAP inverse MSE: {result_a2l['inverse_mse']:.6f}")
    print(f"  LEAP -> Allegro forward MSE: {result_l2a['forward_mse']:.6f}")
    print(f"  LEAP -> Allegro inverse MSE: {result_l2a['inverse_mse']:.6f}")

    # Verify calibration files
    print("\nVerifying saved files...")
    a2l_data = np.load(a2l_path)
    l2a_data = np.load(l2a_path)
    print(f"  Allegro->LEAP weights shape: {a2l_data['forward_weights'].shape}")
    print(f"  LEAP->Allegro weights shape: {l2a_data['forward_weights'].shape}")


if __name__ == '__main__':
    main()
