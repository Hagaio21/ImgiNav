#!/usr/bin/env python3

import numpy as np
from typing import Tuple

def pca_plane_fit(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a plane to points using PCA. Returns origin and normal."""
    if points.shape[0] < 3:
        return points.mean(axis=0), np.array([0, 0, 1.0], dtype=np.float64)
    
    origin = points.mean(axis=0)
    X = points - origin
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)
    normal = V[:, 0]  # smallest eigenvalue
    return origin.astype(np.float64), normal / (np.linalg.norm(normal) + 1e-12)

def build_orthonormal_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build orthonormal frame from normal vector."""
    Y = np.array([0, 1, 0], dtype=np.float64)
    X = np.array([1, 0, 0], dtype=np.float64)
    
    v = Y - (Y @ normal) * normal
    if np.linalg.norm(v) < 1e-9:
        v = X - (X @ normal) * normal
    v = v / (np.linalg.norm(v) + 1e-12)
    
    u = np.cross(normal, v)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(u, normal)
    v = v / (np.linalg.norm(v) + 1e-12)
    
    return u, v

def angle_from_center(center: np.ndarray, point: np.ndarray) -> float:
    """Compute angle from center to point in XY plane."""
    v = point[:2] - center[:2]  # Use only XY components
    return np.arctan2(v[1], v[0])

def compute_directional_relations(angle_a: float, angle_b: float) -> Tuple[str, str]:
    """Compute directional relations between two angles."""
    d_ang = np.rad2deg((angle_b - angle_a + np.pi*2) % (np.pi*2))
    
    if d_ang < 45 or d_ang > 315:
        return "front_of", "behind"
    elif 45 <= d_ang < 135:
        return "right_of", "left_of"
    elif 135 <= d_ang < 225:
        return "behind", "front_of"
    else:
        return "left_of", "right_of"

