#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple
import numpy as np


def world_to_local_coords(points: np.ndarray, origin: np.ndarray, 
                         u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    R = np.stack([u, v, n], axis=1)
    return (points - origin) @ R


def points_to_image_coords(u_vals: np.ndarray, v_vals: np.ndarray, 
                          uv_bounds: Tuple[float, float, float, float],
                          resolution: int, margin: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    umin, umax, vmin, vmax = uv_bounds
    span = max(umax - umin, vmax - vmin, 1e-6)
    scale = (resolution - 2 * margin) / span
    
    u_pix = (u_vals - umin) * scale + margin
    v_pix = (v_vals - vmin) * scale + margin
    
    x_img = np.clip(np.round(u_pix).astype(np.int32), 0, resolution - 1)
    y_img = np.clip(np.round((resolution - 1) - v_pix).astype(np.int32), 0, resolution - 1)
    
    return x_img, y_img


def pca_plane_fit(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        return points.mean(axis=0), np.array([0, 0, 1.0], dtype=np.float64)
    
    origin = points.mean(axis=0)
    X = points - origin
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)
    normal = V[:, 0]  # smallest eigenvalue
    return origin.astype(np.float64), normal / (np.linalg.norm(normal) + 1e-12)


def build_orthonormal_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def get_2d_bounds(points_2d: np.ndarray) -> Tuple[float, float, float, float]:
    if points_2d.shape[0] == 0:
        return (0.0, 1.0, 0.0, 1.0)
    
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    return float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1])


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def angle_from_center(center: np.ndarray, point: np.ndarray) -> float:

    v = point[:2] - center[:2]  # Use only XY components
    return np.arctan2(v[1], v[0])