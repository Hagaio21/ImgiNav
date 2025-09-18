"""
geometry_utils.py
-----------------
Geometry-related utilities:
  - sampling points from meshes
  - extracting RGB colors
"""

import numpy as np
import trimesh


# ------------------------------------------------------
# Point Sampling
# ------------------------------------------------------
def sample_points_from_mesh(mesh: trimesh.Trimesh,
                            n_points: int = 2048,
                            with_color: bool = True,
                            fallback_color=(127,127,127)) -> np.ndarray:
    """
    Sample points from a mesh surface.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to sample from.
    n_points : int
        Number of points to sample.
    with_color : bool
        If True, try to extract colors (vertex or texture).
    fallback_color : tuple[int,int,int]
        Default RGB if no color found.

    Returns
    -------
    (N, 6) ndarray
        Columns: x, y, z, r, g, b
    """
    # Sample surface points
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)

    # Colors
    if with_color:
        colors = _get_point_colors(mesh, face_indices, fallback_color)
    else:
        colors = np.tile(fallback_color, (n_points, 1))

    return np.hstack([points, colors])


def _get_point_colors(mesh: trimesh.Trimesh,
                      face_indices: np.ndarray,
                      fallback_color=(127,127,127)) -> np.ndarray:
    """
    Extract RGB colors for sampled points.
    Try vertex colors first, then face colors, else fallback.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    face_indices : np.ndarray
        Indices of faces from which points were sampled.
    fallback_color : tuple[int,int,int]

    Returns
    -------
    (N, 3) ndarray of uint8
    """
    # Case 1: vertex colors
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) > 0:
        colors = mesh.visual.vertex_colors[:, :3]  # drop alpha if present
        face_vertices = mesh.faces[face_indices]
        return colors[face_vertices[:,0]]

    # Case 2: face colors
    if hasattr(mesh.visual, "face_colors") and len(mesh.visual.face_colors) > 0:
        colors = mesh.visual.face_colors[:, :3]
        return colors[face_indices]

    # Fallback: solid default color
    return np.tile(fallback_color, (len(face_indices), 1))
