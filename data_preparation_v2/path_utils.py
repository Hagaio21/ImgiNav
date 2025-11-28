#!/usr/bin/env python3
"""
Shared path utilities for the ImgiNav data preparation pipeline.

This module provides a unified way to load paths from a YAML configuration file,
making it easy to configure the pipeline for different environments.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


class PathConfig:
    """
    Configuration manager for pipeline paths.
    
    Loads paths from a YAML configuration file and provides easy access
    to all input/output directories.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration from a YAML file.
        
        Args:
            config_path: Path to the paths.yaml configuration file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)
        
        # Cache resolved paths
        self._resolved_paths: Dict[str, Path] = {}
    
    def _resolve_path(self, path: Optional[str], relative_to: Optional[Path] = None) -> Optional[Path]:
        """Resolve a path, making it absolute if relative."""
        if path is None:
            return None
        
        p = Path(path)
        if p.is_absolute():
            return p
        
        if relative_to is not None:
            return relative_to / p
        
        return p
    
    # ========================================
    # Base directories
    # ========================================
    
    @property
    def base_dir(self) -> Path:
        """Base directory for the project."""
        return Path(self._config.get("base_dir", "."))
    
    @property
    def dataset_root(self) -> Path:
        """Root directory for the dataset."""
        return Path(self._config.get("dataset_root", "."))
    
    # ========================================
    # Input paths
    # ========================================
    
    @property
    def scenes_dir(self) -> Path:
        """Directory containing 3D-FRONT scene JSON files."""
        inputs = self._config.get("inputs", {})
        return Path(inputs.get("scenes_dir", self.dataset_root / "scenes"))
    
    @property
    def model_dir(self) -> Path:
        """Directory containing 3D-FUTURE furniture models."""
        inputs = self._config.get("inputs", {})
        return Path(inputs.get("model_dir", ""))
    
    @property
    def model_info(self) -> Path:
        """Path to model_info.json."""
        inputs = self._config.get("inputs", {})
        return Path(inputs.get("model_info", self.model_dir / "model_info.json"))
    
    @property
    def taxonomy(self) -> Path:
        """Path to taxonomy.json."""
        inputs = self._config.get("inputs", {})
        return Path(inputs.get("taxonomy", self.base_dir / "taxonomy.json"))
    
    @property
    def texture_dir(self) -> Optional[Path]:
        """Optional texture directory."""
        inputs = self._config.get("inputs", {})
        texture_dir = inputs.get("texture_dir")
        return Path(texture_dir) if texture_dir else None
    
    # ========================================
    # Output paths
    # ========================================
    
    def _get_output_path(self, key: str, default: str) -> Path:
        """Get an output path, resolving relative paths to dataset_root."""
        outputs = self._config.get("outputs", {})
        path_str = outputs.get(key, default)
        return self._resolve_path(path_str, self.dataset_root)
    
    @property
    def geometry_dir(self) -> Path:
        """Output directory for geometry GLB files."""
        return self._get_output_path("geometry", "geometry")
    
    @property
    def metadata_dir(self) -> Path:
        """Output directory for metadata JSON files."""
        return self._get_output_path("metadata", "metadata")
    
    @property
    def layouts_dir(self) -> Path:
        """Output directory for layout images."""
        return self._get_output_path("layouts", "layouts")
    
    @property
    def povs_dir(self) -> Path:
        """Output directory for POV images."""
        return self._get_output_path("povs", "povs")
    
    @property
    def graphs_dir(self) -> Path:
        """Output directory for scene/room graphs."""
        return self._get_output_path("graphs", "graphs")
    
    @property
    def manifests_dir(self) -> Path:
        """Output directory for manifest files."""
        return self._get_output_path("manifests", "manifests")
    
    # ========================================
    # HPC settings
    # ========================================
    
    @property
    def logs_dir(self) -> Path:
        """Directory for HPC job logs."""
        hpc = self._config.get("hpc", {})
        return Path(hpc.get("logs_dir", self.base_dir / "logs"))
    
    @property
    def conda_env(self) -> str:
        """Conda environment name."""
        hpc = self._config.get("hpc", {})
        return hpc.get("conda_env", "imginav")
    
    # ========================================
    # Rendering settings
    # ========================================
    
    @property
    def layout_resolution(self) -> int:
        """Resolution for layout images."""
        rendering = self._config.get("rendering", {})
        return rendering.get("layout_resolution", 512)
    
    @property
    def pov_width(self) -> int:
        """Width for POV images."""
        rendering = self._config.get("rendering", {})
        return rendering.get("pov_width", 1280)
    
    @property
    def pov_height(self) -> int:
        """Height for POV images."""
        rendering = self._config.get("rendering", {})
        return rendering.get("pov_height", 720)
    
    @property
    def pov_fov(self) -> float:
        """Field of view for POV images."""
        rendering = self._config.get("rendering", {})
        return rendering.get("pov_fov", 60.0)
    
    # ========================================
    # Utility methods
    # ========================================
    
    def find_scene_file(self, scene_id: str) -> Optional[Path]:
        """
        Find a scene JSON file by scene ID.
        Searches recursively in scenes_dir.
        """
        # First try direct path
        direct_path = self.scenes_dir / f"{scene_id}.json"
        if direct_path.exists():
            return direct_path
        
        # Search recursively
        found = list(self.scenes_dir.rglob(f"{scene_id}.json"))
        if found:
            return found[0]
        
        return None
    
    def find_scene_files(self, scene_ids: List[str]) -> List[Path]:
        """
        Find scene JSON files for a list of scene IDs.
        Returns list of found paths (may be shorter than input if some not found).
        """
        scene_files = []
        for scene_id in scene_ids:
            scene_file = self.find_scene_file(scene_id)
            if scene_file:
                scene_files.append(scene_file)
        return scene_files
    
    def get_scene_metadata_path(self, scene_id: str) -> Path:
        """Get the metadata path for a scene."""
        return self.metadata_dir / "scenes" / f"{scene_id}.json"
    
    def get_room_metadata_paths(self, scene_id: str) -> List[Path]:
        """Get all room metadata paths for a scene."""
        rooms_dir = self.metadata_dir / "rooms"
        if not rooms_dir.exists():
            return []
        return list(rooms_dir.glob(f"{scene_id}_*.json"))
    
    def get_geometry_paths(self, scene_id: str) -> Dict[str, Path]:
        """Get geometry GLB paths for a scene."""
        return {
            "tex": self.geometry_dir / "tex" / f"{scene_id}_tex.glb",
            "seg": self.geometry_dir / "seg" / f"{scene_id}_seg.glb",
        }
    
    def ensure_output_dirs(self):
        """Create all output directories if they don't exist."""
        for path in [
            self.geometry_dir / "tex",
            self.geometry_dir / "seg",
            self.metadata_dir / "scenes",
            self.metadata_dir / "rooms",
            self.layouts_dir / "tex",
            self.layouts_dir / "seg",
            self.povs_dir / "tex",
            self.povs_dir / "seg",
            self.graphs_dir / "jsons",
            self.graphs_dir / "texts",
            self.manifests_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self) -> str:
        return f"PathConfig({self.config_path})"
    
    def summary(self) -> str:
        """Return a summary of all configured paths."""
        lines = [
            "=" * 60,
            "Path Configuration Summary",
            "=" * 60,
            f"Config file: {self.config_path}",
            "",
            "Base directories:",
            f"  base_dir: {self.base_dir}",
            f"  dataset_root: {self.dataset_root}",
            "",
            "Input paths:",
            f"  scenes_dir: {self.scenes_dir}",
            f"  model_dir: {self.model_dir}",
            f"  model_info: {self.model_info}",
            f"  taxonomy: {self.taxonomy}",
            f"  texture_dir: {self.texture_dir}",
            "",
            "Output paths:",
            f"  geometry_dir: {self.geometry_dir}",
            f"  metadata_dir: {self.metadata_dir}",
            f"  layouts_dir: {self.layouts_dir}",
            f"  povs_dir: {self.povs_dir}",
            f"  graphs_dir: {self.graphs_dir}",
            f"  manifests_dir: {self.manifests_dir}",
            "",
            "HPC settings:",
            f"  logs_dir: {self.logs_dir}",
            f"  conda_env: {self.conda_env}",
            "",
            "Rendering settings:",
            f"  layout_resolution: {self.layout_resolution}",
            f"  pov_width: {self.pov_width}",
            f"  pov_height: {self.pov_height}",
            f"  pov_fov: {self.pov_fov}",
            "=" * 60,
        ]
        return "\n".join(lines)


def load_scene_list(scene_list_path: Path) -> List[str]:
    """
    Load scene IDs from a text file (one per line).
    
    Lines starting with # are treated as comments.
    Empty lines are skipped.
    """
    scenes = []
    with open(scene_list_path, "r", encoding="utf-8") as f:
        for line in f:
            scene_id = line.strip()
            if scene_id and not scene_id.startswith("#"):
                scenes.append(scene_id)
    return scenes


def add_common_args(parser):
    """Add common arguments to an argument parser."""
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to paths.yaml configuration file"
    )
    parser.add_argument(
        "--scene-list", "-s",
        type=str,
        default=None,
        help="Path to text file with scene IDs (one per line)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of scenes to process"
    )


if __name__ == "__main__":
    # Test the configuration
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python path_utils.py <paths.yaml>")
        sys.exit(1)
    
    config = PathConfig(sys.argv[1])
    print(config.summary())