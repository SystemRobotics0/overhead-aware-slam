#!/usr/bin/env python3
"""Shared helpers for evaluating ROS occupancy maps.

The helpers intentionally use the YAML origin and resolution for alignment
instead of comparing raw image pixels directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import yaml
from PIL import Image


@dataclass(frozen=True)
class RosMap:
    name: str
    yaml_path: Path
    image_path: Path
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float
    pixels: np.ndarray
    occupied: np.ndarray

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        ox, oy, yaw = self.origin
        if abs(yaw) > 1e-9:
            raise ValueError(f"{self.yaml_path}: non-zero map yaw is not supported")
        return (
            ox,
            oy,
            ox + self.width * self.resolution,
            oy + self.height * self.resolution,
        )


DEFAULT_ENV_MAPS: Dict[str, Dict[str, str]] = {
    "env1": {
        "ground_truth": "maps/gt_map_v1.yaml",
        "lidar_only": "maps/lidar_map_v1.yaml",
        "sensor_fusion": "maps/augmented_map.yaml",
        "rtabmap": "maps/rtab_map1_v1.yaml",
    },
    "env2": {
        "ground_truth": "maps/gt_map2_v1.yaml",
        "lidar_only": "maps/lidar_map2_v1.yaml",
        "sensor_fusion": "maps/augmented_map_2.yaml",
        "rtabmap": "maps/rtab_map2_v1.yaml",
    },
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_ros_map(yaml_path: Path, name: str | None = None) -> RosMap:
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    image_path = Path(meta["image"])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image_path = image_path.resolve()

    image = Image.open(image_path).convert("L")
    pixels = np.array(image, dtype=np.uint8)

    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.196))

    if negate:
        occupancy_prob = pixels.astype(np.float32) / 255.0
    else:
        occupancy_prob = (255.0 - pixels.astype(np.float32)) / 255.0
    occupied = occupancy_prob >= occupied_thresh

    origin_raw = meta.get("origin", [0.0, 0.0, 0.0])
    origin = tuple(float(v) for v in origin_raw[:3])

    return RosMap(
        name=name or yaml_path.stem,
        yaml_path=yaml_path,
        image_path=image_path,
        resolution=float(meta["resolution"]),
        origin=origin,
        negate=negate,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
        pixels=pixels,
        occupied=occupied,
    )


def load_env_maps(repo_root: Path, env_name: str) -> Dict[str, RosMap]:
    spec = DEFAULT_ENV_MAPS[env_name]
    return {
        method: load_ros_map(repo_root / rel_path, method)
        for method, rel_path in spec.items()
    }


def common_grid(maps: Iterable[RosMap]) -> Tuple[float, float, float, int, int]:
    maps = list(maps)
    min_x = min(m.bounds[0] for m in maps)
    min_y = min(m.bounds[1] for m in maps)
    max_x = max(m.bounds[2] for m in maps)
    max_y = max(m.bounds[3] for m in maps)
    resolution = min(m.resolution for m in maps)

    min_x = np.floor(min_x / resolution) * resolution
    min_y = np.floor(min_y / resolution) * resolution
    max_x = np.ceil(max_x / resolution) * resolution
    max_y = np.ceil(max_y / resolution) * resolution

    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    return float(min_x), float(min_y), float(resolution), width, height


def occupied_world_points(map_data: RosMap) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(map_data.occupied)
    ox, oy, _ = map_data.origin
    x = ox + (cols.astype(np.float64) + 0.5) * map_data.resolution
    y = oy + (map_data.height - rows.astype(np.float64) - 0.5) * map_data.resolution
    return x, y


def rasterize_on_common_grid(
    map_data: RosMap,
    min_x: float,
    min_y: float,
    resolution: float,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x, y = occupied_world_points(map_data)
    cols = np.floor((x - min_x) / resolution).astype(np.int64)
    rows = np.floor((y - min_y) / resolution).astype(np.int64)
    valid = (0 <= cols) & (cols < width) & (0 <= rows) & (rows < height)
    mask[rows[valid], cols[valid]] = True
    return mask


def format_float(value: float) -> str:
    return f"{value:.4f}"
