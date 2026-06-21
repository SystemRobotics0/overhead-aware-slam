#!/usr/bin/env python3
"""Create a readable two-environment map comparison figure from raw maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from eval_common import DEFAULT_ENV_MAPS, load_ros_map, repo_root_from_script


COLUMNS = [
    ("ground_truth", "Ground truth"),
    ("lidar_only", "2D LiDAR"),
    ("sensor_fusion", "Ours"),
    ("rtabmap", "3D LiDAR"),
]


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def map_panel(map_path: Path, title: str, panel_w: int, panel_h: int, title_h: int) -> Image.Image:
    image = Image.open(map_path).convert("L")
    image = image.convert("RGB")
    body_h = panel_h - title_h
    scale = min((panel_w - 20) / image.width, (body_h - 20) / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resample = getattr(Image, "Resampling", Image).NEAREST
    image = image.resize(new_size, resample)

    panel = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_w - 1, title_h - 1), fill=(238, 241, 245))
    draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=(170, 178, 190), width=1)

    font = load_font(18)
    title_box = draw.textbbox((0, 0), title, font=font)
    title_w = title_box[2] - title_box[0]
    draw.text(((panel_w - title_w) // 2, 7), title, fill=(25, 34, 48), font=font)

    x = (panel_w - image.width) // 2
    y = title_h + (body_h - image.height) // 2
    panel.paste(image, (x, y))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--output", type=Path, default=Path("results/map_figure.png"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    panel_w, panel_h, title_h = 360, 300, 36
    row_label_w = 90
    gap = 12
    fig_w = row_label_w + len(COLUMNS) * panel_w + (len(COLUMNS) + 1) * gap
    fig_h = 2 * panel_h + 3 * gap
    figure = Image.new("RGB", (fig_w, fig_h), (248, 250, 252))
    draw = ImageDraw.Draw(figure)
    row_font = load_font(20)

    for row_idx, env_name in enumerate(("env1", "env2")):
        y = gap + row_idx * (panel_h + gap)
        row_label = "Env 1" if env_name == "env1" else "Env 2"
        draw.text((18, y + 12), row_label, fill=(25, 34, 48), font=row_font)

        for col_idx, (method, title) in enumerate(COLUMNS):
            ros_map = load_ros_map(repo_root / DEFAULT_ENV_MAPS[env_name][method], method)
            x = row_label_w + gap + col_idx * (panel_w + gap)
            panel = map_panel(ros_map.image_path, title, panel_w, panel_h, title_h)
            figure.paste(panel, (x, y))

    figure.save(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
