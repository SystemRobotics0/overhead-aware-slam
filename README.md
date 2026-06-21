# Overhead-Aware 2D LiDAR SLAM

## Abstract

Two-dimensional LiDAR SLAM is widely used for indoor mobile robot navigation, but its single horizontal scan plane can miss overhanging obstacles such as tables and chairs, whose widest extents lie above the laser cross-section. We propose a semantic map augmentation method that combines monocular object detection with 2D LiDAR to recover obstacle footprints poorly represented by planar scanning. A MobileNet SSD detector identifies furniture classes, associates detections with LiDAR scans inside a camera-projected angular wedge, and compares camera-implied object width with measured LiDAR support width to classify obstacles as solid or overhanging. Class-informed footprints are then painted into a copy of the SLAM occupancy map.

We evaluate the method in two custom Gazebo indoor environments with explicit overhang collision geometry. Compared with a SLAM Toolbox baseline using 2D LiDAR, the proposed method improves Jaccard/IoU by 7-10 pp, Dice coefficient by 6-10 pp, and recall by 20-23 pp. Across 60 navigation trials, goal success increases from 45.0% to 83.3%, and stuck/recovery events decrease from 61.7% to 28.3%. Although RTAB-Map with 3D LiDAR remains the strongest navigation reference, the results show that semantic augmentation substantially improves 2D LiDAR navigation in structured simulated indoor environments.

## Demo

Demo video: **coming soon**.

## Methods

| Method | Map source | Navigation |
|---|---|---|
| `lidar_only` | 2D LiDAR SLAM | Nav2 static-map evaluation |
| `sensor_fusion` | 2D LiDAR SLAM + monocular chair/table detections | Same Nav2 setup |
| `rtabmap` | 3D LiDAR / RTAB-Map reference | Same Nav2 setup |

All navigation comparisons use the same planner, controller, velocity limits, and robot radius.

## Selected Results

| Artifact | File |
|---|---|
| Map comparison figure | `results/map_figure.png` |
| Navigation summary | `results/navigation_summary_reset.csv` |
| Runtime profile | `results/runtime_profile_live_summary.csv` |

Result highlights:

- Combined navigation success: 2D LiDAR `45.00%`, sensor fusion `83.33%`, RTAB-Map `98.33%`.
- Live callback timing on development hardware: detector `33.037 ms` mean, fusion `20.020 ms` mean, occupancy-map callback `5.314 ms` mean.
- Detector runs through CPU TensorFlow Lite in the included runtime node.

See `results/README.md` for the result file index.

## Repository Layout

| Path | Contents |
|---|---|
| `src/sensor_fusion_desc/` | Worlds, robot model, maps, Nav2/SLAM configs |
| `src/sensor_fusion_nodes/` | Detector, fusion, occupancy-map, map-saving, and evaluator nodes |
| `src/sensor_fusion_run/` | Launch files |
| `maps/` | Map YAML/PGM files |
| `scripts/` | Analysis and summary scripts |
| `results/` | Selected result summaries and figures |

## Setup

Tested with ROS 2 Jazzy on Ubuntu 24.04.

```bash
source /opt/ros/jazzy/setup.bash

sudo apt update
sudo apt install -y \
  ros-$ROS_DISTRO-slam-toolbox \
  ros-$ROS_DISTRO-ros-gz \
  ros-$ROS_DISTRO-ros-gz-bridge \
  ros-$ROS_DISTRO-ros-gz-sim \
  ros-$ROS_DISTRO-tf2-ros \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-rtabmap-ros \
  ros-$ROS_DISTRO-navigation2 \
  ros-$ROS_DISTRO-nav2-bringup \
  python3-opencv \
  python3-numpy
```

TensorFlow is required for the TFLite detector:

```bash
pip install tensorflow --break-system-packages
```

Build:

```bash
colcon build --symlink-install
source install/setup.bash
```

## Run Fusion Demo

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch sensor_fusion_run test.launch.py
```

Teleoperate:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Optional live runtime profiling:

```bash
ros2 launch sensor_fusion_run test.launch.py \
  runtime_profile_csv:=$PWD/results/runtime_profile_live_raw.csv

python3 scripts/summarize_runtime_profile.py \
  --samples results/runtime_profile_live_raw.csv \
  --output results/runtime_profile_live_summary.csv
```

## Run Static-Map Navigation

Example using the Env 1 sensor-fusion map:

```bash
ros2 launch sensor_fusion_run nav2_static_map.launch.py \
  world_name:=house2.world \
  map:=$PWD/maps/augmented_map.yaml
```

Run the evaluator in another terminal:

```bash
ros2 run sensor_fusion_nodes nav_evaluator \
  --ros-args \
  -p goals_file:=src/sensor_fusion_desc/config/eval_goals_house2.yaml \
  -p method_name:=sensor_fusion \
  -p output_csv:=results/nav_env1_sensor_fusion_new.csv \
  -p record_trajectories:=true \
  -p trajectory_dir:=results/trajectories_new
```

## Scripts

See `scripts/README.md` for the compact script index.

Common offline commands:

```bash
python3 scripts/make_map_figure.py
```

## Notes

- The results are from simulation experiments.
- Development hardware/software details are provided as compact metadata in `results/system_report.csv`.
