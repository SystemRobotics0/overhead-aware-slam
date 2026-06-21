import csv
import pathlib
import time


class RuntimeProfiler:
    """Lightweight CSV sampler for ROS callback timing."""

    FIELDS = [
        'timestamp',
        'node',
        'component',
        'duration_ms',
        'num_detections',
        'num_lidar_points',
        'num_clusters',
        'num_confirmed_objects',
    ]

    def __init__(self, node_name, csv_path='', flush_every=50):
        self.node_name = node_name
        self.csv_path = str(csv_path or '').strip()
        self.flush_every = max(1, int(flush_every))
        self.samples = []

    @property
    def enabled(self):
        return bool(self.csv_path)

    def record_many(
        self,
        durations_ms,
        num_detections='',
        num_lidar_points='',
        num_clusters='',
        num_confirmed_objects='',
    ):
        if not self.enabled:
            return
        stamp = f'{time.time():.6f}'
        for component, duration_ms in durations_ms.items():
            self.samples.append({
                'timestamp': stamp,
                'node': self.node_name,
                'component': component,
                'duration_ms': f'{float(duration_ms):.6f}',
                'num_detections': num_detections,
                'num_lidar_points': num_lidar_points,
                'num_clusters': num_clusters,
                'num_confirmed_objects': num_confirmed_objects,
            })
        if len(self.samples) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.enabled or not self.samples:
            return
        path = pathlib.Path(self.csv_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.FIELDS
            )
            if write_header:
                writer.writeheader()
            writer.writerows(self.samples)
        self.samples = []
