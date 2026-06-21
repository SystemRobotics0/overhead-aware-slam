# Scripts

Compact index of analysis and summary scripts.

| Script | Output |
|---|---|
| `summarize_nav_results.py` | `results/navigation_summary*.csv` |
| `summarize_runtime_profile.py` | `results/runtime_profile_live_summary.csv` |
| `benchmark_detector_runtime.py` | `results/runtime_profile.csv` |
| `make_map_figure.py` | `results/map_figure.png` |

`eval_common.py` contains shared map-loading helpers used by the scripts above.

Typical refresh:

```bash
python3 scripts/make_map_figure.py
```

Runtime summary:

```bash
python3 scripts/summarize_runtime_profile.py \
  --samples results/runtime_profile_live_raw.csv \
  --output results/runtime_profile_live_summary.csv
```
