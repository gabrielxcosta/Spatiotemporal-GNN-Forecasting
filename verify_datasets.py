# -*- coding: utf-8 -*-
"""
Benchmark script for inspecting datasets in torch_geometric_temporal.

Displays:
- Import timing and CUDA info
- Automatic loading of all official datasets
- Meta-information: |V| (nodes), |E| (edges), T (snapshots),
  number of features, signal type, and inferred temporal resolution.
"""

import json
from time import perf_counter
from pprint import pprint
import torch

# =============================================================
# 1. Import timing measurements
# =============================================================
t0 = perf_counter()
import time as _time
t_import_time = perf_counter()

import importlib as _importlib
t_import_importlib = perf_counter()

try:
    from torch_geometric_temporal import dataset as tgtd
    from torch_geometric_temporal.signal import (
        StaticGraphTemporalSignal,
        DynamicGraphTemporalSignal
    )
    t_import_tgtd = perf_counter()
except Exception as e:
    tgtd = None
    t_import_tgtd = perf_counter()
    print("⚠️ Error importing torch_geometric_temporal:", e)

print(f"\n== Import timings (seconds) ==")
print(f" import time module:    {t_import_time - t0:0.4f}s")
print(f" import importlib:      {t_import_importlib - t_import_time:0.4f}s")
print(f" import tgtd:           {t_import_tgtd - t_import_importlib:0.4f}s")

# =============================================================
# 2. CUDA initialization check
# =============================================================
cuda_times = {}
t_cuda_start = perf_counter()
torch_cuda_available = torch.cuda.is_available()
t_cuda_check = perf_counter()
cuda_times['cuda_available_check'] = t_cuda_check - t_cuda_start

if torch_cuda_available:
    t_init_start = perf_counter()
    try:
        torch.cuda.init()
    except Exception:
        pass
    t_init_end = perf_counter()
    cuda_times['cuda_init'] = t_init_end - t_init_start

    t_devname_start = perf_counter()
    try:
        dev_name = torch.cuda.get_device_name(0)
    except Exception as e:
        dev_name = f"Error getting device name: {e}"
    t_devname_end = perf_counter()
    cuda_times['get_device_name'] = t_devname_end - t_devname_start
    cuda_times['device_name'] = dev_name
else:
    cuda_times['note'] = "CUDA not available"

print("\n== CUDA timings & info ==")
pprint(cuda_times)

# =============================================================
# 3. Datasets to test
# =============================================================
datasets_to_test = [
    ("ChickenpoxDatasetLoader", dict(lags=4)),
    ("PedalMeDatasetLoader", dict(lags=4)),
    ("WikiMathsDatasetLoader", dict(lags=8)),
    ("WindmillOutputLargeDatasetLoader", dict(lags=8)),
    ("WindmillOutputMediumDatasetLoader", dict(lags=8)),
    ("WindmillOutputSmallDatasetLoader", dict(lags=8)),
    ("METRLADatasetLoader", dict(num_timesteps_in=12, num_timesteps_out=12)),
    ("PemsBayDatasetLoader", dict(num_timesteps_in=12, num_timesteps_out=12)),
    ("PemsAllLADatasetLoader", dict(lags=12)),
    ("PemsDatasetLoader", dict(lags=12)),
    ("EnglandCovidDatasetLoader", dict(lags=8)),
    ("MontevideoBusDatasetLoader", dict(lags=4)),
    ("MTMDatasetLoader", dict(frames=16)),
]

results = {}
global_start = perf_counter()

# =============================================================
# 4. Load and inspect datasets
# =============================================================
if tgtd is None:
    print("\n❌ torch_geometric_temporal not imported correctly.")
else:
    for name, params in datasets_to_test:
        entry = {"params": params, "status": None, "timings": {}, "meta": {}}
        print(f"\n--- Testing dataset: {name}  params={params} ---")

        # --- Lookup loader class ---
        t_lookup_start = perf_counter()
        try:
            loader_cls = getattr(tgtd, name)
            t_lookup_end = perf_counter()
            entry['timings']['lookup'] = t_lookup_end - t_lookup_start
        except Exception as e:
            entry['status'] = "failed"
            entry['error'] = f"Loader class not found: {e}"
            results[name] = entry
            print(" ❌ Loader class not found:", e)
            continue

        # --- Initialize loader ---
        t_init_loader_start = perf_counter()
        try:
            loader = loader_cls()
            t_init_loader_end = perf_counter()
            entry['timings']['loader_init'] = t_init_loader_end - t_init_loader_start
        except Exception as e:
            entry['status'] = "failed"
            entry['error'] = f"Loader init error: {e}"
            results[name] = entry
            print(" ❌ Error initializing loader:", e)
            continue

        # --- Get dataset ---
        t_get_start = perf_counter()
        try:
            dataset_iter = loader.get_dataset(**params)
            dataset = list(dataset_iter)
            t_get_end = perf_counter()
            entry['timings']['get_dataset'] = t_get_end - t_get_start
        except Exception as e:
            entry['status'] = "failed"
            entry['error'] = f"get_dataset error: {e}"
            results[name] = entry
            print(" ❌ Error in get_dataset:", e)
            continue

        if not dataset:
            entry['status'] = "failed"
            entry['error'] = "Empty dataset after list()"
            results[name] = entry
            print(" ❌ Empty dataset.")
            continue

        # --- Successful load ---
        entry['status'] = "success"
        entry['meta']['snapshots'] = len(dataset)
        print(f" ✅ {name} loaded successfully ({len(dataset)} snapshots)")

        # =============================================================
        # Extract metadata from first snapshot
        # =============================================================
        first_snap = dataset[0]
        if hasattr(first_snap, "x") and first_snap.x is not None:
            num_nodes = first_snap.x.shape[0]
            num_features = first_snap.x.shape[1]
        else:
            num_nodes = num_features = "?"

        if hasattr(first_snap, "edge_index") and first_snap.edge_index is not None:
            num_edges = first_snap.edge_index.shape[1]
        else:
            num_edges = "?"

        entry["meta"]["num_nodes"] = num_nodes
        entry["meta"]["num_edges"] = num_edges
        entry["meta"]["num_features"] = num_features

        print(f"   ▸ |V| (nodes): {num_nodes}")
        print(f"   ▸ |E| (edges): {num_edges}")
        print(f"   ▸ Features per node: {num_features}")

        # Determine signal type
        signal_type = (
            "StaticGraphTemporalSignal"
            if isinstance(first_snap, StaticGraphTemporalSignal)
            else "DynamicGraphTemporalSignal"
            if isinstance(first_snap, DynamicGraphTemporalSignal)
            else "Unknown"
        )
        entry["meta"]["signal_type"] = signal_type
        print(f"   ▸ Signal type: {signal_type}")

        # Infer temporal resolution by dataset name
        if "Covid" in name:
            temporal_res = "Daily"
        elif "PedalMe" in name:
            temporal_res = "Hourly (~30 min)"
        elif "Pems" in name or "METRLA" in name:
            temporal_res = "5 minutes"
        elif "Wiki" in name:
            temporal_res = "Weekly"
        elif "Windmill" in name:
            temporal_res = "Hourly"
        elif "Chickenpox" in name:
            temporal_res = "Weekly"
        elif "Montevideo" in name:
            temporal_res = "1 minute (GPS)"
        else:
            temporal_res = "Unknown"

        entry["meta"]["temporal_resolution"] = temporal_res
        results[name] = entry
        print(f"   ▸ Temporal resolution: {temporal_res}")
        print(f"   ▸ Status: ✅ {entry['status']}")

# =============================================================
# 5. Summary
# =============================================================
total_elapsed = perf_counter() - global_start
fail_list = [n for n, r in results.items() if not str(r.get('status', '')).startswith("success")]

print("\n\n=== Final Summary ===")
print(f"✅ Successful: {len(results) - len(fail_list)}")
print(f"❌ Failed: {len(fail_list)}")

if fail_list:
    print("\n--- Failed Datasets ---")
    for n in fail_list:
        print(f"  ❌ {n}")

if len(results) - len(fail_list) > 0:
    print("\n--- Successfully Loaded Datasets ---")
    for n, r in results.items():
        if str(r.get('status', '')).startswith("success"):
            print(f"  ✅ {n}")

# Save results
out_fname = "dataset_load_times.json"
with open(out_fname, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n📁 Results saved to: {out_fname}")

# =============================================================
# 6. Detailed Analysis
# =============================================================
print("\n\n=== Detailed Analysis of Successfully Loaded Datasets ===")

for name, entry in results.items():
    if not str(entry.get("status", "")).startswith("success"):
        continue

    meta = entry["meta"]
    params = entry["params"]
    print(f"\n📘 Dataset: {name}")
    print(f"   ▸ Parameters: {params}")
    print(f"   ▸ Snapshots (T): {meta.get('snapshots', '—')}")
    print(f"   ▸ |V| (nodes): {meta.get('num_nodes', '?')}")
    print(f"   ▸ |E| (edges): {meta.get('num_edges', '?')}")
    print(f"   ▸ Features per node: {meta.get('num_features', '?')}")
    print(f"   ▸ Signal type: {meta.get('signal_type', '?')}")
    print(f"   ▸ Temporal resolution: {meta.get('temporal_resolution', '?')}")
    print(f"   ▸ Status: ✅ {entry['status']}")