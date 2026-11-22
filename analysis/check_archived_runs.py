"""Check the stored runs and trace the plotted data back to the source snapshots."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
full = root / "full" / "Fase5_validacion_cientifica_20261115"
picasso = full / "audit_picasso" / "results" / "fase5_20261115_01"
validation = full / "scientific_validation"

for tree in (picasso, validation):
    if not tree.is_dir():
        raise SystemExit("Extract both version 1.0 supplementary archives into the same directory.")

runs = sorted(picasso.glob("task_[0-9][0-9][0-9]"))
assert len(runs) == 24
assert all((run / "evolution.npz").is_file() for run in runs)
assert all(json.loads((run / "summary.json").read_text())["status"] == "SUCCESS" for run in runs)

controls = json.loads((validation / "supplementary_runs.json").read_text())["cases"]
assert len(controls) == len({c["directory"] for c in controls}) == 21
for case in controls:
    path = validation / case["directory"] / "evolution.npz"
    assert sha256(path.read_bytes()).hexdigest() == case["evolution_sha256"], path

with np.load(root / "data" / "connectivity_figure_data.npz") as plotted, np.load(
    validation / "near_events" / "e08" / "evolution.npz"
) as raw:
    for key in ("x", "y", "epsilon"):
        np.testing.assert_array_equal(plotted[key], raw[key])
    for time, side in zip(plotted["s"], ("before", "after"), strict=True):
        entries = np.flatnonzero(raw["gaussian_slices_targets"] == time)
        assert len(entries) == 1
        idx = entries[0]
        np.testing.assert_array_equal(plotted["phi_" + side], raw["phi_slices"][idx])
        np.testing.assert_array_equal(plotted["boundary_" + side], raw["boundary_xy"][idx])

print("Verified 24 cluster runs, hashes of 21 additional evolutions, and both plotted slices.")
