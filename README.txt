Field-line versus field-sheet reconnection in vacuum gravity
Francisco R. Villatoro
Vacuum data and software, version 1.0, 22 November 2026

Suggested citation
Francisco R. Villatoro (2026), Vacuum data and software accompanying
“Field-line versus field-sheet reconnection in vacuum gravity”, version 1.0,
supplementary files.

Extract Reconnection_vacuum_reproducibility.zip and
Reconnection_vacuum_validation.zip into the SAME directory. Together they
create Reconnection_vacuum_v1.0/. Run the commands below from that folder.
Each archive has its own checksum manifest, and their paths are disjoint.

What is in the two files
------------------------
Reconnection_vacuum_reproducibility.zip supplies analysis/, data/, figures/
and the original 24 cluster runs with solver, inputs and outputs. The full
original campaign is in full/Fase5_validacion_cientifica_20261115/audit_picasso/.
Its scripts and results are retained under their original names; the later
analysis described in the paper starts from the saved fields.

Reconnection_vacuum_validation.zip supplies the 21 additional evolutions
(18 amplitude, initial-data or refinement checks and 3 near-event runs), the
independent equation checks, raw arrays, evaluation scripts and the original
campaign provenance, under full/Fase5_validacion_cientifica_20261115/.
The configuration inventory is in review/vacuum_configuration_inventory.txt.
data/observables.csv contains the 45 observable combinations measured across
amplitudes and material labels. The 45 configurations and 45 observable rows
are different enumerations.

Checks and figure regeneration
------------------------------
Use Python 3.10 or newer with NumPy, SciPy, SymPy and Matplotlib.
The version 1.0 analysis was checked with CPython 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, SymPy 1.14.0 and Matplotlib 3.10.8. The cycle below reads
archived outputs. It takes seconds to minutes and requires neither Slurm nor
access to the original cluster. Run Python with assertions enabled.

    sha256sum -c SHA256_reproducibility.txt --quiet
    sha256sum -c SHA256_validation.txt --quiet
    python3 analysis/verify_polynomials.py
    python3 analysis/verify_boundary_bounds.py
    python3 analysis/verify_geometric_algebra.py
    python3 analysis/verify_sheet_transport.py
    python3 analysis/verify_twist_identities.py
    python3 analysis/check_archived_runs.py
    python3 analysis/make_figures.py
    python3 analysis/check_manuscript_numbers.py

The first five Python scripts check the analytic coefficients, the boundary
bounds, the initial constraints and selected-form algebra, the sheet
transport identities, and the identities, reduced vacuum equations and
perturbative coefficients for the twisting Killing vector of Section 6 of the
article. The polynomial
certificate contains 24 identities plus three exact rounding tests. The
boundary check contains 11 polynomial identities and a numerical cross-check
along 256 trajectories. The twist check contains 24 statements and writes
analysis/twist_checks.json. The archive check verifies all 21 ancillary
evolution hashes and directly compares the illustrated scalar fields and
boundary coordinates with the original before/after snapshots.
make_figures.py produces figures/connectivity_switch.pdf,
figures/amplitude_scaling.pdf, their PNG counterparts, data/observables.csv,
data/event_table_rows.tex and data/paper_numbers.json.

check_manuscript_numbers.py compares the archived table entries (data/
reported_table_rows.tex) with the underlying observable and constraint values,
and checks aggregate values used in the article. It also accepts source files:

    python3 analysis/check_manuscript_numbers.py path/to/main.tex path/to/supplement.tex

The data/ directory preserves the empirical sensitivity indicators and their
individual terms. They measure the changes seen under the checks documented in
the article. The plots are reproducible from the data on other systems, with
possible rendering differences by backend, font and package version.

Analysis starting from raw evolutions
-------------------------------------
In a working copy of the extracted archives, run

    python3 full/Fase5_validacion_cientifica_20261115/scientific_validation/reproduce.py

With the supplied evolutions already in place, this command reruns the event
and refinement analyses and traces the near-event slices. It writes its JSON
results in scientific_validation/. The version used for the plots is under
data/. To verify integrity use sha256sum before rerunning scripts that regenerate
archived files. The original cluster runtime is recorded in
audit_picasso/results/fase5_20261115_01/runtime.json (under full/
Fase5_validacion_cientifica_20261115/); supplementary_runs.json records the
local runtime and individual configurations of the 21 new evolutions.

Rerunning the nonlinear evolutions
----------------------------------
The archived solver runs locally on a compute machine. For example:

    campaign=full/Fase5_validacion_cientifica_20261115/audit_picasso/results/fase5_20261115_01
    python3 "$campaign/input/solver.py" --config "$campaign/task_000/config.json" --output rerun/task_000

Each of the other 23 task directories has its own config.json. To rerun the
additional 21 evolutions, use a separate copy of the two extracted archives.
Move scientific_validation/extra/ and scientific_validation/near_events/ out
of that working copy before starting scientific_validation/reproduce.py.
The script creates new arrays when the recorded output directories are absent.
The working copy keeps the originals available for comparison.
