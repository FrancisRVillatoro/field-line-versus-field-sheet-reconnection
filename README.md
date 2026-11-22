# Reproducibility data and software for “Field-line versus field-sheet reconnection in vacuum gravity”

Version 1.0 of the vacuum reproducibility archive of the article “Field-line versus field-sheet reconnection in vacuum gravity” by Francisco R. Villatoro (2026).

The files of this repository are those of the two supplementary ZIP files of the article, `Reconnection_vacuum_reproducibility.zip` and `Reconnection_vacuum_validation.zip`, extracted into one directory. The repository root therefore plays the role of the directory `Reconnection_vacuum_v1.0/` named in `README.txt`, which describes the contents and gives every command. Only this file, `LICENSE`, `CITATION.cff`, `.zenodo.json`, `requirements.txt`, `.gitattributes` and `.gitignore` were added for the repository.

## Contents

- `analysis/` holds the exact-arithmetic and symbolic checks, including those for the twisting Killing vector, the figure regeneration and the comparison of the reported tables with the archived outputs.
- `data/` and `figures/` hold the observables, the table rows and the two figures of the article.
- `full/Fase5_validacion_cientifica_20261115/` holds the 24 cluster evolutions (`audit_picasso/`) and the 21 additional evolutions (`scientific_validation/`), with the solver, configurations, raw arrays and runtime provenance.
- `review/vacuum_configuration_inventory.txt` lists the 45 configurations.
- `SHA256_reproducibility.txt` and `SHA256_validation.txt` give the checksums of every archived file.

## Checks

The scripts need Python 3.10 or newer with NumPy, SciPy, SymPy and Matplotlib (`requirements.txt`). From the repository root, verify the files with

    sha256sum -c SHA256_reproducibility.txt --quiet
    sha256sum -c SHA256_validation.txt --quiet

and then run the commands listed in `README.txt`. They read the archived outputs, take seconds to minutes and do not require a computing cluster.

## License

MIT (see `LICENSE`).

## Citation

Please cite the article and this repository through its Zenodo record (`CITATION.cff`).
