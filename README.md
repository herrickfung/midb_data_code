# Data and code for "Individual differences in artificial neural networks capture individual differences in human behavior"

## Project Description
This project examines how individual differences in artificial neural networks can capture and predict individual differences in human behavior across accuracy, confidence, and response time in digits and object recognition tasks.

Full paper available on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.02.10.705061).

This project is also presented at NeurIPS 2025 UniReps & Data on the Brain & Mind workshops. Earlier version of the paper is available [here](https://www.biorxiv.org/content/10.1101/2025.10.25.684448v1.abstract).


---

## Installation

To set up this project locally, follow these steps:


### 1. System Requirements
This code was developed and tested on a standard desktop/laptop computer. No specialized hardware (e.g., a GPU) is required. Reproducing all analyses downloads about 7 GB of data archives, which extract to about 13 GB (see [Reproducing the analyses](#reproducing-the-analyses) for the requirements per analysis).

This code is supported on *macOS* and *Linux* and has been tested on macOS Ventura (v13.2) and Red Hat Enterprise Linux (RHEL) 8.1, using Python 3.9. The setup instructions and required Python dependencies are listed below.

### 2. Clone the repository:
Clone the repository to your local machine and navigate into the project folder:
```bash
git clone https://github.com/herrickfung/midb_data_code.git
cd midb_data_code
```

### 3. Set up a Python environment (optional but recommended)
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
```

### 4. Install dependencies:
The code requires Python 3.9 and the following packages:
1. numpy==2.0.2
2. pandas==2.2.3
3. scipy==1.13.1
4. matplotlib==3.9.4
5. pingouin==0.5.5
6. seaborn==0.13.2
7. requests==2.32.5
8. [indimap](https://github.com/herrickfung/indimap)==0.1.2

To install these dependencies, 
```bash
pip install -r requirements.txt
```

---

## Contents

The repository includes the following structure:

- **`analysis/`**: Contains all code for reproducing the results and figures (see [Reproducing the analyses](#reproducing-the-analyses) below).
     - `analyze.py`: single entry point for all analyses.
     - `util/dataset.py`: loads and preprocesses the human and ANN data into IndiMap configurations.
     - `util/plotting.py`: plotting and statistics for every figure in the paper.
     - `dataset/`, `IndiMap_results/`, `graphs/`: created when the analyses are run (downloaded data, precomputed results, and output figures and statistics); not tracked by git.

- **`human_expt/`**: Contains all code for the 10-choice blurry object recognition experiment that run in a web browser, programmed in JS with jsPsych 7.3.3

- **`model_script/`**: Contains all code for training and testing multiple instances of ANNs, including codes for subsetting EcoSet. 

- **`requirements.txt`**: Lists all the required Python dependencies for the project.

---

## Reproducing the analyses

All commands are run from the `analysis/` directory:
```bash
cd analysis
python3 analyze.py                          # run all four analyses
python3 analyze.py main control             # run a subset
python3 analyze.py untrained --recompute    # recompute results from the raw data instead of loading them
```

The code is self-contained: the raw data and precomputed results are hosted on [Harvard Dataverse](https://doi.org/10.7910/DVN/DVVXJL), and `analyze.py` downloads and extracts only the archives needed for the requested analyses (no manual download is needed). Archives are downloaded on the first run only; later runs reuse the extracted files. By default, precomputed results are loaded and figures and statistics are regenerated in a few minutes (about 2.5 minutes for all four analyses on a laptop, once the data are downloaded). Figures are saved to `analysis/graphs/<analysis>/`, and all statistics printed during plotting are also saved to `analysis/graphs/<analysis>/stats.txt`.

| Analysis | Paper figures | Data archives (besides `midb_data`) | Disk space (download / extracted) |
|---|---|---|---|
| `main` | Fig. 1b–5; Supp. Fig. 1–3, 6–11 | `midb_results_standard_mnist`, `midb_results_standard_ecoset10` | 2.6 GB / 6.7 GB |
| `accuracy_control` | Fig. 4a; Supp. Fig. 4 | `midb_results_standard_mnist`, `midb_results_accuracy_control_*` | 5.2 GB / 9.2 GB |
| `control` | Fig. 4b; Supp. Fig. 5 | `midb_results_standard_mnist`, `midb_results_control` | 1.5 GB / 3.9 GB |
| `untrained` | Supp. Fig. 12 | `midb_results_standard_mnist`, `midb_results_untrained` | 1.9 GB / 4.5 GB |

Each row includes `midb_data` and `midb_results_standard_mnist`, which are shared across analyses and downloaded only once. All four analyses together need 6.8 GB to download and 12.6 GB extracted. Each archive is deleted after it is extracted.

`midb_data` contains the trial-level human and ANN data and is always downloaded. With `--recompute`, only `midb_data` is downloaded and all IndiMap results are recomputed from it (1000 bootstrap iterations per model, which takes considerably longer).

**Note on the pseudo-instance control analysis (`control`).** For each architecture, this analysis creates pseudo-instances from each of 60 ANN instances by varying the stimulus noise at test, runs the full IndiMap analysis on each of the 60 pseudo-instance populations, and averages the results across the 60 populations (correlations are averaged in Fisher-z space). Recomputing all 60 populations and merging them takes a very long time, so only the merged results are provided (`IndiMap_results/control/merged/`). The code for the merge is included (`python3 analyze.py control --remerge`), but it requires the per-instance data, which is available upon request.

---

## Citation
If you use any materials from this project, please cite the paper(s) below. If you use the data, please also cite the dataset on Harvard Dataverse: [https://doi.org/10.7910/DVN/DVVXJL](https://doi.org/10.7910/DVN/DVVXJL).

Fung, H., Murty, N. A. R., & Rahnev, D. (2026). Individual differences in artificial neural networks capture individual differences in human behavior (p. 2026.02.10.705061). bioRxiv. [https://doi.org/10.64898/2026.02.10.705061](https://www.biorxiv.org/content/10.64898/2026.02.10.705061)

Fung, H., Murty, N. A. R., & Rahnev, D. (2025). Human-like individual differences emerge from random weight initializations in neural networks (p. 2025.10.25.684448). bioRxiv. [https://doi.org/10.1101/2025.10.25.684448](https://doi.org/10.1101/2025.10.25.684448)

```bibtex
@article{Fung2026IndividualDifferencesANN,
  title = {Individual differences in artificial neural networks capture individual differences in human behavior},
  author  = {Fung, Herrick and Murty, N. A. R. and Rahnev, Dobromir},
  journal = {bioRxiv},
  year    = {2026},
  pages   = {2026.02.10.705061},
  doi     = {10.64898/2026.02.10.705061},
  url     = {https://doi.org/10.64898/2026.02.10.705061}
}

@article{Fung2025HumanLikeID,
  title   = {Human-like individual differences emerge from random weight initializations in neural networks},
  author  = {Fung, Herrick and Murty, N. A. R. and Rahnev, Dobromir},
  journal = {bioRxiv},
  year    = {2025},
  pages   = {2025.10.25.684448},
  doi     = {10.1101/2025.10.25.684448},
  url     = {https://doi.org/10.1101/2025.10.25.684448}
}
```

---

## Enquiries
[Herrick Fung](mailto:herrickfung@gmail.com)