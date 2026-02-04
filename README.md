# Data and code for "Individual differences in artificial neural networks capture individual differences in human behavior"

## Project Description
This project examines how individual differences in artificial neural networks can capture and predict individual differences in human behavior across accuracy, confidence, and response time in digits and object recognition tasks.

Full paper available on bioRxiv.

This project is also presented at NeurIPS 2025 UniReps & Data on the Brain & Mind workshops. Earlier version of the paper is available [here](https://www.biorxiv.org/content/10.1101/2025.10.25.684448v1.abstract).


---

## Installation

To set up this project locally, follow these steps:

### 1. Clone the repository:
Clone the repository to your local machine and navigate into the project folder:
```bash
git clone https://github.com/herrickfung/midb_data_code.git
cd midb_data_code
```

### 2. Set up a Python environment (optional but recommended)
```bash
python3 -m venv ./venv/
source ./venv/bin/activate
```

### 3. Install dependencies:
1. Python3.9
2. numpy==2.0.2
3. pandas==2.2.3
4. scipy==1.13.1
5. matplotlib==3.9.4
6. pingouin==0.5.5
7. seaborn==0.13.2
8. requests==2.32.5
9. [indimap](https://github.com/herrickfung/indimap)==0.1.1

To install these dependencies, 
```bash
pip install -r requirements.txt
```

---

## Contents

The repository includes the following structure:

- **`analysis/`**: Contains all code for reproducing the results and figures. 
     - By default, running `python3 analyze.py` downloads the data (hosted long-term on [OSF](https://osf.io/n6m7b/files/dqc9s); no manual download is needed), loads precomputed results and generates the figures reported in the paper. 
     - To recompute all results from raw data, pass the `--recompute` flag. This will take about 30 minutes.

- **`human_expt/`**: Contains all code for the 10-choice blurry object recognition experiment that run in a web browser, programmed in JS with jsPsych 7.3.3

- **`model_script/`**: Contains all code for training and testing multiple instances of ANNs, including codes for subsetting EcoSet. 

- **`requirements.txt`**: Lists all the required Python dependencies for the project.

---

## Citation
If you use any materials from this project, please cite:

<!-- Fung, H., Murty, N. A. R., & Rahnev, D. (2026). Individual differences in artificial neural networks capture individual differences in human behavior. -->
Fung, H., Murty, N. A. R., & Rahnev, D. (2025). Human-like individual differences emerge from random weight initializations in neural networks (p. 2025.10.25.684448). bioRxiv. [https://doi.org/10.1101/2025.10.25.684448](https://doi.org/10.1101/2025.10.25.684448)

```bibtex
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