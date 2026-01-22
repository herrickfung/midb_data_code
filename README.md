# Data and codes for "Individual differences in artificial neural networks capture individual differences in human behavior"

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
git clone https://github.com/herrickfung/midb_data_and_code.git
cd midb_data_and_code
```

### 2. Set up a Python enviornment (optional but recommended)
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
8. [indimap](https://github.com/herrickfung/IndiMap)==0.0.1

To install these dependencies, 
```bash
pip install -r requirements.txt
pip install git+https://github.com/herrickfung/IndiMap.git
```

---

## Contents

The repository includes the following structure:

- **`analysis/`**: Contains all codes for reproducing the results and figures. Run `analysis/analyze.py` to generate all results and figures. The first run will download the necessary data and processed results from OSF into this directory, which may take some time.

- **`human_expt/`**: Contains all codes for the 10-choice blurry object recognition experiment that run in a web browser, programmed in JS with jsPsych 7.3.3

- **`model_script/`**: Contains all codes for training and testing multiple instances of ANNs.

- **`requirements.txt`**: Lists all the required Python dependencies for the project.

---

## Citation
If you use any materials from this project, please cite:

Fung, H., Murty, N. A. R., & Rahnev, D. (2025). Human-like individual differences emerge from random weight initializations in neural networks (p. 2025.10.25.684448). bioRxiv. [https://doi.org/10.1101/2025.10.25.684448](https://doi.org/10.1101/2025.10.25.684448)

---

## Enquires
[Herrick Fung](mailto:herrickfung@gmail.com)