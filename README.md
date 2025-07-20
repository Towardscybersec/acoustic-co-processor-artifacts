# acoustic-co-processor-artifacts

This repository contains the simulation artifacts and results for the **Acoustic Co‑Processor** research project. It includes Finite Element Method (FEM) models, post‑processing scripts, and related data files generated with FEniCS 2016.2.0 and Python 3.5.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)

   * [1. Single Waveguide Attenuation](#1-single-waveguide-attenuation)
   * [2. Crosstalk vs. Gap Analysis](#2-crosstalk-vs-gap-analysis)
   * [3. Metamaterial Spatial FFT](#3-metamaterial-spatial-fft)
   * [4. Transmission Frequency Sweep](#4-transmission-frequency-sweep)
6. [Results](#results)
7. [Folder Contents](#folder-contents)
8. [Contributing](#contributing)
9. [Citation](#citation)

---

## Project Overview

The goal of this project is to explore the performance and feasibility of an **acoustic co‑processor**—an analog computing device that uses mechanical waves in silicon to perform convolution operations with low energy consumption. The simulations cover:

* Quantifying attenuation in a single waveguide under PML boundary conditions.
* Measuring crosstalk between parallel waveguides separated by phononic barriers.
* Extracting bandgap characteristics via spatial FFT of a metamaterial lattice.
* Evaluating transmission as a function of frequency in a finite crystal.

All simulations are driven by Python scripts using **FEniCS 2016.2.0** + **mshr** and standard scientific Python libraries.

## Repository Structure

```
acoustic-co-processor-artifacts/
├── waveguide_acoustic.py  # Main simulation driver script
├── results/                      # Folder containing generated figures and data
│   ├── attenuation.png
│   ├── attenuation_db.png
│   ├── crosstalk.png
│   ├── metamaterial_fft.png
│   └── transmission_sweep.png
└── README.md                     # This file
```

## Prerequisites

* **Python 3.5**
* **FEniCS 2016.2.0** (including `dolfin` and `mshr`)
* NumPy
* Matplotlib

```bash
# Example installation via apt / pip (Ubuntu)
sudo apt-get install python3-pip python3-numpy python3-matplotlib
# Install FEniCS 2016.2.0 (if available via apt)
sudo apt-get install fenics
# Or use Docker image: fenicsproject/stable:2016.2.0

pip3 install numpy matplotlib
```

## Installation

1. Clone this repository:

   ```bash
   ```

git clone [https://github.com/](https://github.com/)\Towardscybersec/acoustic-co-processor-artifacts.git
cd acoustic-co-processor-artifacts

````
2. Create a Python virtual environment (optional but recommended):
   ```bash
python3 -m venv venv
source venv/bin/activate
````

## Usage

Run the main script to execute all four simulation stages in sequence:

```bash
python3 waveguide_acoustic_updated.py
```

Each stage will generate plots and save them to the `results/` directory. Below is a breakdown of each simulation.

### 1. Single Waveguide Attenuation

* **Script Section:** Lines 60–120
* **Description:** Models a 1 mm × 10 µm silicon waveguide with a customized PML.
* **Output:**

  * `results/attenuation.png` (linear-scale pressure amplitude)
  * `results/attenuation_db.png` (attenuation in dB)

### 2. Crosstalk vs. Gap Analysis

* **Script Section:** Lines 125–210
* **Description:** Evaluates isolation (dB) between two channels separated by phononic barrier configurations with varying gaps (5 µm, 10 µm, 20 µm, 50 µm).
* **Output:**

  * `results/crosstalk.png` (Isolation vs. gap)

### 3. Metamaterial Spatial FFT

* **Script Section:** Lines 215–290
* **Description:** Computes spatial FFT of the metamaterial’s pressure profile to estimate bandgap width.
* **Output:**

  * `results/metamaterial_fft.png` (FFT magnitude with band edges)

### 4. Transmission Frequency Sweep

* **Script Section:** Lines 295–360
* **Description:** Sweeps the operating frequency from 0.8 f₀ to 1.2 f₀ over a finite-length phononic crystal, measuring through‑power and plotting transmission in dB.
* **Output:**

  * `results/transmission_sweep.png` (Transmission vs. frequency)

## Results

All figures are available in the [`results/`](./results) folder. These illustrate:

* **Attenuation constant** (theoretical vs. simulated)
* **Channel isolation performance** for phononic barriers
* **Metamaterial bandgap** extraction via FFT
* **Frequency-dependent transmission** characteristics

Refer to the associated research paper for discussion and detailed interpretation of these plots.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.

## Citation

If you use this code or data in your research, please cite:

> K. K., "Scalable Analog Acoustic Co-Processors: A Novel Paradigm for Scientific Computing Beyond CMOS Limits," *Unpublished manuscript*, 2025.

---

*Happy simulating!*
