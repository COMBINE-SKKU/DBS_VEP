# Thalamic connectivity predicts target-dependent efficacy of deep brain stimulation in epilepsy

Code for:

**Thalamic connectivity predicts target-dependent efficacy of deep brain stimulation in epilepsy.** (manuscript in preparation)

The citation will be added when the paper is published.

----

## Overview

Patient-specific virtual brain modeling of thalamic deep brain stimulation (DBS) in epilepsy,
using a modified Epileptor model in The Virtual Brain (TVB).

## System Requirements

**Operating System:**
- Linux (tested on Ubuntu 20.04)
- Windows and macOS may work but are not officially supported

**Software Dependencies** (all installed by `environment.yml`):
- **Python 3.8** with NumPy 1.22.4, SciPy 1.7.3, Numba 0.58.1, pandas 2.0.3, Matplotlib 3.7.5, MNE 1.6.1, colorednoise 2.2.0, pybv 0.7.5
- **[CmdStan](https://mc-stan.org/users/interfaces/cmdstan) 2.36** and GNU make (optimization)
- **TVB 2.8.3** (modified copy included in `02_simulation/tvb/`, no installation needed)

**Hardware Requirements:**
- **CPU:** Multi-core processor. The optimization runs 70 bootstrap fits in parallel (`make -j70`),
  and the simulation runs one case per CPU core (tested on a 104-core server).
- No specialized hardware required

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/COMBINE-SKKU/DBS_VEP.git
   cd DBS_VEP
   ```

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate epileptor-dbs
   ```

3. Compile the Stan model once (`environment.yml` sets `$CMDSTAN`):
   ```bash
   cd 01_optimization
   make -C $CMDSTAN $PWD/vep/stan/vepmap_hierprior
   ```

## Project Structure

The pipeline has two stages:

- **`01_optimization/`** - Fits x0 per region and the global coupling K to the scalp EEG of each seizure
- **`02_simulation/`** - Simulates DBS on the fitted virtual brains

Each directory contains a README file with usage instructions and outputs.

## Demo

The patient data are not shared. The `*_example` scripts show the full pipeline for one subject with two seizures
(`sub-01`, `sz1` and `sz2`); put your data in the layout below (see Data layout) and run:

```bash
cd 01_optimization
python -m examples.run_optimization_example 2.0 1.5        # 1. Optimization (beta = 2.0, sigma = 1.5)

cd ../02_simulation
python -m src.STIM_virtual_epileptic_seeg_ret_DBS_example 2.0 1.5   # 2. Simulation (no-stim and DBS)
```

The outputs of each stage are described in its README.

## Data layout

Inputs (`data_dir`, one folder per subject):

```
/data/vep_study/vep_open/
├── roi_lut.txt                        # ROI lookup table, one label per line (e.g. L_THAL)
└── {sub}/
    ├── tvb/connectivity.vep.zip       # TVB connectivity (weights, tract lengths, region labels)
    └── func/{sub}_{sz}_seeg.npz       # seeg (scalp EEG, 512 Hz), gain, SC, ch_names
```

Results are written to `save_data` as BIDS derivatives (`derivatives/tvb/{sub}/{sz}/`).
`{sub}` and `{sz}` are the subject and seizure names (e.g. `sub-01`, `sz1`).

## References

- Jirsa, V. K., Stacey, W. C., Quilichini, P. P., Ivanov, A. I., & Bernard, C. (2014). On the nature of seizure dynamics. *Brain*, 137(8), 2210-2230.
- Dollomaja, B., Wang, H. E., Guye, M., Makhalova, J., Bartolomei, F., & Jirsa, V. K. (2025). Virtual epilepsy patient cohort: generation and evaluation. *PLOS Computational Biology*, 21(4), e1012911.
- Sanz Leon, P., et al. (2013). The Virtual Brain: a simulator of primate brain network dynamics. *Frontiers in Neuroinformatics*, 7, 10.
