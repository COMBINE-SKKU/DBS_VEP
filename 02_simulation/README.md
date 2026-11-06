# 2. Simulation

`src/STIM_virtual_epileptic_seeg_ret_DBS.py` simulates each seizure with and without thalamic DBS,
using the x0 and K fitted in `01_optimization` (see "Inputs for the simulation" in its README). The folder layout follows virtual_epilepsy_patient_cohort.

## Files

```
02_simulation/
├── src/
│   ├── STIM_virtual_epileptic_seeg_ret_DBS.py          # main script (template: fill in the cases)
│   ├── STIM_virtual_epileptic_seeg_ret_DBS_example.py  # same script filled in for one subject
│   ├── utils_functions/
│   │   ├── model_2populations_stim.py   # DBS Epileptor model (EpileptorStim2Populations)
│   │   ├── integrator.py                # adapted Heun integrator
│   │   ├── model.py
│   │   └── vep_prepare_ret.py
│   └── utils_data/VepMrtrixLut.txt
└── tvb/                                 # modified TVB 2.8.3 (subset used by the simulation)
```

## Model

The Epileptor (Jirsa et al., 2014) is extended with two state variables (`src/utils_functions/model_2populations_stim.py`):

- **m** (slope): accumulates the stimulus. The stimulated ROIs receive the pulse train directly;
  every other ROI receives it through SC with conduction delays.
- **s** (DBS gating): when m crosses its threshold, the gate opens and shifts the effective excitability of z.

## Usage

1. Set `data_dir` and `save_data` at the top of `src/STIM_virtual_epileptic_seeg_ret_DBS.py`.
2. Fill in the case lists, one entry per seizure, in the same order in every list:

| List | Content | Example |
|------|---------|---------|
| `caselist1` | `{subject}/{seizure}` | `'sub-01/sz1'` |
| `caselist2` | `{subject}_{seizure}` (must match the case names in `K_mean_all_scale_b{beta}_s{sigma}.txt`) | `'sub-01_sz1'` |
| `caselist3` | `{subject}` | `'sub-01'` |
| `timelist` | length of the seizure EEG in samples (512 Hz) | `10240` (20 s) |
| `onset`, `offset` | seizure onset / offset per seizure | `5` |

3. Run with the `beta` and `sigma` of the optimization run whose x0 and K are used:

```bash
cd 02_simulation
python -m src.STIM_virtual_epileptic_seeg_ret_DBS 2.0 1.5
```

Each case is simulated twice: no-stim (`do_stim = False`) and DBS (`do_stim = True`).

## Outputs

`{save_data}/derivatives/tvb/{sub}/{sz}/simulation_b{beta}_s{sigma}_a{a}_tt{tt}/{no_stim, dbs}/`

| File | Content |
|------|---------|
| `{sub}_{sz}_roi_sig.npy` | source signal x1 - x2, shape (time, regions) |
| `img/` | figures of SC and source signals |

## Changes to TVB

`tvb/` is a subset of TVB 2.8.3 with these changes:

- `simulator/simulator.py`: propagates the stimulus from the stimulated ROIs through SC (with delays) at each step;
  `Simulator.run` also returns the per-region dynamics.
- `simulator/integrators.py`: forwards the propagated stimulus to the model.
- `basic/neotraits/_core.py`: compatibility aliases for newer NumPy versions.

TVB is distributed under the GNU GPL v3 (see `tvb/LICENSE`).
