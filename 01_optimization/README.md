# 1. Optimization

`examples/run_optimization.py` fits the Epileptor 2D model (Jirsa et al., 2014) to the scalp EEG data features of each seizure
with Stan (L-BFGS optimization), and estimates x0 per region and the global coupling K.
The folder layout follows VEP_INS_workflow.

## Files

```
01_optimization/
├── examples/
│   ├── run_optimization.py           # builds the Stan inputs and runs the bootstrap fits (template: fill in the cases)
│   └── run_optimization_example.py   # same script filled in for one subject
└── vep/
    ├── core/vep_prepare_op.py        # data features (SLP, channel power)
    └── stan/
        ├── stan.py                   # Stan I/O
        ├── vepmap_hierprior.stan     # Epileptor 2D model with the hierarchical subcortical x0 prior
        └── vep_parallel.mk           # runs the bootstrap fits in parallel
```

## Model

- **Observation**: the scalp EEG is explained by the cortical regions only; the SLP over time and the channel power are fitted.
- **Cortical x0**: prior normal(-3, 1).
- **Subcortical x0**: hierarchical prior centered on an SC-weighted softmax of the x0 of the connected cortical regions
  (temperature `beta`), with a fixed spread `sigma`.

## Usage

1. Set `data_dir`, `save_data` and the case lists (`caselist1`, `caselist2`) at the top of `examples/run_optimization.py`.
2. Set `SUBCORTEX_LABELS`: the subcortical ROI labels without the hemisphere prefix (e.g. `"THAL"` for `L_THAL` and `R_THAL`).
   These regions get the hierarchical x0 prior; all other regions are treated as cortex.
3. Compile the Stan model (see Installation in the main README), then run with `beta` and `sigma`:

```bash
cd 01_optimization
python -m examples.run_optimization 2.0 1.5
```

For each seizure, the script fits the model to 100 bootstrap data sets, each leaving out one randomly chosen EEG channel.

## Outputs

`{save_data}/derivatives/tvb/{sub}/{sz}/optimization_b{beta}_s{sigma}/`

| Folder | Content |
|--------|---------|
| `Rfiles/` | Stan data and initial values |
| `RfilesBT/` | bootstrap Stan data |
| `OptimalBT/` | fitted parameters per bootstrap (CmdStan CSV) |
| `logs/` | CmdStan logs |

### Inputs for the simulation

Summarize the bootstrap fits in `OptimalBT/` into the two files read by `02_simulation`:

| File | Content |
|------|---------|
| `{sub}/{sz}/optimization_b{beta}_s{sigma}/x0_transformed.csv` | fitted x0 per region; columns `Brain regions`, `x0`; rows in the region order of `connectivity.vep.zip` |
| `K_mean_all_scale_b{beta}_s{sigma}.txt` (in `derivatives/tvb/`) | mean global coupling K per seizure, one line per seizure: `{sub}_{sz}<TAB>{K}` |
