"""
run_optimization_example.py
===========================
run_optimization.py filled in for one subject (sub-01, two seizures).

Usage: python this_script.py <beta> <sigma>
Example: python this_script.py 2.0 1.5

Output: {save_data}/derivatives/tvb/{sub}/{sz}/optimization_b{beta}_s{sigma}/
"""

import numpy as np
import mne, glob, os, re, sys, pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import pylab as pl

sys.path.append('../vep/core/')
sys.path.append('../vep/stan/')
from vep.stan import stan
from vep.core import vep_prepare_op

data_dir = '/data/vep_study/vep_open'           # TODO change: input data (one folder per subject)
save_data = '/data/vep_study/vep_open_results'  # TODO change: results (BIDS derivatives)

# ---- Parse arguments ----
if len(sys.argv) != 3:
        print("Usage: python this_script.py <beta> <sigma>")
        print("Example: python this_script.py 2.0 1.5")
        sys.exit(1)

BETA_SOFTMAX = float(sys.argv[1])
SIGMA_HIER = float(sys.argv[2])
TAG = f'b{BETA_SOFTMAX:.1f}_s{SIGMA_HIER:.1f}'
print(f">> Running with beta={BETA_SOFTMAX}, sigma={SIGMA_HIER} (tag={TAG})")

# ---- ROI setup ----
roi_path = f'{data_dir}/roi_lut.txt'  # ROI lookup table, one label per line
with open(roi_path, 'r') as f:
        roi_raw = f.read().splitlines()
roi_names = [r.strip().strip('"') for r in roi_raw if r.strip()]

SUBCORTEX_LABELS = {"THAL"}  # subcortical ROI labels without the hemisphere prefix, e.g. "THAL" for L_THAL / R_THAL

def _strip_hemisphere(name):
        return name.split("_", 1)[1] if "_" in name else name

cortex_idx, subcortex_idx = [], []
for i, n in enumerate(roi_names):
        if _strip_hemisphere(n) in SUBCORTEX_LABELS:
                subcortex_idx.append(i)
        else:
                cortex_idx.append(i)
cortex_idx = np.array(cortex_idx, dtype=int)
subcortex_idx = np.array(subcortex_idx, dtype=int)

nn = len(roi_names)
nc = len(cortex_idx)
nsub = len(subcortex_idx)

# Cases: one entry per seizure, in the same order in every list
caselist1 = ['sub-01/sz1', 'sub-01/sz2']
caselist2 = ['sub-01_sz1', 'sub-01_sz2']

weight = 'scale'
sfreq = 512.0
stan_fname = 'vep/stan/vepmap_hierprior'  # CmdStan executable compiled from vep/stan/vepmap_hierprior.stan
hpf = 10
lpf = 0.3

import importlib
importlib.reload(vep_prepare_op)

for i in range(len(caselist1)):
        case1 = caselist1[i]
        case2 = caselist2[i]

        pid = case1.split('/')[0]
        subj_raw_dir = f'{data_dir}/{pid}/func'
        results_dir = f'{save_data}/derivatives/tvb/{case1}/optimization_{TAG}'
        os.makedirs(results_dir, exist_ok=True)
        basicfilename = f'{case2}_{weight}'

        vhdrname = f'{subj_raw_dir}/{case2}_seeg.npz'
        data_sim = np.load(vhdrname)
        gain = data_sim['gain']
        seeg = data_sim['seeg']
        SC = data_sim['SC']
        ch_names = data_sim['ch_names'].tolist()

        for sub in ['logs', 'figures', 'Rfiles', 'OptimalBT', 'figuresBT', 'RfilesBT']:
                os.makedirs(f'{results_dir}/{sub}', exist_ok=True)

        seeg_4feature = seeg
        data = vep_prepare_op.prepare_data_feature_op(SC, seeg_4feature.T, gain, sfreq, hpf, lpf)

        ns_full = data['gain'].shape[0]
        gain_cortex = data['gain'][:, cortex_idx]
        SC_full = data['SC']
        SC_sub_cortex = SC_full[np.ix_(subcortex_idx, cortex_idx)]
        SC_sub_cortex_sum = SC_sub_cortex.sum(axis=1)

        ds_freq = max(1, int(data['slp'].shape[0] / 150))
        slp = data['slp'][0:-1:ds_freq]
        slp = slp - np.min(slp)
        snsr_pwr = (slp ** 2).mean(axis=0)
        nt = slp.shape[0]

        stan_data = {
                'nn': nn, 'nc': nc, 'nsub': nsub, 'ns': ns_full, 'nt': nt,
                'cortex_idx': (cortex_idx + 1).astype(int),
                'subcortex_idx': (subcortex_idx + 1).astype(int),
                'gain': gain_cortex,
                'SC': data['SC'],
                'SC_sub_cortex': SC_sub_cortex,
                'SC_sub_cortex_sum': SC_sub_cortex_sum,
                'slp': slp, 'snsr_pwr': snsr_pwr,
                'x0_mu': -3.0 * np.ones(nn),
                'beta_softmax': BETA_SOFTMAX,
                'sigma_hier': SIGMA_HIER,
        }

        param_init = {
                'x0': -3.0 * np.ones(nn),
                'amplitude': 1.0, 'offset_param': 0, 'K': 1.0, 'tau0': 20,
                'x_init': -2.0 * np.ones(nn), 'z_init': 3.5 * np.ones(nn),
                'eps_slp': 1.0, 'eps_snsr_pwr': 1.0,
        }

        fname_suffix = f'{basicfilename}_hpf{hpf}_lpf{lpf}'
        stan.rdump(f'{results_dir}/Rfiles/param_init_{fname_suffix}.R', param_init)
        stan.rdump(f'{results_dir}/Rfiles/fit_data_{fname_suffix}.R', stan_data)

        ii = np.random.randint(0, stan_data['ns'] - 1, 100)
        for id_bt, ibt in enumerate(ii):
                idata = dict(stan_data)
                idata['slp']      = np.delete(stan_data['slp'], ibt, 1)
                idata['snsr_pwr'] = np.delete(stan_data['snsr_pwr'], ibt)
                idata['gain']     = np.delete(stan_data['gain'], ibt, 0)
                idata['ns']       = idata['gain'].shape[0]
                stan.rdump(f'{results_dir}/RfilesBT/fit_data_{fname_suffix}_{id_bt}.R', idata)

        cmd_boots = (
                f"make -k -j70 -B -f vep/stan/vep_parallel.mk N=99 "
                f"RESULTS_DIR={results_dir} FNAME_SUFFIX={fname_suffix} STAN_FNAME={stan_fname} | nl"
        )
        print(f'  >> {case2} ({TAG})...')
        os.system(cmd_boots)
