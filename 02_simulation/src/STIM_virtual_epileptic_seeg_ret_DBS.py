#!/usr/bin/env python
# coding: utf-8
import sys
if len(sys.argv) != 3:
    print("Usage: python -m src.STIM_virtual_epileptic_seeg_ret_DBS <beta> <sigma>")
    sys.exit(1)
TAG = f'b{float(sys.argv[1]):.1f}_s{float(sys.argv[2]):.1f}'
print(f">> DBS simulation for tag={TAG}")

# Importing libraries
import matplotlib
import random
random.seed(0)
from tvb.simulator.lab import *
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import matplotlib as mpl
import matplotlib.image as mpimg
import colorednoise as cn
import os.path as op
from pybv import write_brainvision
import pandas as pd
import time as tm
import csv
import json
from src.utils_functions import vep_prepare_ret
from src.utils_functions.model import EpileptorStim
from src.utils_functions.model_2populations_stim import EpileptorStim2Populations
from src.utils_functions.integrator import HeunDeterministicAdapted, HeunStochasticAdapted
roi = vep_prepare_ret.read_vep_mrtrix_lut()
from multiprocessing import Pool, cpu_count

data_dir = '/data/vep_study/vep_open'           # TODO change: input data (one folder per subject)
save_data = '/data/vep_study/vep_open_results'  # TODO change: results (BIDS derivatives)

clinical_hypothesis = False
run = 1

# Cases: one entry per seizure, in the same order in every list
caselist1 = ['{SUBJECT}/{SEIZURE}']  # e.g. 'sub-01/sz1'
caselist2 = ['{SUBJECT}_{SEIZURE}']  # e.g. 'sub-01_sz1'
caselist3 = ['{SUBJECT}']            # e.g. 'sub-01'
timelist = [N_SAMPLES]               # length of the seizure EEG in samples (512 Hz)
onset = [ONSET]                      # seizure onset per seizure
offset = [OFFSET]                    # seizure offset per seizure

timelist = np.array(timelist) / 512

tt_val = '3'      # time scale condition: tt = 10.7143 / 3 / time_length (see process_case)
a_val = -0.2      # coupling.Difference a
target = 'THAL'   # stimulated ROIs: L_{target}, R_{target}

def process_case(case_info):
        case1, case2, sub, ses, time_length, simulation_length, Ks_val, onset_val, offset_val, tt_cond, a_val, target, do_stim = case_info
        
        if tt_cond == '3':
                ##################################### SETTING #####################################
                # set tt variable
                # formula : tt = (0.05*1500*dt) / (0.00035*1000*n)
                # formula : tt = (fit r * fit timepoints * sim dt) / (sim r * 1000ms * time_length)
                tt = 10.7143/3 / time_length # dt -> 0.05
                tt = round(tt, 3)
        elif tt_cond == '4':
                tt = 10.7143/4 / time_length # dt -> 0.05
                tt = round(tt, 3)
        else:
                tt = 10.7143/5 / time_length # dt -> 0.05
                tt = round(tt, 3)
                
        save_base = f'{save_data}/derivatives/tvb/{sub}/{ses}/simulation_{TAG}_a{a_val}_tt{tt_cond}'

        sub_zip_path = f'{data_dir}/{sub}/tvb'
        con = connectivity.Connectivity.from_file(f'{sub_zip_path}/connectivity.vep.zip')
        roi = con.region_labels
        nb_regions = len(roi)
        x0_csv = pd.read_csv(f'{save_data}/derivatives/tvb/{case1}/optimization_{TAG}/x0_transformed.csv')
        x0_dict = dict(zip(x0_csv['Brain regions'], x0_csv['x0']))
        x0_vals = np.array([x0_dict.get(region) for region in roi])

        vhdrname = f'{data_dir}/{sub}/func/{case2}_seeg.npz'
        data_sim = np.load(vhdrname)
        invgain = data_sim['gain']
        seeg_xyz_names = data_sim['ch_names']
        seeg_xyz_names = seeg_xyz_names.tolist()

        # Gain matrix
        gain = invgain

        if do_stim:
                savedir = f'{save_base}/dbs'
        else:
                savedir = f'{save_base}/no_stim'
        
        os.makedirs(savedir, exist_ok=True)
        os.makedirs(f'{savedir}/img', exist_ok=True)
        # # 1. Importing patient specific information
        # - Choosing a patient to work on.
        # - Defining the folder to save the results for the patient.
        # - Importing the structural connectivity, the list of regions from the parcellation.

        # Connectivity
        con.tract_lengths = con.tract_lengths
        inv_tract = np.divide(1.0, con.tract_lengths, 
                      out=np.zeros_like(con.tract_lengths), 
                      where=con.tract_lengths!=0)
        con.configure()
        
        SC = con.weights
        fig = plt.figure(figsize=(10,10))
        plt.imshow(SC,norm=matplotlib.colors.LogNorm(vmin=1e-6, vmax=SC.max()))
        plt.title(f'Normalized SC (log scale)',fontsize=12, fontweight='bold')
        plt.xticks(np.r_[:len(roi)], roi, rotation = 90)
        plt.yticks(np.r_[:len(roi)], roi)
        plt.colorbar()
        fig.tight_layout()
        plt.savefig(f'{savedir}/img/{case2}_SC.png')
        
        plot = False
        if plot:
                nsensor, nchan = np.shape(gain)
                fig = plt.figure(figsize=(24,14))
                im = plt.imshow(gain,norm=matplotlib.colors.LogNorm(vmin=gain.min(), vmax=gain.max()))
                plt.xticks(np.r_[:len(roi)], roi, rotation = 90, fontsize=8)
                plt.yticks(np.r_[:len(seeg_xyz_names)], seeg_xyz_names, fontsize=8)
                plt.xlabel('Region#', fontsize=12)
                plt.ylabel('Channel#', fontsize=12)
                plt.title(f' Gain Matrix (log scale)',fontsize=12, fontweight='bold')
                plt.colorbar(im, fraction=0.03, pad=0.01)
                fig.tight_layout()

        # # 3. Setting up the simulation with TVB
        # - Here we use the Epileptor
        # - We set up the needed ingredients for the simulator (model, connectivity, coupling, integrator, initial conditions, monitors and more)

        if clinical_hypothesis:
                sub_folder = 'clinical_hypothesis'
                print()
        else:
                sub_folder = 'vep_hypothesis'

        print(f'Simulation using EZ from \"{sub_folder}\".')

        # ## 3b. Configure the stimulation
        # - Select a pair of electrodes to apply the stimulation from
        # - Calculate the effect of those electrodes' stimulation on the network nodes using the gain matrix

        # # Calculate the projection of the SEEG electrode stimulation onto the network nodes
        # Stimuli parameters
        freq = 130 # Hz
        sfreq = 512 # how many steps there are in one second
        dt = 0.05
        onset = 800 # ms
        n_sec = simulation_length * 0.2 # ms
        stim_length = onset + n_sec
        T = 1/freq * sfreq # pulse repetition period [ms]
        tau = 0.15  # pulse duration [ms]
        I = 2.0  # pulse intensity [mA]

        n_regions = len(con.region_labels)
        class vector1D(equations.DiscreteEquation):
                equation = equations.Final(default="emp")
        # stimulus parameters
        eqn_t = vector1D()
        parameters = {'T': T, 'tau': tau, 'amp': I, 'onset': onset}
        pulse1, _ = equations.PulseTrain(parameters=parameters).get_series_data(max_range=stim_length, step=dt)
        pulse1_ts = [p[1] for p in pulse1]
        parameters = {'T': T, 'tau': tau, 'amp': I, 'onset': onset + tau}
        pulse2, _ = equations.PulseTrain(parameters=parameters).get_series_data(max_range=stim_length, step=dt)
        pulse2_ts = [p[1] for p in pulse2]
        pulse_ts = np.asarray(pulse1_ts) - np.asarray(pulse2_ts)
        expected_length = len(np.r_[0.0: simulation_length: dt])
        stimulus_ts = np.hstack((pulse_ts[:-1], np.zeros(expected_length - len(pulse_ts) + 1)))
        eqn_t.parameters['emp'] = np.copy(stimulus_ts)
        
        # Stimuli applied from the SEEG electrode
        print("Applied Stimuli")
        plt.figure(figsize=(30, 1))
        plt.plot(stimulus_ts)
        plt.figure(figsize=(30, 1))
        plt.plot(stimulus_ts[0:50000])

        # Stimuli weighted accross the network nodes according to the gain matrix
        stim_weights = np.zeros((nb_regions))
        selected_roi = [f'L_{target}', f'R_{target}']
        selected_idx = [x0_csv.index[x0_csv['Brain regions'] == roi][0] for roi in selected_roi]
        if do_stim:
                weights = np.array([6, 6])
                stim_weights[selected_idx] = np.array([weights])
        stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                        connectivity=con,
                                        weight=stim_weights)
        stimulus.configure_space()
        stimulus.configure_time(np.arange(0., len(stimulus_ts), 1))

        print("Stimuli weighted accross the network nodes")
        def plot_pattern(pattern_object):
                """
                pyplot in 2D the given X, over T.
                """
                plt.figure(figsize=(30, 25))
                plt.subplot(221)
                plt.bar(np.r_[0:pattern_object.spatial_pattern.shape[0]], pattern_object.spatial_pattern[:, 0])
                plt.xticks(np.r_[:len(roi)], roi, rotation=90, size=6)
                plt.title("Space")
                plt.subplot(223)
                plt.plot(pattern_object.time.T, pattern_object.temporal_pattern.T, linewidth=0.1)
                plt.title("Time")
                plt.subplot(122)
                plt.imshow(pattern_object()[:, 0:50000], aspect="auto", cmap='bwr')
                plt.colorbar()
                plt.title("Stimulus")
                plt.yticks(np.r_[:len(roi)], roi)
                plt.xlabel("Time")
                plt.ylabel("Space")
        plot_pattern(stimulus)

        epileptors = EpileptorStim2Populations(variables_of_interest=['x1', 'y1', 'z', 'x2', 'y2', 'g', 'm'])
        epileptors.r = np.array([0.00035])
        r_val = 0.0006
        epileptors.r2 = np.ones(len(roi)) * (r_val)
        epileptors.Istim = np.ones(len(roi)) * (0.)
        epileptors.n_stim = np.zeros(len(roi))
        epileptors.Iext = np.ones(n_regions) * (3.1)
        epileptors.Iext2 = np.ones(n_regions) * (0.45)
        epileptors.Ks = np.ones(n_regions) * Ks_val
        epileptors.Kf = np.ones(n_regions) * (0)
        epileptors.Kvf = np.ones(n_regions) * (0)
        epileptors.threshold = np.ones(len(roi)) * (0.1)
        epileptors.tt = np.array([tt])
        
        ## DBS parameter
        epileptors.tau_s = np.array([100.0])   # s decay time constant
        ##################################

        period = 3.
        dt = 0.05

        epileptors.x0 = x0_vals
        epileptors.inv_tract = inv_tract
        epileptors.stim_L_idx = np.array([selected_idx[0]])
        epileptors.stim_R_idx = np.array([selected_idx[1]])
        epileptors.threshold[selected_idx[0]] = 1.0
        epileptors.threshold[selected_idx[1]] = 1.0

        epileptors.n_stim = np.ones(len(roi)) # TODO fix this once and for all by removing it !!!

        # Coupling
        coupl = coupling.Difference(a=np.array([a_val]))

        # Integrator
        noisy_sim = False
        if noisy_sim:
                nsig = np.array([0., 0., 0., 1e-5, 1e-5, 0., 0.])
                hiss = noise.Additive(nsig=nsig)
                hiss.ntau = 1  # for gaussian distributed coloured noise
                heunint = HeunStochasticAdapted(dt=dt, noise=hiss)
        else:
                nsig = []
                heunint = HeunDeterministicAdapted(dt=dt)

        # Monitors
        mon_tavg = monitors.TemporalAverage(period=period)

        init_cond = np.array([-1.46242601e+00, -9.69344913e+00, 3.1, -1.05214059e+00, -4.95543740e-20, -1.98742113e-01, 0.0, 0.0])
        init_cond_reshaped = np.repeat(init_cond, nb_regions).reshape((1, len(init_cond), nb_regions, 1))

        # Simulator
        sim = simulator.Simulator(model=epileptors,
                                initial_conditions=init_cond_reshaped,
                                connectivity=con,
                                stimulus=stimulus, # TODO change
                                coupling=coupl,
                                integrator=heunint,
                                conduction_speed=np.inf,
                                monitors=[mon_tavg])

        sim.configure()
        print("Starting simulation...")
        tic = tm.time()
        ttavg, dyn, H_dyn = sim.run(simulation_length=simulation_length)
        print("Finished simulation.")
        print('execute for ', (tm.time() - tic) / 60.0, 'mins')
        save_fig = False

        # # 4. Visualizing the simulated data on the source level
        tts = ttavg[0][0]
        tavg = ttavg[0][1]
        srcSig = tavg[:, 0, :, 0] - tavg[:, 3, :, 0]
        start_idx = 0
        end_idx = tts.shape[0] - 1
        srcSig_normal = srcSig / np.ptp(srcSig)

        #Plot raw time series
        figure = plt.figure(figsize=(24, 14))
        mpl.rcParams['font.family'] = 'Arial'
        colors = ['red', 'green', 'magenta', 'blue', 'black', 'purple']

        ymins, ymaxs, labels = [], [], []

        # Fig 1) visualize source signal of Epileptor
        for ind in range(len(roi)):
                trace = srcSig[start_idx:end_idx, ind]
                trace_centered = trace - trace.mean()
                
                baseline_shift = trace[start_idx] - trace.mean()
                trace_aligned = trace_centered - baseline_shift + ind

                color = colors[ind % len(colors)]
                plt.plot(tts[start_idx:end_idx], trace_aligned, color=color, lw=0.7)

                ymins.append(trace_aligned.min())
                ymaxs.append(trace_aligned.max())
                labels.append(roi[ind])

        plt.title("Epileptors Time Series", fontsize=12)
        plt.yticks(np.arange(len(roi)), labels, fontsize=10)
        y_margin = 1.0
        plt.ylim(min(ymins) - y_margin, max(ymaxs) + y_margin)
        plt.xlim([tts[start_idx], tts[end_idx - 1]])
        plt.tight_layout()
        plt.savefig(f'{savedir}/img/{case2}_source_all.png')
        np.save(f'{savedir}/{case2}_roi_sig.npy', srcSig)
        
        r_val = 0.0006
        
        plt.rcParams['font.family'] = 'Arial'
        for roi_name in roi:
                id = np.where(np.array(roi) == roi_name)[0][0]
                print(id)
                y = tavg[:, :, [id], 0]
                print(tavg.shape)
                print('Region ' + roi[id])
                fig, axs = plt.subplots(5, 1, tight_layout=True, figsize=(9, 9))
                axs[0].set_title(f"After Stimulation {roi[id]}", fontsize=15)
                axs[0].plot(tts[:], y[:, 0, :] - y[:, 3, :], 'blue', linewidth=0.5)
                axs[0].set_ylabel('x1-x2', fontsize=15)
                axs[1].plot(tts[:], y[:, 2, :], 'gray', linewidth=1.5)
                axs[1].set_ylabel('z', fontsize=15)
                axs[2].plot(tts[:], y[:, 6, :], 'gray', linewidth=1.5)
                axs[2].axhline(y=sim.model.threshold[id], color='C3', linewidth=2.0)
                axs[2].set_xlabel('Time[ms]', fontsize=15)
                axs[2].set_ylabel('m', fontsize=15)
                axs[3].plot(tts[:], H_dyn[:, id], 'gray', linewidth=1.5)
                axs[3].set_ylabel('H')
                axs[4].plot(stimulus.time.T, stimulus.temporal_pattern.T * stimulus.weight[id], 'gray', linewidth=1.5)
                axs[4].set_ylabel('Istim', fontsize=15)
                axs[4].set_xlabel('Time[ms]', fontsize=15)
                for i in [0, 1, 3]:
                        axs[i].tick_params(labelbottom=False)
                plt.savefig(f'{savedir}/img/{case2}_{roi_name}_source_all.png', dpi=300, bbox_inches='tight')
                plt.close(fig)


if __name__ == '__main__':
        # Prepare case information for parallel processing
        case_info_list = []
        for i in range(len(caselist1)):
                time_length = timelist[i]
                simulation_length = timelist[i] * 1000
                simulation_length = np.round(simulation_length)
                subid, sesid = caselist1[i].split('/')
                        
                K_df = pd.read_csv(f'{save_data}/derivatives/tvb/K_mean_all_scale_{TAG}.txt', header=None, sep='\t')
                sub_ind = K_df[K_df[0] == caselist2[i]].index[0]
                ks = float(K_df.iloc[sub_ind, 1])
                ## extract target roi
                roi_file = f'{data_dir}/roi_lut.txt'
                with open(roi_file, 'r', encoding='utf-8') as f:
                        rois = [line.strip() for line in f if line.strip()]
                                                
                for do_stim in [False, True]:
                        case_info = (
                                caselist1[i],
                                caselist2[i],
                                caselist3[i],
                                sesid,
                                time_length,
                                simulation_length,
                                ks,
                                onset[i],
                                offset[i],
                                tt_val,
                                a_val,
                                target,
                                do_stim,
                                )
                        case_info_list.append(case_info)

        # Get number of CPU cores
        num_cores = cpu_count()
        print(f"Using {num_cores} CPU cores for parallel processing")
        
        # Create a pool of workers and process cases in parallel
        with Pool(processes=num_cores) as pool:
                results = pool.map(process_case, case_info_list)

        # Print results
        for result in results:
                print(result)