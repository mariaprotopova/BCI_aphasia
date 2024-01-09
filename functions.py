#!/usr/bin/env python
# coding: utf-8

# In[3]:
import os
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import csv
import gc

path = 'C:/Users/HP pavilion X360/Desktop/BCI'

def tsv_to_fif(subj, run, path=path,
              lpass = 20, hpass = .5, sfreq = 500):
    stim_code = pd.read_csv(f'{path}/BCI_HSE{subj[1:]}/stim_code/stim_code_{subj}R{run}.csv')
    data = np.zeros((stim_code.shape[0], 13))
    with open(f"{path}/BCI_HSE{subj[1:]}/tsv/BCI_HSE{subj}R{run}.ascii") as tsv:
        i = 0
        for line in csv.reader(tsv, dialect="excel-tab"):
            if i == 0:
                columns = line
                i += 1
            elif i <= stim_code.shape[0]:
                data[i-1] = line
                i += 1
    smart = pd.DataFrame(data, columns = columns)
    smart['STIM101'] = stim_code
    n_channels = 2
    
    # ch_names = columns
    ch_names = ['Fp1', 'Fp2', 'STIM101']
    ch_types = ['eeg']*2 + ['stim'] #['eeg']*2 + ['misc']*5 + ['eog']*3 + ['misc']*3
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(smart.values[:, [0, 1, 13]].T, info)

    path_out = f'{path}/BCI_HSE{subj[1:]}/preprocessed/'
    os.makedirs(path_out, exist_ok = True)
    
    raw_filt = raw.copy().filter(
        hpass, lpass, l_trans_bandwidth='auto', picks = ['eeg'],
        h_trans_bandwidth='auto', filter_length='auto', phase='zero',
        fir_window='hamming', fir_design='firwin', n_jobs=4)
    raw_filt.save(f'{path_out}/{subj}R{run}_filt_raw.fif', overwrite = True)
    return raw, raw_filt


# In[2]:


def find_ica(subj, run, path = path):
    path_out = f'{path}/BCI_HSE{subj[1:]}/preprocessed/'
    from mne.preprocessing import ICA
    get_ipython().run_line_magic('matplotlib', 'qt')
    raw = mne.io.read_raw(f"BCI_HSE{subj[1:]}/preprocessed/{subj}R{run}_filt_raw.fif", preload = True)
    ica = ICA(n_components=2, max_iter="auto", random_state=97)
    ica.fit(raw)
    fig = ica.plot_sources(raw, show_scrollbars=False, show = False)
    fig.savefig(f'ICA/{subj}_{run}_ica_comp.png', dpi = 300, bbox_inches = 'tight')
    # raw.save(f'{path_out}/{subj}R{run}_filt_raw_ica.fif', overwrite = True)
    return raw, ica


# In[4]:


def plot_stat_comparison_timecourse_1samp_new(comp1, comp2, time, y_low = None, y_high = None, title='demo_title',
                         comp1_label='comp1', comp2_label='comp2'):
    assert(comp1.shape[1] == comp2.shape[1] == len(time))
    fig = plt.figure()
    ax1 = fig.add_subplot()
    
    plt.rcParams['axes.facecolor'] = 'none'
    plt.xlim(time[0], time[-1])
    if y_low is not None:
        plt.ylim(y_low, y_high)
    plt.plot([0, 0.000], [-500, 500], color='k', linewidth=1, linestyle='--', zorder=1)
    plt.plot([-10000, 10000], [0, 0.00], color='k', linewidth=1, linestyle='--', zorder=1)
    plt.plot(time, comp1.mean(axis = 0), color='turquoise', linewidth=1.5, label=comp1_label)
    plt.plot(time, comp2.mean(axis = 0), color='salmon', linewidth=1.5, label=comp2_label)
    
    # ax1.set_ylabel(r'$\mu$V')
    ax1.set_xlabel('Time (ms)')
    
    plt.xticks(ticks=np.arange(time[0], time[-1], 100))
    plt.tick_params(labelsize = 12)
    ax1.legend()

    ci_1 = np.std(comp1, axis = 0)/np.sqrt(comp1.shape[0])
    ci_2 = np.std(comp2, axis = 0)/np.sqrt(comp2.shape[0])
    ax1.fill_between(time, (comp1.mean(axis = 0)-ci_1), (comp1.mean(axis = 0)+ci_1), color='turquoise', alpha=.2)
    ax1.fill_between(time, (comp2.mean(axis = 0)-ci_2), (comp2.mean(axis = 0)+ci_2), color='salmon', alpha=.2)

    plt.title(title, fontsize = 12)
    plt.show()
    return fig


# In[ ]:




