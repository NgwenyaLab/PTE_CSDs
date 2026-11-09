# Seizure susceptibility after TBI and cortical spreading depolarizations

Analysis code accompanying Udeozor et al., *Seizure susceptibility following a single acute cluster of cortical spreading depolarizations in a rat model of traumatic brain injury* (Experimental Neurology, https://doi.org/10.1016/j.expneurol.2026.116113).

## Overview

Traumatic brain injury (TBI) is a major risk factor for post-traumatic epilepsy, and cortical spreading depolarizations (CSDs) occur in a large fraction of patients with moderate-to-severe TBI. Whether an acute cluster of CSDs changes later epileptogenesis is unknown. This study tested that question in adult male Sprague Dawley rats using a 2×2 design: lateral fluid percussion injury or sham surgery, followed by two hours of KCl-induced CSDs or saline. Animals were followed for about seven months with intermittent video-EEG, a pentylenetetrazol (PTZ) seizure-susceptibility challenge at six months, and histology at the end of the study.

The manuscript reports three related sets of outcomes:

- **Spontaneous ictal activity**, scored before and after PTZ. Seizures were rhythmic, high-amplitude events lasting more than 10 s and staged on a modified Racine scale. Transient ictal events (TIEs) were shorter discharges (spike-and-wave under 10 s, or spikes of about 20–70 ms) that occurred with a behavioral jerk.
- **PTZ challenge.** Repeated low doses of PTZ were given until a generalized (Racine 4–5) seizure or a maximum of nine doses. Outcomes included latency, seizure count, duration of the longest seizure, number of doses to a generalized seizure, TIE count, and electrographic intensity. Intensity was the root-mean-square (RMS) amplitude of each animal’s longest seizure, and the typical TIE amplitude, both taken from the lead ipsilateral to injury and z-scored to a pre-challenge baseline.
- **Histopathology** of the hippocampus and injured cortex (volume, parvalbumin, calretinin, and ZnT3). Those measurements were not computed in this repository. Cortical volume is used here only as a correlate of the EEG measures.

Spontaneous seizures were uncommon and limited to injured animals. TBI, with or without CSDs, increased the electrographic intensity of PTZ-induced seizures relative to sham. A single acute CSD cluster did not significantly raise seizure incidence. CSDs were associated with lower hilar ZnT3 immunopositivity, and greater cortical tissue loss tracked with higher seizure RMS.

## How this repository fits the study

This folder is the EEG analysis used for the PTZ day and for the custom statistics in the methods. Chronic seizures and behavioral Racine scores were identified by a blinded reviewer in NeuroScore. Those scores, and the animal-level summary table built from them, are the inputs to the R notebooks. The Python code is the “custom Python v3.11” pipeline described in the methods: it loads the exported EDF recordings and Excel markers, detects candidate TIEs as peaks more than five standard deviations from each animal’s baseline, supports visual rejection of artifact, and computes z-scored seizure RMS on the channel ipsilateral to injury. Reviewed spike lists are written to CSV and then modeled in R.

Group comparisons follow the manuscript. Binary incidence of seizures and TIEs, before and after PTZ, is tested with Firth’s penalized logistic regression so that groups with zero events can still be fit. Continuous PTZ outcomes that were not normally distributed, including seizure count, use an aligned-rank-transform two-way ANOVA. Spike amplitude is also modeled at the event level with a linear mixed-effects model (amplitude predicted by group, with a random intercept for animal). Pairwise Pearson correlations relate cortical volume to the continuous EEG measures.

## Layout

```
TBI_PTE/
├── _tbi_pte/                  Python package (I/O, detection, PTZ plots)
├── ptz_day_analysis.ipynb     PTZ-day seizure timing and counts
├── spike_counting_intensity copy.ipynb
│                              TIE review and seizure RMS
└── R_scripts/                 Group statistics and cortical-volume correlations
```

Raw EDF files, marker spreadsheets, the animal-level EEG table, and per-animal spike CSVs are not in this folder. The notebooks point at those files on a local data directory.

## Python

The notebooks import `_tbi_pte` from this directory. They were run with Python 3.11, MNE-Python, NumPy, SciPy, pandas, and Matplotlib.

### `_tbi_pte`

Shared library for one animal’s recording. It reads EDF files and Excel event markers, aligns marker clock times to the start of the recording, and bandpass-filters the EEG. Spike detection lives here: peaks on the absolute z-scored trace, a minimum gap between peaks, and optional exclusion of samples that fall inside Racine-labeled seizures. Detected events can be reviewed interactively, saved, and reloaded. Seizure intensity is the RMS of the channel of interest, z-scored to that animal’s baseline, summarized by default on the longest seizure. Small plotting helpers summarize Racine counts, seizure duration, latency from the first PTZ injection, and the number of PTZ doses before a generalized seizure.

### `ptz_day_analysis.ipynb`

Per-animal description of the PTZ challenge from the marker files. EEG is loaded only to place marker times on the recording clock. The notebook then plots Racine-stage counts, mean and longest duration of generalized seizures (Racine 4–6), latency from the first PTZ injection to the first Racine 5–6 event, and how many PTZ doses preceded the first generalized seizure.

### `spike_counting_intensity copy.ipynb`

PTZ-day TIE detection and seizure intensity. Recordings are filtered at 1–50 Hz. On the challenge file, candidate spikes are peaks of absolute z-score above 5, at least 1 s apart, from the start of the file through 15 min after the last PTZ dose. Peaks inside Racine seizures are dropped. Each detection is then accepted or rejected by hand. Included spikes are saved as CSV (`z`-scored amplitude and related fields) for the R analyses. The same notebook reports the RMS of the longest seizure on the ipsilateral channel after z-scoring to baseline. Animals whose EEG was too poor for this scoring are skipped in the notebook.

## R

R notebooks in `R_scripts/` use the animal-level workbook (`eeg_datatable_082126.xlsx`) and, where noted, the spike CSVs written by the Python review. Factor columns split the four experimental groups into injury (sham vs TBI) and CSD (saline vs CSD). Two notebooks label the CSD arm `SD` in the sheet (`Sham+SD`, `TBI+SD`); the spike notebooks use `CSD` (`Sham+CSD`, `TBI+CSD`). Both refer to the same 2×2 design. Analyses were run in R 4.5 with the tidyverse, `readxl`, `ggplot2`, `ARTool`, `emmeans`, `lme4`, `lmerTest`, and `logistf`.

### `prepost_PTZ_firthLR.Rmd`

Incidence of at least one seizure, and of at least one TIE, in the pre-PTZ and post-PTZ recording periods. Firth logistic regression is used because some groups had no events. The notebook prints the factorial model and draws the percent of animals with an event in each group.

### `PTZ_Seizures_ART.Rmd`

Aligned-rank-transform two-way ANOVA (`Injury × SDs`) for a chosen numeric column in the EEG table, with FDR-adjusted contrasts for the four groups and for the two main effects. This is the analysis used for PTZ outcomes that failed a normality check, including seizure count. The column to test is set by index at the bottom of the notebook.

### `spike_analysis_unilateral_original.Rmd`

Event-level model of TIE amplitude on one hemisphere. Absolute z-scored amplitudes are read from the spike CSVs (the notebook is pointed at `spike_files_contra`), and a linear mixed-effects model predicts amplitude from group with a random intercept for animal. The notebook also shows the spike distribution and group summaries used to decide between a mean and a median description. The same model form is the secondary TIE-amplitude analysis in the paper; the primary contrast used the lead ipsilateral to injury.

### `spike_analysis_bilateral_Spike_level.Rmd`

Same spike lists, restricted to animals recorded on both sides (`spike_files_ipsi` and `spike_files_contra`). Each TIE remains one observation. The mixed model is amplitude predicted by group, hemisphere, and their interaction, with a random intercept for animal.

### `spike_analysis_bilateral_Animal_level.Rmd`

Bilateral companion to the spike-level model. Amplitudes are first reduced to one median per animal per hemisphere, then fit with the same group × hemisphere mixed model. Summaries are the mean of those animal medians.

### `CorticalVolume_EEG_Correlations.Rmd`

Pearson correlations (Spearman is available in the same function) between cortical volume and the continuous EEG measures in `CorticalVolume_EEG_correlation.xlsx`. This is the `cor.test` analysis in the methods: cortical tissue loss against TIE count, seizure RMS, and TIE amplitude. The notebook can write one scatter plot per pair.

## Citation

Udeozor M, DeSana AJ, Osbourne JA, Mallabo K, Sood PS, Hopkins TM, Turbiarz SW, Danzer SC, Hartings JA, Ngwenya LB. Seizure susceptibility following a single acute cluster of cortical spreading depolarizations in a rat model of traumatic brain injury. *Experimental Neurology*. https://doi.org/10.1016/j.expneurol.2026.116113
