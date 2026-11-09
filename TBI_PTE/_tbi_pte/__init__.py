"""
TBI-PTE analysis package.

Public API for subject I/O, EEG utilities, spike review, and PTZ-day plots.
"""

from .subject import EEGSubject
from .spike_review import SpikeReview
from .utils import (
    perform_mne_preprocessing,
    calculate_tfr,
    calculate_psd,
    select_data_segment,
    plot_eeg_manual,
    rms,
    select_channels,
    zscore,
)
from .plots import (
    compare_seizure_counts,
    compare_ed_counts,
    avg_marker_duration,
    longest_marker_duration,
    plot_latency,
    plot_ptz_doses_until_marker,
)

__all__ = [
    "EEGSubject",
    "SpikeReview",
    "perform_mne_preprocessing",
    "calculate_tfr",
    "calculate_psd",
    "select_data_segment",
    "plot_eeg_manual",
    "rms",
    "select_channels",
    "zscore",
    "compare_seizure_counts",
    "compare_ed_counts",
    "avg_marker_duration",
    "longest_marker_duration",
    "plot_latency",
    "plot_ptz_doses_until_marker",
]
