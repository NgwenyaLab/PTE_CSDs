"""
EEG signal-processing utilities for TBI-PTE analyses.

This module holds shared helpers used by ``EEGSubject`` and the analysis
notebooks: MNE preprocessing, time-frequency and PSD estimates, channel
selection, z-scoring, RMS, and a simple multi-channel EEG plotter.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt


def perform_mne_preprocessing(
    eeg_data,
    filt_method="fir",
    high_pass_filter=None,
    low_pass_filter=None,
    notch_freq=None,
    down_sample_rate=None,
    epoch_duration=None,
    epoch_overlap=0,
):
    """
    Copy an MNE object and optionally filter, notch, resample, and epoch it.

    Each step is skipped when its argument is ``None``. The input object is
    never modified in place.

    Parameters
    ----------
    eeg_data : mne.io.BaseRaw or mne.Epochs
        Continuous or epoched recording to preprocess.
    filt_method : str, optional
        MNE filter implementation. Default is ``"fir"``.
    high_pass_filter : float or None, optional
        High-pass cutoff in Hz. ``None`` leaves the low edge unfiltered.
    low_pass_filter : float or None, optional
        Low-pass cutoff in Hz. ``None`` leaves the high edge unfiltered.
    notch_freq : float, array-like, or None, optional
        Line-noise frequency or frequencies in Hz. ``None`` skips the notch.
    down_sample_rate : float or None, optional
        Target sampling rate in Hz. ``None`` keeps the original rate.
    epoch_duration : float or None, optional
        Fixed epoch length in seconds. ``None`` returns continuous data.
    epoch_overlap : float, optional
        Overlap between successive epochs in seconds. Default is ``0``.

    Returns
    -------
    temp : mne.io.BaseRaw or mne.Epochs
        Preprocessed copy. Type is ``Epochs`` only if ``epoch_duration``
        was provided.
    """
    # Work on a copy so the caller's Raw/Epochs object is unchanged.
    temp = eeg_data.copy()

    # Bandpass (or one-sided) filter when either cutoff is set.
    if high_pass_filter is not None or low_pass_filter is not None:
        temp.filter(
            l_freq=high_pass_filter,
            h_freq=low_pass_filter,
            method=filt_method,
            verbose=False,
        )
        print(f"Applied bandpass filter: {high_pass_filter}-{low_pass_filter}Hz")

    # Notch out line noise (and harmonics, if an array was passed).
    if notch_freq is not None:
        temp.notch_filter(freqs=notch_freq, verbose=False)
        print(f"Applied notch filter at {notch_freq}Hz")

    # Reduce temporal resolution after filtering.
    if down_sample_rate is not None:
        temp.resample(down_sample_rate, npad="auto")
        print(f"Downsampled to {down_sample_rate}Hz")

    # Cut the continuous recording into fixed-length epochs.
    if epoch_duration is not None:
        # Raw (n_chans, n_times) -> Epochs (n_epochs, n_chans, n_times_epoch)
        temp = mne.make_fixed_length_epochs(
            temp, duration=epoch_duration, overlap=epoch_overlap
        )

    return temp


def calculate_tfr(
    eeg_data,
    freq_params=None,
    down_sample_rate=None,
    method="multitaper",
    cycles=7,
    output="power",
    average_epochs=False,
):
    """
    Compute a time-frequency representation (TFR) of EEG data.

    Parameters
    ----------
    eeg_data : mne.io.BaseRaw or mne.Epochs
        Recording to decompose. Epoched data is the typical input.
    freq_params : sequence of float
        ``[fmin, fmax, fstep]`` in Hz, passed to ``numpy.arange``.
        Required; ``None`` raises ``ValueError``.
    down_sample_rate : float or None, optional
        If set, resample the copy before the TFR. Default is ``None``.
    method : str, optional
        MNE TFR method (e.g. ``"multitaper"``, ``"morlet"``).
        Default is ``"multitaper"``.
    cycles : float or array-like, optional
        Number of cycles per frequency (MNE ``n_cycles``). Default is ``7``.
    output : str, optional
        TFR quantity, typically ``"power"`` or ``"phase"``. Default is
        ``"power"``.
    average_epochs : bool, optional
        If ``True``, average the TFR across epochs. Default is ``False``.

    Returns
    -------
    tfr : mne.time_frequency.AverageTFR or mne.time_frequency.EpochsTFR
        Time-frequency estimate. Shape is
        ``(n_epochs, n_chans, n_freqs, n_times)`` when unaveraged, or
        ``(n_chans, n_freqs, n_times)`` when ``average_epochs=True``.

    Raises
    ------
    ValueError
        If ``freq_params`` is not provided.
    """
    # Copy so resampling cannot mutate the caller's object.
    temp = eeg_data.copy()

    # Optional extra downsample before a costly TFR.
    if down_sample_rate is not None:
        temp.resample(down_sample_rate, npad="auto")

    if freq_params is None:
        raise ValueError("freq_params must be provided. (e.g. [1, 50, 1])")

    # Frequency grid: [fmin, fmax) with spacing fstep.
    freqs = np.arange(freq_params[0], freq_params[1], freq_params[2])

    # Decompose; MNE returns power or phase depending on ``output``.
    tfr = temp.compute_tfr(
        method=method,
        freqs=freqs,
        return_itc=False,
        n_cycles=cycles,
        n_jobs=-1,
        average=average_epochs,  # (n_epochs, n_chans, n_freqs, n_times) -> (n_chans, n_freqs, n_times) if True
        output=output,
        verbose="ERROR",
    )

    return tfr


def calculate_psd(
    eeg_data,
    freq_params=(1, 100),
    method="multitaper",
    output="power",
    average_epochs=False,
    decibel=False,
):
    """
    Compute a power spectral density (PSD) of EEG data.

    Parameters
    ----------
    eeg_data : mne.io.BaseRaw or mne.Epochs
        Recording to transform.
    freq_params : sequence of float, optional
        ``(fmin, fmax)`` in Hz. Default is ``(1, 100)``.
    method : str, optional
        MNE PSD method. Default is ``"multitaper"``.
    output : str, optional
        PSD quantity, typically ``"power"`` or ``"complex"``. Default is
        ``"power"``.
    average_epochs : bool, optional
        If ``True`` and the input is epoched, average across epochs after
        the PSD. Default is ``False``.
    decibel : bool, optional
        If ``True``, convert power to ``10 * log10(power)``. Default is
        ``False``.

    Returns
    -------
    psds : mne.time_frequency.Spectrum or mne.time_frequency.EpochsSpectrum
        PSD object. Data array is ``(n_chans, n_freqs)`` for continuous or
        averaged input, or ``(n_epochs, n_chans, n_freqs)`` for unaveraged
        epochs.
    """
    # Copy so spectrum conversion cannot mutate the caller's object.
    temp = eeg_data.copy()

    # Multitaper PSD over the requested band.
    psds = temp.compute_psd(
        method=method,
        adaptive=True,
        normalization="full",
        fmin=freq_params[0],
        fmax=freq_params[1],
        n_jobs=-1,
        output=output,
        verbose="ERROR",
    )

    # Collapse the epoch axis when a single spectrum is desired.
    if average_epochs:
        psds = psds.average()  # (n_epochs, n_chans, n_freqs) -> (n_chans, n_freqs)

    # Optional dB scaling of the power array, in place on the Spectrum object.
    if decibel:
        psds.data = 10 * np.log10(psds.data)

    return psds


def select_data_segment(eeg_data, start_time, end_time):
    """
    Crop a copy of an MNE recording to a time window.

    Parameters
    ----------
    eeg_data : mne.io.BaseRaw or mne.Epochs
        Recording to crop.
    start_time : float
        Window start in seconds (``tmin``).
    end_time : float
        Window end in seconds (``tmax``).

    Returns
    -------
    cropped : mne.io.BaseRaw or mne.Epochs
        Copy restricted to ``[start_time, end_time]``.
        Time axis: ``(n_times)`` -> ``(n_times_window)``.
    """
    # Crop a copy so the full recording remains available on the subject.
    temp = eeg_data.copy()
    return temp.crop(tmin=start_time, tmax=end_time)


def plot_eeg_manual(
    eeg_data,
    sfreq=None,
    ch_names=None,
    alphas=None,
    colors=None,
    ylim=None,
    ylabel=None,
    title=None,
):
    """
    Overlay channels of a short EEG segment on a single matplotlib axis.

    If ``eeg_data`` is an MNE object, sampling rate, channel names, and
    the data array are read from it. A plain ``ndarray`` is also accepted
    when ``sfreq`` is supplied.

    Parameters
    ----------
    eeg_data : mne.io.BaseRaw, mne.Epochs, or ndarray
        Traces to plot. Arrays must be ``(n_chans, n_samples)``.
    sfreq : float or None, optional
        Sampling rate in Hz. Inferred from MNE info when possible.
    ch_names : list of str or None, optional
        Legend labels. Inferred from MNE info, otherwise ``ch1, ch2, ...``.
    alphas : sequence of float or None, optional
        Per-channel line opacity. ``None`` uses opaque lines.
    colors : sequence or None, optional
        Per-channel matplotlib colors. ``None`` uses the default cycle.
    ylim : tuple of float or None, optional
        Y-axis limits. ``None`` autoscales.
    ylabel : str or None, optional
        Y-axis label. Default is ``"Amplitude"``.
    title : str or None, optional
        Axes title. Default is no title.

    Returns
    -------
    None
        Displays the figure. Nothing is returned.
    """
    # Pull metadata from MNE when present; ignore failures for raw arrays.
    try:
        sfreq = eeg_data.info["sfreq"]
        if ch_names is None:
            ch_names = eeg_data.info["ch_names"]
        eeg_data = eeg_data.get_data()  # MNE object -> (n_chans, n_samples)
    except Exception:
        pass

    if sfreq is None:
        raise ValueError("sfreq is not provided and cannot be inferred from the segment.")

    if ch_names is None:
        ch_names = [f"ch{i + 1}" for i in range(eeg_data.shape[0])]

    # Build a time vector in seconds from the sample index.
    n_samples = eeg_data.shape[1]
    t = np.arange(n_samples) / sfreq  # (n_samples,) in seconds

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(eeg_data.shape[0]):
        color = None if colors is None else colors[i]
        if alphas is not None:
            ax.plot(t, eeg_data[i], label=ch_names[i], color=color, alpha=alphas[i])
        else:
            ax.plot(t, eeg_data[i], label=ch_names[i], color=color)

    ax.set_xlabel("Time (s)")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel("Amplitude" if ylabel is None else ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def rms(eeg_data):
    """
    Root-mean-square amplitude along the time axis, one value per channel.

    Accepts an MNE object or a numpy array so callers can pass either a
    Raw/Epochs crop or a channel-selected array.

    Parameters
    ----------
    eeg_data : mne.io.BaseRaw, mne.Epochs, or ndarray
        Data with a trailing time axis. 1-D traces are treated as a single
        channel.

    Returns
    -------
    rms_vals : ndarray, shape (n_channels,)
        RMS of each channel over time.

    Raises
    ------
    ValueError
        If the array cannot be interpreted as ``(n_channels, n_samples)``.
    """
    # Accept MNE objects or already-extracted arrays.
    arr = np.asarray(eeg_data.get_data() if hasattr(eeg_data, "get_data") else eeg_data)

    # Promote a 1-D trace to a single-channel matrix.
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]  # (n_samples,) -> (1, n_samples)

    if arr.ndim != 2:
        raise ValueError("Data must be shape (n_channels, n_samples)")

    # Energy over time, then square-root: one RMS per channel.
    return np.sqrt(np.mean(arr**2, axis=1))  # (n_channels, n_samples) -> (n_channels,)


def select_channels(data, chan="all"): 
    """
    Return a channel subset as an array plus the integer channel indices.

    Parameters
    ----------
    data : mne.io.BaseRaw, mne.Epochs, or ndarray
        Continuous data with channels along axis 0 after ``get_data()``.
    chan : {"all", int, sequence of int}, optional
        Channels to keep. ``"all"`` keeps every channel. Default is
        ``"all"``.

    Returns
    -------
    arr : ndarray, shape (n_selected, n_samples)
        Selected traces.
    chans : list of int
        Integer indices of the selected channels in the original array.
    """
    # Extract (n_channels, n_samples) from MNE or pass arrays through.
    arr = np.asarray(data.get_data() if hasattr(data, "get_data") else data)

    if chan == "all":
        chans = list(range(arr.shape[0]))
    else:
        chans = [int(c) for c in np.atleast_1d(chan)]

    return arr[chans], chans  # (n_channels, n_samples) -> (n_selected, n_samples)


def zscore(signal, mu=None, sd=None):
    """
    Z-score along the last axis (time), broadcasting ``mu`` and ``sd``.

    If ``mu`` or ``sd`` is omitted, that statistic is computed from
    ``signal`` itself. A zero standard deviation is replaced by ``1`` so
    the division is defined.

    Parameters
    ----------
    signal : array-like
        Data to standardize. Typical shape ``(n_channels, n_samples)``.
    mu : array-like or None, optional
        Mean used for centering. Broadcast onto ``signal``. ``None``
        computes the mean over the last axis.
    sd : array-like or None, optional
        Standard deviation used for scaling. Broadcast onto ``signal``.
        ``None`` computes the std over the last axis.

    Returns
    -------
    z : ndarray
        Z-scored array, same shape as ``signal``.
    """
    arr = np.asarray(signal, dtype=float)

    # Mean over time, keeping a trailing singleton for broadcasting.
    if mu is None:
        mu = np.mean(arr, axis=-1, keepdims=True)  # (..., n_samples) -> (..., 1)
    else:
        mu = np.asarray(mu, dtype=float)
        if mu.ndim < arr.ndim:
            # Pad trailing singleton axes until mu matches arr.ndim.
            mu = np.reshape(mu, mu.shape + (1,) * (arr.ndim - mu.ndim))

    # Std over time, same broadcasting rules as mu.
    if sd is None:
        sd = np.std(arr, axis=-1, keepdims=True)  # (..., n_samples) -> (..., 1)
    else:
        sd = np.asarray(sd, dtype=float)
        if sd.ndim < arr.ndim:
            sd = np.reshape(sd, sd.shape + (1,) * (arr.ndim - sd.ndim))

    # Avoid divide-by-zero on flat channels.
    sd = np.where(sd == 0, 1.0, sd)

    return (arr - mu) / sd  # same shape as arr
