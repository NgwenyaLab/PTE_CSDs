"""
Subject container for TBI-PTE analyses.

``EEGSubject`` loads EDF recordings and Excel markers, stores preprocessed
data, and exposes spike detection, manual review, and seizure-RMS methods.
Spike lists live on the subject (``detected_spikes``, ``included_spikes``);
detection and review helpers are in ``spike_review.SpikeReview``.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .spike_review import SpikeReview
from .utils import (
    perform_mne_preprocessing,
    calculate_tfr,
    select_data_segment,
    select_channels,
    zscore,
    rms,
)


class EEGSubject:
    """
    One experimental subject: recordings, markers, preprocessing, and spikes.

    Attributes
    ----------
    subject_id : str
        Animal identifier (used to locate EDF/XLSX files).
    group : str
        Experimental-group label (e.g. treatment assignment).
    directory : pathlib.Path
        Folder containing this subject's EDF and XLSX files.
    coi : int, sequence of int, or {"all"}
        Channel(s) of interest.
    detected_spikes : dict
        ``{date: [(index, voltage, z), ...]}`` from automatic detection.
    included_spikes : dict
        ``{date: [(index, voltage, z), ...]}`` after manual review or load.
    raw : dict
        ``{date: mne.io.Raw}`` after :meth:`load_eeg`.
    markers : dict
        ``{date: {label: [(start_s, end_s), ...]}}`` after :meth:`load_markers`.
    preprocessed : dict
        ``{date: Raw or Epochs}`` after :meth:`preprocess_data`.
    """

    def __init__(self, subject_id, group, data_dir, coi="all"):
        """
        Store identity, group, channel of interest, and on-disk data path.

        Parameters
        ----------
        subject_id : str
            Animal identifier matching the exported EDF/XLSX filenames.
        group : str
            Experimental-group label stored on the instance.
        data_dir : str or pathlib.Path
            Directory that holds this subject's EDF and marker files.
        coi : int, sequence of int, or {"all"}, optional
            Channel(s) of interest.
            Default is ``"all"``.
        """
        # Identity and experimental assignment.
        self.subject_id = subject_id
        self.group = group

        # On-disk location of this animal's EDF and marker files.
        self.directory = Path(data_dir)

        # Spike containers and channel of interest (spike-counting workflow).
        self.detected_spikes = {}
        self.included_spikes = {}
        self.coi = coi

    def load_eeg(self, dates=None, filename_flag=None, rescale_eeg=None):
        """
        Find and load EDF files whose filename date tag is in ``dates``.

        The date tag is the full basename when ``filename_flag`` is ``None``,
        or ``file.name[start:stop]`` when ``filename_flag`` is ``(start, stop)``.
        If ``rescale_eeg`` is set, the loaded traces are multiplied by that
        factor (used to correct exported channel gain).

        Parameters
        ----------
        dates : sequence of str
            Date tags to load (e.g. ``["05_30_2024_part2"]``).
        filename_flag : tuple of int or None, optional
            ``(start, stop)`` unpacked as ``file.name[start:stop]``.
            ``None`` uses the entire filename as the date key.
        rescale_eeg : float or None, optional
            Multiplier applied to ``raw._data`` after loading. ``None``
            leaves amplitudes unchanged.

        Returns
        -------
        None
            Populates ``self.raw`` as ``{date: mne.io.Raw}``.

        Raises
        ------
        FileNotFoundError
            If no matching EDF is found.
        """
        self.raw = {}
        for file in self.directory.glob(f"{self.subject_id}*.edf"):
            # Unpack (start, stop) into a filename substring, or keep the full name.
            if filename_flag is None:
                date = file.name
            else:
                start, stop = filename_flag
                date = file.name[start:stop]
            if date in dates:
                print(f"Loading {date}")
                raw = mne.io.read_raw_edf(str(file), preload=True)
                if rescale_eeg is not None:
                    raw._data = rescale_eeg * raw._data  # (n_chans, n_times), scaled in place
                self.raw[date] = raw

        if not self.raw:
            raise FileNotFoundError(f"No EDF files found for subject {self.subject_id}")

    def load_markers(self, dates=None, filename_flag=None):
        """
        Load Excel markers and convert clock times to seconds from recording start.

        The first spreadsheet row is treated as a header leftover and
        skipped. Labels are lower-cased. Each event becomes
        ``(start_offset_s, end_offset_s)`` relative to
        ``raw.info["meas_date"]``. Filename matching uses the same
        ``filename_flag`` convention as :meth:`load_eeg`.

        Parameters
        ----------
        dates : sequence of str
            Date tags to load. Each date must already exist in ``self.raw``.
        filename_flag : tuple of int or None, optional
            ``(start, stop)`` unpacked as ``file.name[start:stop]``.
            ``None`` uses the entire filename as the date key.

        Returns
        -------
        None
            Populates ``self.markers`` as
            ``{date: {label: [(start_s, end_s), ...]}}``.

        Raises
        ------
        ValueError
            If EEG for that date has not been loaded, so the time origin
            cannot be determined.
        """
        self.markers = {}
        for file in self.directory.glob(f"{self.subject_id}*.xlsx"):
            marker_dict = {}
            # Unpack (start, stop) into a filename substring, or keep the full name.
            if filename_flag is None:
                date = file.name
            else:
                start, stop = filename_flag
                date = file.name[start:stop]
            if date in dates:
                print(f"Loading markers for subject {self.subject_id} and date {date}")
                df_markers = pd.read_excel(file)
                df_valid = df_markers.iloc[1:]  # skip leftover header row

                if date not in self.raw:
                    raise ValueError(
                        f"No EEG data found for date {date}, Cannot determine start time for markers"
                    )

                # Recording start, with timezone stripped for datetime arithmetic.
                start_time = self.raw[date].info["meas_date"].replace(tzinfo=None)

                for _, row in df_valid.iterrows():
                    start_dt = pd.to_datetime(row["Start Time"])
                    end_dt = pd.to_datetime(row["End Time"])
                    start_offset = (start_dt - start_time).total_seconds()
                    end_offset = (end_dt - start_time).total_seconds()
                    label = str(row["Label"]).lower()

                    if label not in marker_dict:
                        marker_dict[label] = []
                    marker_dict[label].append((start_offset, end_offset))
                self.markers[date] = marker_dict

    def preprocess_data(
        self,
        down_sample_rate=None,
        high_pass_filter=None,
        low_pass_filter=None,
        notch_freq=None,
        epoch_duration=None,
    ):
        """
        Filter, optionally resample, and optionally epoch every loaded recording.

        Parameters
        ----------
        down_sample_rate : float or None, optional
            Target sampling rate in Hz. ``None`` keeps the original rate.
        high_pass_filter : float or None, optional
            High-pass cutoff in Hz.
        low_pass_filter : float or None, optional
            Low-pass cutoff in Hz.
        notch_freq : float, array-like, or None, optional
            Notch frequency or frequencies in Hz.
        epoch_duration : float or None, optional
            Fixed epoch length in seconds. ``None`` leaves data continuous.

        Returns
        -------
        None
            Populates ``self.preprocessed`` as ``{date: Raw or Epochs}``.

        Raises
        ------
        RuntimeError
            If :meth:`load_eeg` has not been called.
        """
        if not hasattr(self, "raw") or not self.raw:
            raise RuntimeError("No raw EEG data found. Call load_eeg() first.")

        self.preprocessed = {}
        for date in self.raw:
            self.preprocessed[date] = perform_mne_preprocessing(
                self.raw[date],
                high_pass_filter=high_pass_filter,
                low_pass_filter=low_pass_filter,
                notch_freq=notch_freq,
                down_sample_rate=down_sample_rate,
                epoch_duration=epoch_duration,
            )

    def get_tfr(
        self,
        down_sample_rate=None,
        freqs=(1, 100, 1),
        method="multitaper",
        n_cycles=7,
        output="power",
        average_epochs=False,
    ):
        """
        Compute a time-frequency representation for every preprocessed recording.

        Parameters
        ----------
        down_sample_rate : float or None, optional
            Extra resample applied before the TFR. Default is ``None``.
        freqs : sequence of float, optional
            ``[fmin, fmax, fstep]`` in Hz. Default is ``(1, 100, 1)``.
        method : str, optional
            MNE TFR method. Default is ``"multitaper"``.
        n_cycles : float or array-like, optional
            Cycles per frequency. Default is ``7``.
        output : str, optional
            ``"power"`` or ``"phase"``. Default is ``"power"``.
        average_epochs : bool, optional
            If ``True``, average TFR across epochs. Default is ``False``.

        Returns
        -------
        None
            Populates ``self.tfr`` as ``{date: TFR}``.
        """
        self.tfr = {}
        for date in self.preprocessed:
            self.tfr[date] = calculate_tfr(
                self.preprocessed[date],
                freq_params=freqs,
                down_sample_rate=down_sample_rate,
                method=method,
                cycles=n_cycles,
                output=output,
                average_epochs=average_epochs,
            )

    def get_baseline(self):
        """
        Return the first EEG segment whose marker label contains ``"baseline"``.

        Returns
        -------
        baseline : mne.io.BaseRaw
            Cropped preprocessed recording for that marker window.

        Raises
        ------
        ValueError
            If no baseline marker is found on any date.
        """
        for d in getattr(self, "markers", {}):
            for label, ranges in self.markers.get(d, {}).items():
                if "baseline" in str(label).lower() and ranges:
                    t0, t1 = ranges[0]
                    return select_data_segment(self.preprocessed[d], t0, t1)
        raise ValueError(f"No baseline marker found for subject {self.subject_id}")

    def get_seizure_windows(self, date):
        """
        Racine-labeled marker windows on one date, as time in seconds.

        Parameters
        ----------
        date : str
            Date tag present in ``self.markers``.

        Returns
        -------
        windows : list of tuple
            ``(racine_label, t0_sec, t1_sec)`` for every Racine event.
        """
        windows = []
        for racine_level, ranges in self.markers.get(date, {}).items():
            if "racine" in str(racine_level).lower():
                for t0, t1 in ranges:
                    windows.append((str(racine_level), float(t0), float(t1)))
        return windows

    def seizure_rms(self, summary="longest", zscored=True):
        """
        RMS of Racine-labeled seizures across all marker days.

        Each seizure crop is optionally z-scored to the baseline, RMS is
        computed per channel, then channels are averaged to one value
        per seizure. ``summary`` reduces that list.

        Parameters
        ----------
        summary : {"longest", "mean", "median", "max", "all"}, optional
            Reduction over seizures. ``"longest"`` (default) returns RMS
            of the longest-duration event. ``"all"`` returns the array.
        zscored : bool, optional
            If ``True``, z-score each seizure by baseline before RMS.
            Default is ``True``.

        Returns
        -------
        result : float or ndarray
            Scalar summary, or an array of per-seizure RMS when
            ``summary="all"``. ``nan`` / empty list if no seizures exist.
        """
        print(f"Calculating RMS for {self.subject_id}")
        baseline = self.get_baseline() if zscored else None
        durations = []
        rms_vals = []

        for date in self.markers:
            if date not in self.preprocessed:
                continue
            eeg = self.preprocessed[date]
            tmax = float(eeg.times[-1])
            for racine_level, t0, t1 in self.get_seizure_windows(date):
                t0 = max(0.0, t0)
                t1 = min(tmax, t1)
                duration = t1 - t0
                if duration <= 0:
                    continue

                # Crop the seizure, then keep only the channel(s) of interest.
                seg = select_data_segment(eeg, t0, t1)
                arr, _ = select_channels(seg, chan=self.coi)  # (n_chans, n_times) -> (n_coi, n_times)

                if zscored:
                    base, _ = select_channels(baseline, chan=self.coi)
                    mu = base.mean(axis=1, keepdims=True)  # (n_coi, n_base) -> (n_coi, 1)
                    sd = base.std(axis=1, keepdims=True)
                    arr = zscore(arr, mu=mu, sd=sd)  # (n_coi, n_times)

                # Mean RMS across selected channels -> one scalar per seizure.
                rms_val = float(np.mean(rms(arr)))  # (n_coi, n_times) -> (n_coi,) -> scalar
                print(f"{date}: {racine_level}, duration = {duration:.2f} s, RMS = {rms_val:.4g}")
                durations.append(duration)
                rms_vals.append(rms_val)

        if not rms_vals:
            print("No seizures found")
            return [] if summary == "all" else float("nan")

        rms_vals = np.asarray(rms_vals, dtype=float)  # list -> (n_seizures,)
        durations = np.asarray(durations, dtype=float)
        n = len(rms_vals)
        unit = "zscored V" if zscored else "V"

        if summary == "all":
            result = rms_vals
            print(f"All {n} seizure RMS values for {self.subject_id} ({unit}): {result}\n")
        elif summary == "longest":
            i = int(np.argmax(durations))
            result = float(rms_vals[i])
            print(
                f"Longest-seizure RMS for {self.subject_id}: {result} {unit}"
                f"(duration {durations[i]:.2f} s)\n"
            )
        elif summary == "mean":
            result = float(np.mean(rms_vals))
            print(f"Mean seizure RMS for {self.subject_id} across {n} seizures: {result} {unit}\n")
        elif summary == "median":
            result = float(np.median(rms_vals))
            print(f"Median seizure RMS for {self.subject_id} across {n} seizures: {result} {unit}\n")
        elif summary == "max":
            result = float(np.max(rms_vals))
            print(f"Max seizure RMS for {self.subject_id} across {n} seizures: {result} {unit}\n")
        else:
            raise ValueError("summary must be 'longest', 'mean', 'median', 'max', or 'all'")
        return result

    def plot_seizures(self, date=None):
        """
        Plot each Racine-labeled seizure in MNE's Qt browser, one after another.

        Close a window to open the next seizure. ``date=None`` plots every
        marker day.

        Parameters
        ----------
        date : str or None, optional
            Single date tag, or ``None`` for all dates in ``self.markers``.

        Returns
        -------
        None
            Opens blocking MNE plot windows. Nothing is returned.
        """
        print(f"Plotting seizures for {self.subject_id}")
        dates = [date] if date is not None else list(self.markers)
        seizures = []
        for d in dates:
            if d not in self.preprocessed:
                continue
            eeg = self.preprocessed[d]
            tmax = float(eeg.times[-1])
            for racine_level, t0, t1 in self.get_seizure_windows(d):
                t0 = max(0.0, t0)
                t1 = min(tmax, t1)
                if t1 <= t0:
                    continue
                seizures.append((d, racine_level, t0, t1))

        if not seizures:
            print("No seizures found")
            return

        n = len(seizures)
        for i, (d, racine_level, t0, t1) in enumerate(seizures, start=1):
            duration = t1 - t0
            print(f"{d}: {racine_level}, duration = {duration:.2f} s  [{i}/{n}]")
            seg = select_data_segment(self.preprocessed[d], t0, t1)
            title = (
                f"{self.subject_id}  {d}  {racine_level}  "
                f"{duration:.1f} s  [{i}/{n}]"
            )

            # Scale the browser to the interquartile range of finite samples.
            data = np.asarray(seg.get_data(), dtype=float)
            finite = data[np.isfinite(data)]
            if finite.size:
                scale = float(np.subtract(*np.percentile(finite, [75, 25])))
                if scale <= 0:
                    scale = float(np.std(finite))
            else:
                scale = 0.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 20e-6
            scalings = {ch_type: scale for ch_type in set(seg.get_channel_types())}
            seg.plot(
                duration=float(seg.times[-1] - seg.times[0]) or duration,
                n_channels=len(seg.ch_names),
                scalings=scalings,
                title=title,
                block=True,
                proj=False,
            )

    def detect_spikes(
        self,
        day,
        threshold=3,
        spike_buffer=0.05,
        plot=False,
        remove_seizures=None,
        time_range=None,
    ):
        """
        Detect spikes on ``self.coi`` and store them on ``self.detected_spikes``.

        Parameters
        ----------
        day : str
            Date tag present in ``self.preprocessed``.
        threshold : float, optional
            ``|z|`` peak height. Default is ``3``.
        spike_buffer : float, optional
            Minimum inter-peak distance in seconds. Default is ``0.05``.
        plot : bool, optional
            If ``True``, overlay detections on the (rectified) trace.
        remove_seizures : {None, False, True, sequence of tuple}, optional
            ``True`` drops peaks inside any Racine window. A list of
            ``(start, stop)`` sample tuples is used as custom windows.
            ``None`` / ``False`` keeps every detection.
        time_range : tuple of int or None, optional
            ``(start_idx, end_idx)`` sample window. ``None`` uses the
            full recording.

        Returns
        -------
        detected_spikes : list of tuple
            ``(index, voltage, z)`` stored at ``self.detected_spikes[day]``.
        """
        print(f"Detecting spikes for {self.subject_id} on {day}")
        eeg = self.preprocessed[day]
        baseline = self.get_baseline()

        if remove_seizures is True:
            sfreq = eeg.info["sfreq"]
            seizure_windows = [
                (int(round(t0 * sfreq)), int(round(t1 * sfreq)))
                for racine_level, t0, t1 in self.get_seizure_windows(day)
            ]
        elif remove_seizures:
            seizure_windows = remove_seizures
        else:
            seizure_windows = None

        detected_spikes = SpikeReview.detect(
            eeg,
            baseline,
            threshold=threshold,
            spike_buffer=spike_buffer,
            chan=self.coi,
            seizure_windows=seizure_windows,
            time_range=time_range,
        )
        self.detected_spikes[day] = detected_spikes
        print(f"Detected {len(detected_spikes)} spikes")
        if plot:
            SpikeReview.plot_detected(
                eeg, detected_spikes, self.coi, rectify=True, time_range=time_range
            )
        return detected_spikes

    def plot_detected_spikes(self, day, rectify=False, time_range=None):
        """
        Plot previously detected spikes for one recording day.

        Parameters
        ----------
        day : str
            Date key in ``self.detected_spikes`` and ``self.preprocessed``.
        rectify : bool, optional
            If ``True``, plot ``|EEG|``. Default is ``False``.
        time_range : tuple of int or None, optional
            Sample window to display. ``None`` plots the full recording.

        Returns
        -------
        None
        """
        print(f"Spikes for {self.subject_id} on {day}")
        eeg = self.preprocessed[day]
        SpikeReview.plot_detected(
            eeg,
            self.detected_spikes[day],
            self.coi,
            rectify=rectify,
            time_range=time_range,
        )

    def review_spikes(
        self,
        day,
        spikes=None,
        short_window=1,
        long_window=10,
        zscore=True,
        y=None,
    ):
        """
        Manually review detections and store the included subset.

        Parameters
        ----------
        day : str
            Date tag used to look up EEG and to store the result.
        spikes : sequence of tuple or None, optional
            Events to review. ``None`` uses ``self.detected_spikes[day]``.
        short_window, long_window : float, optional
            Half-widths of the two review panels, in seconds.
        zscore : bool, optional
            If ``True``, plot baseline z-scored traces. Default is ``True``.
        y : sequence of float or None, optional
            Shared y-limits. ``None`` autoscales.

        Returns
        -------
        included : list of tuple
            Stored at ``self.included_spikes[day]``.
        """
        print(f"Reviewing spikes for {self.subject_id} on {day}")
        eeg = self.preprocessed[day]
        if spikes is None:
            spikes = self.detected_spikes.get(day, [])
        self.included_spikes[day] = SpikeReview.review(
            eeg,
            self.get_baseline(),
            spikes,
            short_window=short_window,
            long_window=long_window,
            zscored=zscore,
            y=y,
        )
        return self.included_spikes[day]

    def review_included_spikes(
        self, day, short_window=1, long_window=10, zscore=True, y=None
    ):
        """
        Re-review the already-included spike list for one day.

        Parameters
        ----------
        day : str
            Date key in ``self.included_spikes``.
        short_window, long_window : float, optional
            Half-widths of the two review panels, in seconds.
        zscore : bool, optional
            If ``True``, plot baseline z-scored traces. Default is ``True``.
        y : sequence of float or None, optional
            Shared y-limits. ``None`` autoscales.

        Returns
        -------
        included : list of tuple
            Updated ``self.included_spikes[day]``.
        """
        return self.review_spikes(
            day,
            spikes=self.included_spikes.get(day, []),
            short_window=short_window,
            long_window=long_window,
            zscore=zscore,
            y=y,
        )

    def spike_summary(self, day=None, avg_func=np.mean, zscore=True):
        """
        Print detection vs inclusion counts and amplitude summaries.

        Parameters
        ----------
        day : str or None, optional
            Date key, ``"pool"``, or ``None`` for every date.
        avg_func : callable, optional
            Amplitude reduction (e.g. ``np.median``). Default is ``np.mean``.
        zscore : bool, optional
            Summarize ``|z|`` if ``True``, else ``|voltage|``.

        Returns
        -------
        None
        """
        print(f"Spike summary for {self.subject_id}")
        SpikeReview.summary(
            self.detected_spikes,
            self.included_spikes,
            day=day,
            avg_func=avg_func,
            zscore=zscore,
        )

    def save_spikes(self, date, suffix = ""):
        """
        Write included spikes for one day to ``{id}_{date}_spikes.csv``.

        Parameters
        ----------
        date : str
            Date key in ``self.included_spikes``.
        suffix : str, optional
            Suffix to add to the filename. Default is an empty string.
        Returns
        -------
        path : pathlib.Path
            CSV path under ``self.directory``.
        """
        print(f"Saving spikes for {self.subject_id} on {date}")
        path = self.directory / f"{self.subject_id}_{date}_spikes{suffix}.csv"
        return SpikeReview.save(path, self.included_spikes.get(date, []))

    def load_spikes(self, date, suffix = ""):
        """
        Read included spikes for one day from ``{id}_{date}_spikes.csv``.
 
        Parameters
        ----------
        date : str
            Date tag encoded in the CSV filename.
        suffix : str, optional
            Suffix to add to the filename. Default is an empty string.

        Returns
        -------
        spikes : list of tuple
            Stored at ``self.included_spikes[date]``.
        """
        print(f"Loading spikes for {self.subject_id} on {date}")
        path = self.directory / f"{self.subject_id}_{date}_spikes{suffix}.csv"
        self.included_spikes[date] = SpikeReview.load(path)
        return self.included_spikes[date]

    def plot_spike_amplitude_hist(self, date, zscore=True):
        """
        Histogram of included-spike amplitudes for one day or the pool.

        Parameters
        ----------
        date : str
            Date key, or ``"pool"`` to concatenate every day.
        zscore : bool, optional
            Histogram ``|z|`` if ``True``, else ``|voltage|``.

        Returns
        -------
        None
        """
        print(f"Spike amplitude histogram for {self.subject_id} on {date}")
        SpikeReview.plot_amplitude_hist(self.included_spikes, date, zscore=zscore)
