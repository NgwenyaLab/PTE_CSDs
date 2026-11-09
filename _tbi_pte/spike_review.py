"""
Stateless spike detection, review, and I/O helpers for TBI-PTE analyses.

Spike lists live on ``EEGSubject`` (``detected_spikes``, ``included_spikes``).
Each detected event is a tuple ``(sample_index, voltage, signed_z)``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from .utils import select_channels, zscore


class SpikeReview:
    """
    Stateless helpers for spike detection, plotting, review, and I/O.

    Spike lists are stored on ``EEGSubject`` (``detected_spikes`` /
    ``included_spikes``), not on this class. Each detected event is a
    tuple ``(sample_index, voltage, signed_z)``.
    """

    @staticmethod
    def detect(
        eeg,
        baseline,
        threshold=3,
        spike_buffer=0.05,
        chan="all",
        seizure_windows=None,
        time_range=None,
    ):
        """
        Find peaks on the absolute z-scored EEG relative to a baseline.

        Peak indices are always expressed in the original recording's
        sample clock, even when ``time_range`` restricts the search.

        Parameters
        ----------
        eeg : mne.io.BaseRaw
            Recording in which to detect spikes.
        baseline : mne.io.BaseRaw
            Baseline segment used to estimate per-channel mean and SD.
        threshold : float, optional
            Height threshold on ``|z|``. Default is ``3``.
        spike_buffer : float, optional
            Minimum inter-peak distance in seconds. Default is ``0.05``.
        chan : {"all", int, sequence of int}, optional
            Channel(s) of interest. Default is ``"all"``.
        seizure_windows : sequence of (int, int) or None, optional
            Sample-index windows ``(start, stop)`` in which detections
            are discarded. ``None`` keeps every peak.
        time_range : tuple of int or None, optional
            ``(start_idx, end_idx)`` sample window to search. ``None``
            uses the full recording.

        Returns
        -------
        detected_spikes : list of tuple
            ``(index, voltage, signed_z)`` for each accepted peak.
        """
        sfreq = eeg.info["sfreq"]
        buffer_samp = max(1, int(round(spike_buffer * sfreq)))

        # Restrict both the search recording and the baseline to the same channels.
        volt, chans = select_channels(eeg, chan=chan)  # (n_chans, n_times)
        base, _ = select_channels(baseline, chan=chan)  # (n_chans, n_base)

        # Baseline moments used to z-score the full recording.
        mu = base.mean(axis=1, keepdims=True)  # (n_chans, n_base) -> (n_chans, 1)
        sd = base.std(axis=1, keepdims=True)  # (n_chans, n_base) -> (n_chans, 1)
        z = zscore(volt, mu=mu, sd=sd)  # (n_chans, n_times)
        z_rectified = np.abs(z)  # (n_chans, n_times)

        # Optionally crop to a sample window; stored indices stay absolute.
        start_idx = 0
        if time_range is not None:
            start_idx, end_idx = int(time_range[0]), int(time_range[1])
            z = z[:, start_idx:end_idx]  # (n_chans, n_times) -> (n_chans, n_window)
            z_rectified = z_rectified[:, start_idx:end_idx]
            volt = volt[:, start_idx:end_idx]

        detected_spikes = []
        n_before = 0
        for r in range(len(chans)):
            # Peaks of |z| on this channel, separated by spike_buffer.
            spks, _ = find_peaks(z_rectified[r], height=threshold, distance=buffer_samp)
            n_before += len(spks)
            for p in np.atleast_1d(np.asarray(spks, dtype=int)):
                p = int(p)
                p_abs = p + start_idx # adding the start_idx to get the absolute index (regardless of cropped time range)
                # Drop peaks that fall inside a seizure window.
                if seizure_windows and any(
                    int(w[0]) <= p_abs <= int(w[1]) for w in seizure_windows
                ):
                    continue
                detected_spikes.append((p_abs, float(volt[r, p]), float(z[r, p])))

        if seizure_windows:
            print(f"Removed {n_before - len(detected_spikes)} spikes from seizure/s")
        return detected_spikes

    @staticmethod
    def plot_detected(eeg, spikes, chans, rectify=False, time_range=None):
        """
        Plot EEG traces with detected spike times marked as red crosses.

        Parameters
        ----------
        eeg : mne.io.BaseRaw
            Recording to display.
        spikes : sequence of tuple
            ``(index, voltage, z)`` detections. Only ``index`` is used.
        chans : {"all", int, sequence of int}
            Channel(s) to plot (same convention as ``select_channels``).
        rectify : bool, optional
            If ``True``, plot ``|EEG|``. Default is ``False``.
        time_range : tuple of int or None, optional
            ``(start_idx, end_idx)`` sample window. ``None`` plots the
            full recording.

        Returns
        -------
        None
            Displays the figure. Nothing is returned.
        """
        eeg_data, chans = select_channels(eeg, chan=chans)  # (n_chans, n_times)

        start_idx = 0
        if time_range is not None:
            start_idx, end_idx = int(time_range[0]), int(time_range[1])
            eeg_data = eeg_data[:, start_idx:end_idx]  # (n_chans, n_times) -> (n_chans, n_window)

        sfreq = eeg.info["sfreq"]
        if rectify:
            eeg_data = np.abs(eeg_data)

        t = (np.arange(eeg_data.shape[1]) + start_idx) / sfreq  # sample index -> seconds
        n = eeg_data.shape[1]

        # Convert absolute spike indices to the (possibly cropped) plot axis.
        indices = [i - start_idx for i, v, z in spikes if start_idx <= i < start_idx + n]

        fig, axes = plt.subplots(len(chans), 1, figsize=(12, 3 * len(chans)), sharex=True)
        if len(chans) == 1:
            axes = [axes]
        for ch_i, ax in enumerate(axes):
            ax.plot(t, eeg_data[ch_i], color="black", linewidth=0.8)
            if indices:
                ax.plot(t[indices], eeg_data[ch_i, indices], "x", color="red", markersize=8)
            ax.set_ylabel(f"Ch {chans[ch_i] + 1}")
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def review(
        eeg,
        baseline,
        spikes,
        short_window=1,
        long_window=10,
        zscored=True,
        y=None,
    ):
        """
        Interactively include or exclude detected spikes from a two-window view.

        All passed-in spikes start as included. Keys:

        * ``y`` include, ``n`` exclude, ``q`` quit
        * arrows: left/right ±1, up/down ±5
        * ``1``–``9`` skip ahead ``10 * digit`` spikes

        Parameters
        ----------
        eeg : mne.io.BaseRaw
            Recording that contains the spikes.
        baseline : mne.io.BaseRaw
            Baseline used when ``zscored=True``.
        spikes : sequence of tuple
            ``(index, voltage, z)`` events to review.
        short_window : float, optional
            Half-width of the left panel in seconds. Default is ``1``.
        long_window : float, optional
            Half-width of the right panel in seconds. Default is ``10``.
        zscored : bool, optional
            If ``True``, plot baseline z-scored traces. Default is ``True``.
        y : sequence of float or None, optional
            ``(ymin, ymax)`` for every panel. ``None`` autoscales.

        Returns
        -------
        included : list of tuple
            Subset of ``spikes`` marked included at quit.
        """
        included = list(spikes)
        indices = [i for i, v, z in spikes]
        if not indices:
            print("No spikes to review.")
            return []

        sfreq = eeg.info["sfreq"]
        volt, plot_chans = select_channels(eeg, chan="all")  # (n_chans, n_times)
        n_times = volt.shape[1]

        # Choose voltage vs z-scored traces for the review display.
        if zscored:
            base, _ = select_channels(baseline, chan="all")
            mu = base.mean(axis=1, keepdims=True)  # (n_chans, n_base) -> (n_chans, 1)
            sd = base.std(axis=1, keepdims=True)
            trace = zscore(volt, mu=mu, sd=sd)  # (n_chans, n_times)
            ylab = "(z)"
        else:
            trace = volt
            ylab = "(V)"

        short_samp = max(1, int(round(short_window * sfreq)))
        long_samp = max(1, int(round(long_window * sfreq)))
        ylim = None if y is None else (float(y[0]), float(y[1]))

        keep = [True] * len(included)
        state = {"i": 0}

        fig, axes = plt.subplots(
            len(plot_chans), 2, figsize=(14, 2.8 * len(plot_chans)), sharey="row"
        )
        axes = np.atleast_2d(axes)
        try:
            fig.canvas.manager.set_window_title(
                "Spike review  (← →  ↑↓±5  1-9 skip 10×  y / n / q)"
            )
        except Exception:
            pass

        def draw():
            # Redraw both time scales for the current spike index.
            p = indices[state["i"]]
            d = keep[state["i"]]
            mark_color = {True: "green", False: "red"}[d]
            status = {True: "included", False: "excluded"}[d]
            for row, ch in enumerate(plot_chans):
                for col, half in enumerate((short_samp, long_samp)):
                    ax = axes[row, col]
                    start = max(0, p - half)
                    stop = min(n_times, p + half + 1)
                    t = np.arange(start, stop) / sfreq
                    ax.clear()
                    ax.plot(t, trace[row, start:stop], color="black", linewidth=0.8)
                    if start <= p < stop:
                        ax.plot(p / sfreq, trace[row, p], "x", color=mark_color, markersize=10)
                    if ylim is not None:
                        ax.set_ylim(ylim)
                    ax.grid(True, alpha=0.3)
                axes[row, 0].set_ylabel(f"Ch {ch + 1} {ylab}")
            axes[0, 0].set_title(f"±{short_window} s")
            axes[0, 1].set_title(f"±{long_window} s")
            axes[-1, 0].set_xlabel("Time (s)")
            axes[-1, 1].set_xlabel("Time (s)")
            n_in = sum(keep)
            fig.suptitle(
                f"Spike {state['i'] + 1} / {len(indices)}  [{status}]  "
                f"|  {n_in} included spikes  |  ← →  ↑↓ ±5  1-9 skip 10×  y include  n exclude  q quit"
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.canvas.draw_idle()

        def on_key(event):
            # Update inclusion state and/or the current spike index.
            key = event.key
            n = len(indices)
            i = state["i"]
            if key == "q":
                plt.close(fig)
                return
            if key == "left":
                state["i"] = max(0, i - 1)
            elif key == "right":
                state["i"] = min(n - 1, i + 1)
            elif key == "up":
                state["i"] = max(0, i - 5)
            elif key == "down":
                state["i"] = min(n - 1, i + 5)
            elif key in "123456789":
                state["i"] = min(n - 1, i + 10 * int(key))
            elif key == "y":
                keep[state["i"]] = True
                if state["i"] < n - 1:
                    state["i"] += 1
            elif key == "n":
                keep[state["i"]] = False
                if state["i"] < n - 1:
                    state["i"] += 1
            else:
                return
            draw()

        fig.canvas.mpl_connect("key_press_event", on_key)
        draw()
        plt.show(block=True)

        included = [value for value, k in zip(spikes, keep) if k]
        print(f"Kept {len(included)} / {len(spikes)} included spikes")
        return included

    @staticmethod
    def summary(detected_spikes, included_spikes, day=None, avg_func=np.mean, zscore=True):
        """
        Print detection vs inclusion counts and amplitude summaries.

        Parameters
        ----------
        detected_spikes : dict
            ``{date: [(index, voltage, z), ...]}``.
        included_spikes : dict
            ``{date: [(index, voltage, z), ...]}``.
        day : str or None, optional
            Date key to summarize, ``"pool"`` to pool every date, or
            ``None`` to print each date separately.
        avg_func : callable, optional
            Reduction applied to absolute amplitudes (e.g. ``np.mean``,
            ``np.median``). Default is ``np.mean``.
        zscore : bool, optional
            If ``True``, summarize signed-z (tuple index 2); otherwise
            raw voltage (tuple index 1). Default is ``True``.

        Returns
        -------
        None
            Results are printed only.
        """
        if day == "pool":
            # Collapse every recording day into one amplitude distribution.
            n_orig = sum(len(s) for s in detected_spikes.values())
            included_all = []
            for day_included in included_spikes.values():
                included_all.extend(day_included)
            n_inc = len(included_all)
            if n_inc:
                amps = np.abs([value[2] if zscore else value[1] for value in included_all])
                avg = avg_func(amps)
                mx = np.max(amps)
            else:
                avg = float("nan")
                mx = float("nan")
            unit = "zscored V" if zscore else "V"
            print(
                f"pool: original spikes = {n_orig}, included spikes = {n_inc}, "
                f"avg {unit} = {avg}, max {unit} = {mx}"
            )
            return

        days = [day] if day is not None else list(detected_spikes.keys())
        for d in days:
            n_orig = len(detected_spikes.get(d, []))
            day_included = included_spikes.get(d, [])
            n_inc = len(day_included)
            if n_inc:
                amps = np.abs([value[2] if zscore else value[1] for value in day_included])
                avg = avg_func(amps)
                mx = np.max(amps)
            else:
                avg = float("nan")
                mx = float("nan")
            unit = "z" if zscore else "V"
            print(
                f"{d}: original spikes = {n_orig}, included spikes = {n_inc}, "
                f"avg {unit} = {avg}, max {unit} = {mx}"
            )

    @staticmethod
    def save(path, included_spikes):
        """
        Write included spikes to a CSV with columns index, voltage, and z.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination CSV. Parent directories are created as needed.
        included_spikes : sequence of tuple
            ``(index, voltage, z)`` rows to write.

        Returns
        -------
        path : pathlib.Path
            Path that was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "spike_index": [int(value[0]) for value in included_spikes],
                "spike_amp": [float(value[1]) for value in included_spikes],
                "z_amp": [float(value[2]) for value in included_spikes],
            }
        )
        df.to_csv(path, index=False)
        print(f"Saved {len(included_spikes)} spikes to {path}")
        return path

    @staticmethod
    def load(path):
        """
        Read a spike CSV written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            CSV with ``spike_index``, ``spike_amp``, and optionally
            ``z_amp``.

        Returns
        -------
        spikes : list of tuple
            ``(sample_index, voltage, z)``. Missing ``z_amp`` becomes
            ``nan``.
        """
        path = Path(path)
        df = pd.read_csv(path)
        has_z = "z_amp" in df.columns
        spikes = []
        for _, row in df.iterrows():
            sample = int(row["spike_index"])
            amp = float(row["spike_amp"])
            z_val = float(row["z_amp"]) if has_z else float("nan")
            spikes.append((sample, amp, z_val))
        print(f"Loaded {len(spikes)} spikes from {path}")
        return spikes

    @staticmethod
    def plot_amplitude_hist(included_spikes, date, zscore=True):
        """
        Histogram of absolute included-spike amplitudes for one day or the pool.

        Parameters
        ----------
        included_spikes : dict
            ``{date: [(index, voltage, z), ...]}``.
        date : str
            Date key, or ``"pool"`` to concatenate every day.
        zscore : bool, optional
            If ``True``, histogram ``|z|``; otherwise ``|voltage|``.
            Default is ``True``.

        Returns
        -------
        None
            Displays the figure, or prints a message if no spikes exist.
        """
        if date == "pool":
            inc = []
            for day_included_spikes in included_spikes.values():
                inc.extend(day_included_spikes)
        else:
            inc = included_spikes.get(date, [])

        if not inc:
            print(f"No included spikes for {date}")
            return

        amps = np.abs([value[2] if zscore else value[1] for value in inc])  # list -> (n_spikes,)
        unit = "z" if zscore else "V"
        mean_amp = np.mean(amps)
        median_amp = np.median(amps)

        plt.figure()
        plt.hist(amps, bins="auto", color="black", edgecolor="white")
        plt.axvline(mean_amp, color="red", linestyle="--", label=f"mean = {mean_amp:.3g}")
        plt.axvline(median_amp, color="blue", linestyle=":", label=f"median = {median_amp:.3g}")
        plt.xlabel(f"Spike amplitude ({unit})")
        plt.ylabel("Count")
        plt.title(f"{date}: {len(amps)} spikes")
        plt.legend()
        plt.tight_layout()
        plt.show()
