"""
PTZ-day behavioral plots for TBI-PTE analyses.
"""

import numpy as np
import matplotlib.pyplot as plt


def _get_seizure_durations(subject_list, racines, seizure_colors):
    """
    Collect Racine seizure durations (end - start, seconds) per subject.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects whose ``markers`` dictionaries will be scanned.
    racines : sequence of str or {"all"}
        Seizure-type labels to include. ``"all"`` uses every key in
        ``seizure_colors``.
    seizure_colors : dict
        Label-to-color map. Keys define the ``"all"`` set of seizure types.

    Returns
    -------
    durations : list of list of float
        One inner list of durations per subject, in ``subject_list`` order.
    """
    r = list(seizure_colors.keys()) if racines == "all" else racines

    # Flatten (date, label, event) into a duration list for each subject.
    return [
        [
            seizure[1] - seizure[0]
            for date in subj.markers
            for m in subj.markers[date]
            if m in r
            for seizure in subj.markers[date][m]
        ]
        for subj in subject_list
    ]


def _get_marker_durations(subject_list, markers):
    """
    Collect durations (end - start, seconds) for named markers, per subject.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects whose ``markers`` dictionaries will be scanned.
    markers : sequence of str
        Marker labels to include (e.g. ``["racine4", "racine5"]``).

    Returns
    -------
    durations_per_subject : list of list of float
        One inner list of durations per subject, in ``subject_list`` order.
    """
    markers = list(markers)
    durations_per_subject = []

    # Walk every date and every matching label on each subject.
    for subj in subject_list:
        this_subj = []
        for date in subj.markers:
            for m in subj.markers[date]:
                if m in markers:
                    this_subj.extend(
                        (end - start) for (start, end) in subj.markers[date][m]
                    )
        durations_per_subject.append(this_subj)

    return durations_per_subject


def compare_seizure_counts(subject_list, seizure_colors, plot=True):
    """
    Count Racine-stage seizures per subject and optionally draw a stacked bar plot.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects with a ``markers`` dict of the form
        ``{date: {label: [(start_s, end_s), ...]}}``.
    seizure_colors : dict
        Mapping from seizure-type label (e.g. ``"racine4"``) to a
        matplotlib color. Keys define which labels are counted and stacked.
    plot : bool, optional
        If ``True``, show a stacked bar chart. Default is ``True``.

    Returns
    -------
    out : dict
        ``{subject_id: {seizure_type: count}}``. Also printed.
    """
    seizure_types = list(seizure_colors.keys())
    subject_ids = [s.subject_id for s in subject_list]

    # Initialize a zero count for every Racine stage on every subject.
    out = {s.subject_id: {stype: 0 for stype in seizure_types} for s in subject_list}

    # Accumulate event counts from all recording dates.
    for subj in subject_list:
        for date in subj.markers:
            for stype in subj.markers[date]:
                if stype in seizure_types:
                    out[subj.subject_id][stype] += len(subj.markers[date][stype])

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        bottom = [0] * len(subject_ids)
        for stype in seizure_types:
            heights = [out[s][stype] for s in subject_ids]
            ax.bar(subject_ids, heights, bottom=bottom, label=stype, color=seizure_colors[stype])
            # Raise the stack baseline by this stage's counts.
            bottom = [bottom[i] + out[s][stype] for i, s in enumerate(subject_ids)]
        ax.set_xlabel("Subject")
        ax.set_xticklabels(subject_ids, rotation=45)
        ax.set_ylabel("Number of Seizures")
        ax.set_title("Seizure Counts by Subject and Type")
        ax.legend(title="Seizure Type")
        plt.tight_layout()
        plt.show()

    print(out)
    return out


def compare_ed_counts(subject_list, type="eeg", plot=True):
    """
    Count epileptic-discharge (ED) markers per subject.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects with a populated ``markers`` dictionary.
    type : {"eeg", "behavior", "both"}, optional
        Which ED family to count, matched by substring in the marker label:

        * ``"eeg"`` — labels containing ``"eeg_only"``
        * ``"behavior"`` — labels containing ``"eeg+behavior"``
        * ``"both"`` — labels containing ``"epileptiform"``

        Default is ``"eeg"``.
    plot : bool, optional
        If ``True``, show a bar chart. Default is ``True``.

    Returns
    -------
    out : dict
        ``{subject_id: ed_count}``. Also printed.
    """
    ed_key = {"eeg": "eeg_only", "behavior": "eeg+behavior", "both": "epileptiform"}[type]
    subject_ids = [s.subject_id for s in subject_list]

    # Sum matching marker lists across all dates for each subject.
    out = {
        subj.subject_id: sum(
            len(subj.markers[date][m])
            for date in subj.markers
            for m in subj.markers[date]
            if ed_key in m
        )
        for subj in subject_list
    }

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(subject_ids, [out[s] for s in subject_ids], color="#4C72B0")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Number of EDs")
        ax.set_title("Epileptic Discharge (ED) Counts by Subject")
        plt.tight_layout()
        plt.show()

    print(out)
    return out


def avg_marker_duration(subject_list, markers, avg="median", plot=True):
    """
    Mean or median marker duration per subject, with individual events overlaid.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects with a populated ``markers`` dictionary.
    markers : sequence of str
        Marker labels whose durations are pooled (e.g. Racine 4–6).
    avg : {"median", "mean"}, optional
        Summary statistic. Anything other than ``"median"`` uses the mean.
        Default is ``"median"``.
    plot : bool, optional
        If ``True``, show a bar chart with jittered event durations.
        Default is ``True``.

    Returns
    -------
    out : dict
        ``{subject_id: {"avg": float, "durations": list of float}}``.
        Also printed. Subjects with no matching events have ``avg=0``.
    """
    subject_ids = [s.subject_id for s in subject_list]
    durations_all = _get_marker_durations(subject_list, markers)
    avg_func = np.median if avg == "median" else np.mean

    # One summary value per subject; empty duration lists become 0.
    avg_durations = [avg_func(d) if d else 0 for d in durations_all]

    out = {
        sid: {"avg": a, "durations": d}
        for sid, a, d in zip(subject_ids, avg_durations, durations_all)
    }

    if plot:
        x_pos = np.arange(len(subject_ids))
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            x_pos,
            avg_durations,
            alpha=0.7,
            color="steelblue",
            label=f"{avg.capitalize()} duration",
            width=0.6,
        )

        # Overlay every individual event duration with a small x jitter.
        for i, durs in enumerate(durations_all):
            if durs:
                jitter = np.random.normal(0, 0.05, len(durs))
                ax.scatter(
                    i + jitter,
                    durs,
                    color="red",
                    alpha=0.5,
                    s=30,
                    zorder=10,
                    edgecolors="darkred",
                    linewidths=0.5,
                )

        for bar, d in zip(bars, avg_durations):
            if d > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{d:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xlabel("Subject ID", fontsize=12)
        ax.set_ylabel("Duration (seconds)", fontsize=12)
        marker_str = ", ".join(markers)
        ax.set_title(
            f"{marker_str} durations by subject\n, {avg} with individual values)",
            fontsize=14,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subject_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    print(out)
    return out


def longest_marker_duration(subject_list, markers, plot=True):
    """
    Longest marker duration per subject for the specified labels.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects with a populated ``markers`` dictionary.
    markers : sequence of str
        Marker labels whose durations are pooled.
    plot : bool, optional
        If ``True``, show a lollipop plot. Default is ``True``.

    Returns
    -------
    out : dict
        ``{subject_id: longest_duration_sec}``. Also printed.
        Subjects with no matching events have ``0``.
    """
    subject_ids = [s.subject_id for s in subject_list]
    durations_all = _get_marker_durations(subject_list, markers)
    longest = [max(d) if d else 0 for d in durations_all]
    out = {sid: ld for sid, ld in zip(subject_ids, longest)}

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(subject_ids))

        ax.vlines(x_pos, 0, longest, color="lightgray", linewidth=3, zorder=1)
        ax.scatter(
            x_pos,
            longest,
            s=120,
            color="steelblue",
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
        for x, y in zip(x_pos, longest):
            if y > 0:
                ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Subject ID", fontsize=12)
        ax.set_ylabel("Duration (seconds)", fontsize=12)
        marker_str = ", ".join(markers)
        ax.set_title(f"Longest {marker_str} duration by subject", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subject_ids, rotation=45)
        ax.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.show()

    print(out)
    return out


def plot_latency(
    subject_list,
    from_marker="ptz",
    to_markers=["racine4", "racine5", "racine6"],
    plot=True,
):
    """
    Latency from the first ``from_marker`` event to the first of ``to_markers``.

    Uses the chronologically first recording date (``min(subj.markers)``)
    for each subject.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects with a populated ``markers`` dictionary.
    from_marker : str, optional
        Label of the reference event (typically the first PTZ injection).
        Default is ``"ptz"``.
    to_markers : sequence of str, optional
        Candidate target labels; the earliest start time among them is used.
        Default is Racine 4–6.
    plot : bool, optional
        If ``True``, show a lollipop plot. Default is ``True``.

    Returns
    -------
    out : dict
        ``{subject_id: latency_sec}``. Also printed.
        Subjects with no matching target event have ``0.0``.
    """
    out = {}
    for subj in subject_list:
        # First recording date for this subject (string-sorted file tag).
        date = min(subj.markers)
        start_time = subj.markers[date][from_marker][0][0]

        # Earliest start among the requested target labels on that date.
        first_t = None
        for to_m in to_markers:
            if to_m in subj.markers[date]:
                t = subj.markers[date][to_m][0][0]
                if first_t is None or t < first_t:
                    first_t = t

        if first_t is None:
            out[subj.subject_id] = float(0)
            continue
        out[subj.subject_id] = first_t - start_time

    if plot:
        subject_ids = list(out.keys())
        latency_values = list(out.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(subject_ids))

        ax.vlines(x_pos, 0, latency_values, color="lightgray", linewidth=3, zorder=1)
        ax.scatter(
            x_pos,
            latency_values,
            s=120,
            color="steelblue",
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
        for x, y in zip(x_pos, latency_values):
            if y > 0:
                ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Subject ID", fontsize=12)
        ax.set_ylabel("Latency (seconds)", fontsize=12)
        ax.set_title(f"Latency from first {from_marker} to first {to_markers}", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subject_ids, rotation=45)
        ax.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.show()

    print(out)
    return out


def plot_ptz_doses_until_marker(
    subject_list,
    to_markers=["racine4", "racine5", "racine6"],
    plot=True,
):
    """
    Number of PTZ injections before the first of the target markers.

    Uses the chronologically first recording date for each subject.
    Subjects without a ``ptz`` marker or without any target marker are
    omitted from the result.

    Parameters
    ----------
    subject_list : sequence of EEGSubject
        Subjects with a populated ``markers`` dictionary.
    to_markers : sequence of str, optional
        Candidate target labels; the earliest start time among them is used.
        Default is Racine 4–6.
    plot : bool, optional
        If ``True``, show a lollipop plot. Default is ``True``.

    Returns
    -------
    out : dict
        ``{subject_id: n_doses}``. Also printed.
    """
    out = {}
    for subj in subject_list:
        date = min(subj.markers)
        if "ptz" not in subj.markers[date]:
            continue
        ptz_events = subj.markers[date]["ptz"]

        # Earliest target-marker onset, or skip the subject if none exist.
        first_t = min(
            (subj.markers[date][m][0][0] for m in to_markers if m in subj.markers[date]),
            default=None,
        )
        if first_t is None:
            continue

        # Count PTZ events whose start precedes the first target marker.
        out[subj.subject_id] = sum(1 for (s, e) in ptz_events if s < first_t)

    if plot:
        subject_ids = list(out.keys())
        dose_counts = list(out.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(subject_ids))

        ax.vlines(x_pos, 0, dose_counts, color="lightgray", linewidth=3, zorder=1)
        ax.scatter(
            x_pos,
            dose_counts,
            s=120,
            color="steelblue",
            edgecolors="black",
            linewidths=0.6,
            zorder=2,
        )
        for x, y in zip(x_pos, dose_counts):
            if y > 0:
                ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Subject ID", fontsize=12)
        ax.set_ylabel("Number of PTZ doses", fontsize=12)
        ax.set_title(f"PTZ doses until first of {to_markers}", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subject_ids, rotation=45)
        ax.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        plt.show()

    print(out)
    return out
