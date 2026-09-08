"""
Plot throughput against temperature over the course of a run.

Runs on the laptop -- matplotlib is not installed on the Jetson, and there is no
reason to put it there.

    python benchmarks/make_report.py --run <run_id>
    python benchmarks/make_report.py --run <run_id> --compare <other_run_id>

Reads results/raw/<run_id>.npz and the matching row from results/speed.csv, and
writes results/plots/thermal_<run_id>.png.

THE ROLLING MEAN MATTERS. The test video is 1040 frames and loops during a long
run, so scene density repeats on a fixed cycle -- detection count drives NMS cost,
which drives frame time. Raw per-frame FPS therefore carries a periodic wobble
that has nothing to do with temperature. Averaging over exactly one loop period
cancels it and leaves the thermal trend visible.
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")               # no display on a headless machine
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed -- pip install matplotlib")


VIDEO_FRAMES = 1040                     # one full loop of data/test_video.mp4


def load_row(results_dir, run_id):
    """Find this run's metadata. Returns {} if the CSV predates the run."""
    for name in ("speed.csv", "speed_part2to5.csv", "speed_exploratory.csv"):
        path = os.path.join(results_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("run_id", "").startswith(run_id):
                    return row
    return {}


def rolling(values, window):
    """Centred rolling mean, with the window clipped at the array bounds."""
    if len(values) < window:
        window = max(1, len(values) // 4)
    kernel = np.ones(window) / float(window)
    # 'same' keeps the output length; edges are biased low but only cosmetically.
    return np.convolve(values, kernel, mode="same")


def plot_run(ax_fps, ax_temp, npz, row, label, colour):
    wall = npz["wall_time_s"]
    total_ms = npz["total_ms"]

    inst_fps = 1000.0 / total_ms
    smoothed = rolling(inst_fps, VIDEO_FRAMES)

    # Raw samples behind the smoothed line: shows the spread without implying
    # the wobble is signal.
    ax_fps.plot(wall, inst_fps, color=colour, alpha=0.12, linewidth=0.5)
    ax_fps.plot(wall, smoothed, color=colour, linewidth=2.0,
                label="%s -- FPS (%d-frame mean)" % (label, VIDEO_FRAMES))

    if "temp_wall_s" in npz and len(npz["temp_wall_s"]) > 1:
        ax_temp.plot(npz["temp_wall_s"], npz["gpu_temp_c"], color=colour,
                     linestyle="--", linewidth=1.5,
                     label="%s -- GPU temp" % label)

        if "cpu_temp_c" in npz and np.isfinite(npz["cpu_temp_c"]).any():
            ax_temp.plot(npz["temp_wall_s"], npz["cpu_temp_c"], color=colour,
                         linestyle=":", linewidth=1.2, alpha=0.7,
                         label="%s -- CPU temp" % label)

        # Mark where the governor engaged. This is the event the whole curve is
        # read against, so it gets a vertical line rather than a legend entry.
        fan = npz.get("fan_pwm")
        if fan is not None and len(fan) == len(npz["temp_wall_s"]):
            engaged = np.where(np.asarray(fan) > 0)[0]
            if len(engaged):
                t = npz["temp_wall_s"][engaged[0]]
                temp_at = npz["gpu_temp_c"][engaged[0]]
                ax_fps.axvline(t, color=colour, linestyle="-.", alpha=0.6)
                ax_fps.annotate("fan on (%.0f C)" % temp_at,
                                xy=(t, ax_fps.get_ylim()[0]),
                                xytext=(t + 5, ax_fps.get_ylim()[0] + 0.3),
                                fontsize=8, color=colour)

    return {
        "frames": len(total_ms),
        "duration_s": float(wall[-1]),
        "fps_first_60s": float(np.mean(inst_fps[wall <= 60])) if (wall <= 60).any() else float("nan"),
        "fps_last_60s": float(np.mean(inst_fps[wall >= wall[-1] - 60])),
        "temp_start": float(npz["gpu_temp_c"][0]) if "gpu_temp_c" in npz and len(npz["gpu_temp_c"]) else float("nan"),
        "temp_end": float(npz["gpu_temp_c"][-1]) if "gpu_temp_c" in npz and len(npz["gpu_temp_c"]) else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run_id, or a unique prefix")
    p.add_argument("--compare", default=None, help="a second run_id to overlay")
    p.add_argument("--results-dir", dest="results_dir", default="results")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    raw_dir = os.path.join(args.results_dir, "raw")

    def find(run_id):
        for name in os.listdir(raw_dir):
            if name.startswith(run_id) and name.endswith(".npz"):
                return os.path.join(raw_dir, name), name[:-4]
        raise SystemExit("no .npz matching %s in %s" % (run_id, raw_dir))

    path_a, id_a = find(args.run)
    npz_a = np.load(path_a)
    row_a = load_row(args.results_dir, id_a)

    fig, ax_fps = plt.subplots(figsize=(11, 5.5))
    ax_temp = ax_fps.twinx()

    label_a = "%s %s" % (row_a.get("runtime", "run"), row_a.get("precision", ""))
    if row_a.get("power_mode"):
        label_a += " @ %s" % row_a["power_mode"]

    stats_a = plot_run(ax_fps, ax_temp, npz_a, row_a, label_a, "#1f77b4")
    summaries = [(label_a, id_a, stats_a)]

    if args.compare:
        path_b, id_b = find(args.compare)
        npz_b = np.load(path_b)
        row_b = load_row(args.results_dir, id_b)
        label_b = "%s %s" % (row_b.get("runtime", "run"), row_b.get("precision", ""))
        if row_b.get("power_mode"):
            label_b += " @ %s" % row_b["power_mode"]
        stats_b = plot_run(ax_fps, ax_temp, npz_b, row_b, label_b, "#d62728")
        summaries.append((label_b, id_b, stats_b))

    ax_fps.set_xlabel("elapsed (s)")
    ax_fps.set_ylabel("end-to-end FPS")
    ax_temp.set_ylabel("temperature (C)")
    ax_fps.grid(alpha=0.25)

    title = "Sustained throughput and thermal behaviour"
    if row_a.get("host_profile"):
        title += "\\n%s" % row_a["host_profile"]
    ax_fps.set_title(title, fontsize=11)

    # One legend for both axes, or matplotlib draws two overlapping boxes.
    h1, l1 = ax_fps.get_legend_handles_labels()
    h2, l2 = ax_temp.get_legend_handles_labels()
    ax_fps.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, framealpha=0.9)

    plots_dir = os.path.join(args.results_dir, "plots")
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
    out = args.out or os.path.join(plots_dir, "thermal_%s.png" % id_a)

    fig.tight_layout()
    fig.savefig(out, dpi=140)

    print("wrote %s" % out)
    print("")
    for label, run_id, st in summaries:
        drop = 100.0 * (st["fps_first_60s"] - st["fps_last_60s"]) / st["fps_first_60s"]
        print("%s  [%s]" % (label, run_id))
        print("  frames          : %d over %.0f s" % (st["frames"], st["duration_s"]))
        print("  FPS first 60 s  : %.2f" % st["fps_first_60s"])
        print("  FPS last 60 s   : %.2f" % st["fps_last_60s"])
        print("  degradation     : %+.2f%%" % -drop)
        print("  GPU temp        : %.1f -> %.1f C" % (st["temp_start"], st["temp_end"]))
        print("")


if __name__ == "__main__":
    sys.exit(main())
