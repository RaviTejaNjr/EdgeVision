"""
Compose two rendered videos side by side, played at their real relative speed.

Run on the laptop after copying the two MP4s and their JSON metadata across.

    python benchmarks/compose_demo.py \
        --left  assets/demo_torchscript_fp32.mp4 \
        --right assets/demo_tensorrt_fp16.mp4

The point of retiming is that both clips cover the same 300 source frames, but one
took far longer to produce them. Playing both at 30 fps would hide the difference
entirely -- the FPS labels would differ while the videos finished together.

Instead each side is played at its measured throughput, so the faster runtime
visibly finishes first and the slower one keeps going. The speed difference
becomes something you see rather than something you read.
"""

import argparse
import json
import os
import subprocess
import sys


def load_meta(video_path):
    meta_path = video_path.replace(".mp4", ".json")
    if not os.path.isfile(meta_path):
        raise SystemExit(
            "%s not found. render_demo.py writes it alongside the video." % meta_path
        )
    with open(meta_path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True, help="slower runtime, shown on the left")
    p.add_argument("--right", required=True, help="faster runtime")
    p.add_argument("--layout", default="vstack", choices=["vstack", "hstack"],
                   help="vstack gives a 16:9-ish frame that survives downscaling "
                        "to a README GIF; hstack is 32:9 and very wide")
    p.add_argument("--divider", type=int, default=6,
                   help="pixels of separator between the two panes")
    p.add_argument("--out", default="assets/demo_comparison.mp4")
    p.add_argument("--gif", default="assets/demo_comparison.gif")
    p.add_argument("--gif-width", type=int, default=640,
                   help="GIF width. With vstack, 640 halves the source and keeps "
                        "the overlay text legible; hstack needs more.")
    p.add_argument("--gif-fps", type=int, default=12)
    p.add_argument("--gif-seconds", type=float, default=12.0)
    p.add_argument("--gif-speed", type=float, default=3.0,
                   help="playback multiplier for the GIF. The composed MP4 is "
                        "already retimed to real throughput, so a GIF sampled "
                        "from it plays at a fraction of real speed. Speeding it "
                        "up restores watchable motion; the ratio between the two "
                        "panes is unaffected, so the comparison stays honest.")
    p.add_argument("--no-retime", action="store_true",
                   help="play both at 30 fps instead of real relative speed")
    args = p.parse_args()

    lm = load_meta(args.left)
    rm = load_meta(args.right)

    print("left  : %-22s %6.2f FPS" % (lm["label"], lm["mean_fps"]))
    print("right : %-22s %6.2f FPS" % (rm["label"], rm["mean_fps"]))
    print("speedup: %.2fx" % (rm["mean_fps"] / lm["mean_fps"]))
    print("")

    # A visible separator between the panes. Without it the two frames butt
    # together and read as one confusing image.
    if args.layout == "vstack":
        pad = "pad=iw:ih+%d:0:0:color=0x202020" % args.divider
        stack = "vstack=inputs=2"
    else:
        pad = "pad=iw+%d:ih:0:0:color=0x202020" % args.divider
        stack = "hstack=inputs=2"

    if args.no_retime:
        left_filter = "[0:v]%s[l]" % pad
        right_filter = "[1:v]null[r]"
    else:
        # Videos were written at 30 fps. setpts rescales presentation timestamps
        # so each plays at the throughput actually measured.
        left_rate = lm["mean_fps"] / 30.0
        right_rate = rm["mean_fps"] / 30.0
        left_filter = "[0:v]setpts=PTS/%.6f,%s[l]" % (left_rate, pad)
        right_filter = "[1:v]setpts=PTS/%.6f[r]" % right_rate
        print("retiming: left x%.3f, right x%.3f" % (left_rate, right_rate))

    filter_complex = "%s;%s;[l][r]%s[v]" % (left_filter, right_filter, stack)

    cmd = [
        "ffmpeg", "-y",
        "-i", args.left,
        "-i", args.right,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",          # required for browser playback
        args.out,
    ]

    print("composing %s ..." % args.out)
    subprocess.run(cmd, check=True)

    # A GIF is what renders inline in a GitHub README.
    #
    # setpts comes FIRST and is the important part. The composed MP4 is already
    # retimed to real throughput, so sampling a GIF from it directly plays back
    # at a fraction of real speed -- the two slowdowns compound. Speeding up here
    # restores watchable motion. Both panes are scaled equally, so the ratio
    # between them is unchanged and the FPS counters still read true.
    #
    # split/palettegen/paletteuse in a single pass: the default 216-colour GIF
    # palette makes video look terrible, and a single filtergraph avoids the
    # temporary palette file entirely.
    vf = ("setpts=PTS/%.4f,fps=%d,scale=%d:-1:flags=lanczos,"
          "split[a][b];[a]palettegen[p];[b][p]paletteuse"
          % (args.gif_speed, args.gif_fps, args.gif_width))

    subprocess.run([
        "ffmpeg", "-y", "-i", args.out,
        "-vf", vf,
        "-t", str(args.gif_seconds),
        "-loop", "0",
        args.gif,
    ], check=True)

    print("")
    print("video : %s" % args.out)
    print("gif   : %s  (%.1f MB)"
          % (args.gif, os.path.getsize(args.gif) / (1024.0 * 1024.0)))
    print("")
    print("If the GIF is over ~10 MB, reduce --gif-width, --gif-fps or")
    print("--gif-seconds; GitHub will not render very large files inline.")
    print("")
    print("NOTE: the GIF plays at %.0fx real speed. Say so in the README caption "
          "-- the\n      FPS counters are true, the wall-clock is not."
          % args.gif_speed)


if __name__ == "__main__":
    sys.exit(main())
