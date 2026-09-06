"""
Prepare a fixed COCO val2017 subset for accuracy evaluation.

Runs on the laptop. Downloads the annotations, selects a deterministic subset of
images, writes a trimmed annotation file, and fetches only those images.

    python evaluation/prepare_coco.py --n 500

The subset must be FIXED and identical across every runtime, or mAP comparisons
are meaningless. Selection is by sorted image id with a fixed seed, so re-running
this produces the same subset on any machine.

Downloads ~250 MB of annotations once; the images themselves are ~50 MB for 500.
"""

import argparse
import json
import os
import random
import sys
import zipfile

try:
    from urllib.request import urlretrieve
except ImportError:                                   # pragma: no cover
    from urllib import urlretrieve                    # noqa: F401  (py2 fallback)


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/%s"


def report(count, block_size, total):
    if total <= 0:
        return
    pct = min(100.0, 100.0 * count * block_size / total)
    sys.stdout.write("\r  %.1f%%" % pct)
    sys.stdout.flush()


def ensure_annotations(root):
    ann_path = os.path.join(root, "annotations", "instances_val2017.json")
    if os.path.isfile(ann_path):
        print("annotations already present")
        return ann_path

    zip_path = os.path.join(root, "annotations_trainval2017.zip")
    if not os.path.isfile(zip_path):
        print("downloading annotations (~250 MB)...")
        urlretrieve(ANNOTATIONS_URL, zip_path, report)
        print("")

    print("extracting instances_val2017.json...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_val2017.json", root)

    # The zip is only needed once and is far larger than what was extracted.
    os.remove(zip_path)
    return ann_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/coco")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0,
                   help="fixed so the subset is reproducible on any machine")
    args = p.parse_args()

    img_dir = os.path.join(args.root, "val2017")
    for d in (args.root, img_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    ann_path = ensure_annotations(args.root)

    print("loading annotations...")
    with open(ann_path) as f:
        coco = json.load(f)

    # Sort before sampling: dict/JSON ordering is not guaranteed stable across
    # versions, and the subset must be identical everywhere.
    all_images = sorted(coco["images"], key=lambda im: im["id"])
    random.Random(args.seed).shuffle(all_images)
    subset = sorted(all_images[:args.n], key=lambda im: im["id"])
    subset_ids = set(im["id"] for im in subset)

    subset_anns = [a for a in coco["annotations"] if a["image_id"] in subset_ids]

    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": subset,
        "annotations": subset_anns,
        "categories": coco["categories"],
    }

    subset_path = os.path.join(args.root, "instances_val2017_subset%d.json" % args.n)
    with open(subset_path, "w") as f:
        json.dump(out, f)

    print("subset: %d images, %d annotations" % (len(subset), len(subset_anns)))
    print("wrote %s" % subset_path)

    missing = [im for im in subset
               if not os.path.isfile(os.path.join(img_dir, im["file_name"]))]

    if missing:
        print("downloading %d images..." % len(missing))
        for i, im in enumerate(missing, 1):
            dest = os.path.join(img_dir, im["file_name"])
            urlretrieve(IMAGE_URL % im["file_name"], dest)
            if i % 25 == 0 or i == len(missing):
                sys.stdout.write("\r  %d / %d" % (i, len(missing)))
                sys.stdout.flush()
        print("")
    else:
        print("all images already present")

    total_mb = sum(
        os.path.getsize(os.path.join(img_dir, im["file_name"])) for im in subset
    ) / (1024.0 * 1024.0)

    print("")
    print("images   : %s  (%.1f MB)" % (img_dir, total_mb))
    print("subset   : %s" % subset_path)
    print("")
    print("Copy both to the Jetson:")
    print("  scp -r %s user@<jetson>:~/EdgeVision/data/" % args.root)


if __name__ == "__main__":
    sys.exit(main())
