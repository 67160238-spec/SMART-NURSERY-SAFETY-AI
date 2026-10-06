"""v2: add COCO train2017 images (full 80-class ground truth) to a hazard dataset.

Why (from docs/eval/hazard_v1_full_eval.md): v1 lost scissors (-33 AP) and
person (-5.7 AP) because its training set had ~20 scissors boxes. COCO
train2017 has ~1,000 scissors images with exact labels for every class, and it
is the data yolo11 was trained on, so replaying it is the standard cure for
forgetting. COCO val2017 (our test split) is never touched here.

Also writes `instances.json`: every scissors/knife instance with its polygon
(in the saved, resized image), used by tools/make_small_objects.py to paste
tiny copies (= objects far from the camera).

Usage (Colab):
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -j annotations_trainval2017.zip annotations/instances_train2017.json
    python tools/fetch_coco_replay.py --ann instances_train2017.json \
        --data /content/hazard_full --knife 1500 --others 1500
Local test with val annotations + local images: --ann instances_val2017.json --local-images <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_hazard_dataset import KNIFE, MAX_SIDE, SCISSORS, format_yolo  # noqa: E402

SEED = 0


def coco80_map(categories: list[dict]) -> dict[int, int]:
    """COCO category id (1..90, with gaps) -> Ultralytics 0..79 (sorted id order)."""
    return {c["id"]: i for i, c in enumerate(sorted(categories, key=lambda c: c["id"]))}


def select_images(anns_by_img: dict[int, list], cat80: dict[int, int], n_knife: int, n_others: int,
                  seed: int = SEED) -> list[int]:
    """All scissors images + n_knife knife images + n_others random others (deterministic)."""
    rng = random.Random(seed)
    ids = sorted(anns_by_img)
    sc = [i for i in ids if any(cat80[a["category_id"]] == SCISSORS for a in anns_by_img[i])]
    sc_set = set(sc)
    kn = [i for i in ids if i not in sc_set and any(cat80[a["category_id"]] == KNIFE for a in anns_by_img[i])]
    taken = sc_set | set(kn)
    rest = [i for i in ids if i not in taken]
    rng.shuffle(kn)
    rng.shuffle(rest)
    return sc + kn[:n_knife] + rest[:n_others]


def yolo_boxes(anns: list[dict], cat80: dict[int, int], w: int, h: int) -> list[tuple]:
    out = []
    for a in anns:
        if a.get("iscrowd"):
            continue
        x, y, bw, bh = a["bbox"]
        if bw < 1 or bh < 1:
            continue
        out.append((cat80[a["category_id"]], (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dataset dir from build_hazard_dataset.py")
    ap.add_argument("--knife", type=int, default=1500)
    ap.add_argument("--others", type=int, default=1500)
    ap.add_argument("--local-images", type=Path, help="read images from here instead of coco_url")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    import cv2
    import numpy as np

    coco = json.loads(args.ann.read_text())
    cat80 = coco80_map(coco["categories"])
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    chosen = select_images(anns_by_img, cat80, args.knife, args.others)
    print(f"[coco-replay] {len(chosen)} images selected")

    out_img, out_lab = args.data / "images" / "train", args.data / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    instances = []

    def fetch(img_id: int):
        meta = imgs[img_id]
        name = f"cocotrain_{img_id:012d}"
        dst = out_img / f"{name}.jpg"
        try:
            if args.local_images:
                data = (args.local_images / meta["file_name"]).read_bytes()
            else:
                with urllib.request.urlopen(meta["coco_url"], timeout=30) as r:
                    data = r.read()
        except Exception as exc:  # a missing image is skipped, never fatal
            return None, f"{meta['file_name']}: {exc}"
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            return None, f"{meta['file_name']}: undecodable"
        h, w = im.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        anns = anns_by_img[img_id]
        (out_lab / f"{name}.txt").write_text(format_yolo(yolo_boxes(anns, cat80, w, h)))
        inst = []
        for a in anns:
            c = cat80[a["category_id"]]
            if c in (KNIFE, SCISSORS) and not a.get("iscrowd") and isinstance(a.get("segmentation"), list):
                for poly in a["segmentation"]:
                    if len(poly) >= 6:
                        inst.append({"image": str(dst), "cls": c, "poly": [round(v * s, 1) for v in poly]})
        return inst, None

    errors = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for k, (inst, err) in enumerate(pool.map(fetch, chosen)):
            if err:
                errors += 1
            else:
                instances += inst
            if k % 500 == 0:
                print(f"[coco-replay] {k}/{len(chosen)}", flush=True)
    (args.data / "instances.json").write_text(json.dumps(instances))
    for cache in (args.data / "labels").glob("*.cache"):
        cache.unlink()
    n_sc = sum(i["cls"] == SCISSORS for i in instances)
    print(f"[coco-replay] done: {len(chosen) - errors} images, {errors} failed, "
          f"{n_sc} scissors + {len(instances) - n_sc} knife instances with masks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
