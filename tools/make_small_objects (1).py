"""v2: paste TINY scissors/knives into training images (= objects far from the camera).

Why (docs/eval/hazard_v1_full_eval.md): recall on objects smaller than 3% of
the image was 13% (base) and 0% (v1); the web photos are close-ups, so the
model almost never saw a 10-30 px blade. Real far-away data is not available,
so we synthesise it: cut real scissors/knife instances out of COCO train with
their polygon masks (tools/fetch_coco_replay.py writes them), shrink them to
8-40 px, soften them like a distant object (slight blur, feathered edge) and
paste 1-3 per image, mostly next to a person's box (where hands are).

Each output image is a NEW file (background + pastes) with the background's
labels plus the pasted boxes. Only train split; test is untouched.

Usage:
    python tools/make_small_objects.py --data /content/hazard_full --n 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_hazard_dataset import KNIFE, PERSON, SCISSORS, format_yolo, iou_xywh, parse_yolo  # noqa: E402

SEED = 0


def place_near_person(persons: list[tuple], w: int, h: int, pw: int, ph: int, rng: random.Random):
    """Top-left corner for a pw x ph patch: beside a random person's hands (60%) or anywhere."""
    if persons and rng.random() < 0.6:
        _, xc, yc, bw, bh = rng.choice(persons)
        # hands are around the middle third of the body height, at the sides
        x = (xc + rng.choice((-1, 1)) * bw * rng.uniform(0.2, 0.6)) * w - pw / 2
        y = (yc - bh / 2 + bh * rng.uniform(0.3, 0.7)) * h - ph / 2
    else:
        x, y = rng.uniform(0, w - pw), rng.uniform(0, h - ph)
    return int(min(max(0, x), w - pw)), int(min(max(0, y), h - ph))


def cut_instance(src, poly: list[float]):
    """(BGR patch, float mask 0..1) of one polygon, cropped to its bounding box."""
    import cv2
    import numpy as np
    pts = np.array(poly, np.float32).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(0)).astype(int)
    x2, y2 = np.ceil(pts.max(0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(src.shape[1], x2), min(src.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(mask, [np.round(pts - [x1, y1]).astype(np.int32)], 255)
    return src[y1:y2, x1:x2].copy(), mask.astype(np.float32) / 255.0


def shrink(patch, mask, target: int, angle: float, flip: bool):
    """Rotate, flip and resize so the longest side is `target` px; distant-object softening."""
    import cv2
    import numpy as np
    if flip:
        patch, mask = patch[:, ::-1], mask[:, ::-1]
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    patch = cv2.warpAffine(np.ascontiguousarray(patch), m, (nw, nh), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (nw, nh), flags=cv2.INTER_LINEAR)
    s = target / max(nw, nh)
    size = (max(2, round(nw * s)), max(2, round(nh * s)))
    patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)  # feathered edge, no hard cut-out border
    return patch, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000, help="synthetic images to write")
    ap.add_argument("--min-px", type=int, default=8)
    ap.add_argument("--max-px", type=int, default=40)
    ap.add_argument("--scissors-share", type=float, default=0.6, help="scissors is the primary gate class")
    args = ap.parse_args()

    import cv2
    import numpy as np

    rng = random.Random(SEED)
    inst = json.loads((args.data / "instances.json").read_text())
    by_cls = {c: [i for i in inst if i["cls"] == c] for c in (SCISSORS, KNIFE)}
    if not by_cls[SCISSORS] and not by_cls[KNIFE]:
        print("no instances - run tools/fetch_coco_replay.py first")
        return 1
    img_dir, lab_dir = args.data / "images" / "train", args.data / "labels" / "train"
    backgrounds = sorted(p for p in img_dir.glob("*.jpg") if not p.name.startswith("synsmall_"))
    rng.shuffle(backgrounds)
    src_cache: dict[str, object] = {}
    written = pasted = 0
    for k in range(args.n):
        bg_path = backgrounds[k % len(backgrounds)]
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        h, w = bg.shape[:2]
        labels = parse_yolo((lab_dir / (bg_path.stem + ".txt")).read_text()) \
            if (lab_dir / (bg_path.stem + ".txt")).exists() else []
        persons = [b for b in labels if b[0] == PERSON]
        new = []
        for _ in range(rng.randint(1, 3)):
            cls = SCISSORS if (rng.random() < args.scissors_share and by_cls[SCISSORS]) or not by_cls[KNIFE] \
                else KNIFE
            it = rng.choice(by_cls[cls])
            if it["image"] not in src_cache:
                if len(src_cache) > 500:
                    src_cache.clear()
                src_cache[it["image"]] = cv2.imread(it["image"])
            src = src_cache[it["image"]]
            if src is None:
                continue
            patch, mask = cut_instance(src, it["poly"])
            if patch is None:
                continue
            target = rng.randint(args.min_px, args.max_px)
            patch, mask = shrink(patch, mask, target, rng.uniform(-35, 35), rng.random() < 0.5)
            ph, pw = mask.shape
            if pw >= w or ph >= h:
                continue
            x, y = place_near_person(persons, w, h, pw, ph, rng)
            box = (cls, (x + pw / 2) / w, (y + ph / 2) / h, pw / w, ph / h)
            if any(iou_xywh(box, b) > 0.05 for b in labels + new if b[0] in (KNIFE, SCISSORS)):
                continue
            roi = bg[y:y + ph, x:x + pw].astype(np.float32)
            a = mask[..., None]
            # match brightness of the surroundings a little (distant objects take the scene's light)
            gain = np.clip((roi.mean() + 1) / (patch.mean() + 1), 0.7, 1.3)
            bg[y:y + ph, x:x + pw] = (roi * (1 - a) + np.clip(patch * gain, 0, 255) * a).astype(np.uint8)
            new.append(box)
        if not new:
            continue
        name = f"synsmall_{k:05d}_{bg_path.stem}"[:120]
        cv2.imwrite(str(img_dir / f"{name}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lab_dir / f"{name}.txt").write_text(format_yolo(labels + new))
        written += 1
        pasted += len(new)
        if k % 500 == 0:
            print(f"[small] {k}/{args.n}", flush=True)
    for cache in (args.data / "labels").glob("*.cache"):
        cache.unlink()
    print(f"[small] wrote {written} images with {pasted} tiny scissors/knives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
