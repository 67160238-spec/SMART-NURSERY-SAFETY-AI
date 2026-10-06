"""Run the whole hazard v2 pipeline on your own PC (tested plan for an RTX 4060, 8 GB).

Same steps as the Colab run, in one command, Windows/Mac/Linux:
  1. download data (HOD, Sohas, COCO val2017, COCO train2017 annotations)
  2. build the dataset with the yolo11x teacher        (tools/build_hazard_dataset.py)
  3. add COCO train scissors/knife replay              (tools/fetch_coco_replay.py)
  4. paste tiny scissors/knives (= far away)           (tools/make_small_objects.py)
  5. fine-tune from v1                                 (tools/train_hazard_model.py)
  6. evaluate base / v1 / v2 / v2+zoom + simulated distance (tools/eval_hazard_model.py)
Each step is skipped if its output already exists, so you can stop and re-run.

    pip install ultralytics==8.4.136
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   (NVIDIA GPU)
    python tools/run_v2_local.py --work D:/hazard_work
Needs ~15 GB free disk. Results: models/hazard/hazard_v2.pt, docs/eval/hazard_v2_eval.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*cmd: str) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def download(url: str, dst: Path) -> None:
    if dst.exists():
        return
    print(f">> download {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def sparse_clone(url: str, dst: Path, path: str) -> None:
    if dst.exists():
        return
    run("git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", url, str(dst))
    subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], check=True, cwd=dst)
    (dst / ".git" / "info" / "sparse-checkout").write_text(path + "\n")
    subprocess.run(["git", "checkout", "HEAD"], check=True, cwd=dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="folder for data (not inside the repo)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16, help="lower to 8 if you get CUDA out of memory")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    w = args.work.resolve()
    w.mkdir(parents=True, exist_ok=True)
    data = w / "hazard_full"

    # 1. data
    sparse_clone("https://github.com/poori-nuna/HOD-Benchmark-Dataset.git", w / "hod", "/dataset/class/knife/")
    sparse_clone("https://github.com/ari-dasci/OD-WeaponDetection.git", w / "odw",
                 "/Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5/")
    coco_zip = w / "coco2017val.zip"
    download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017val.zip", coco_zip)
    if not (w / "coco").exists():
        zipfile.ZipFile(coco_zip).extractall(w / "coco")
    ann = w / "instances_train2017.json"
    if not ann.exists():
        az = w / "annotations_trainval2017.zip"
        download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", az)
        with zipfile.ZipFile(az) as z, open(ann, "wb") as f:
            f.write(z.read("annotations/instances_train2017.json"))

    # 2-4. dataset
    if not (data / "data.yaml").exists():
        run(PY, "tools/build_hazard_dataset.py", "--coco", str(w / "coco" / "coco"), "--hod", str(w / "hod"),
            "--sohas", str(w / "odw" / "Weapons and similar handled objects" / "Sohas_weapon-Detection-YOLOv5"),
            "--out", str(data), "--teacher", "yolo11x.pt", "--coco-train", "2000", "--coco-val", "300",
            "--device", args.device)
    if not (data / "instances.json").exists():
        run(PY, "tools/fetch_coco_replay.py", "--ann", str(ann), "--data", str(data),
            "--knife", "1000", "--others", "1000")
    if not any((data / "images" / "train").glob("synsmall_*")):
        run(PY, "tools/make_small_objects.py", "--data", str(data), "--n", "2000")

    # 5. train from v1 (keeps the knife gains)
    if not (ROOT / "models" / "hazard" / "hazard_v2.pt").exists():
        run(PY, "tools/train_hazard_model.py", "--data", str(data / "data.yaml"),
            "--base", "models/hazard/hazard_v1_full.pt", "--name", "hazard_v2", "--epochs", str(args.epochs),
            "--imgsz", "960", "--batch", str(args.batch), "--freeze", "0", "--lr0", "0.002",
            "--device", args.device, "--workers", str(args.workers), "--project", str(w / "runs"))

    # 6. evaluate
    run(PY, "tools/eval_hazard_model.py", "--data", str(data), "--device", args.device,
        "--model", "base=yolo11n.pt@960", "--model", "v1_full=models/hazard/hazard_v1_full.pt@960",
        "--model", "v2=models/hazard/hazard_v2.pt@960", "--model", "v2_zoom=models/hazard/hazard_v2.pt@960+zoom",
        "--sim-scales", "2,3,4", "--out", "docs/eval/hazard_v2_eval.md")
    print("\nDONE -> docs/eval/hazard_v2_eval.md  (send it to Claude, or read the tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
