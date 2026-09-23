"""Audit, train and compare football models without replacing deployed weights.

Run from the repository root; see docs/TRAINING.md. No dataset is downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def prepare_dataset(source, output, task):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if task == "pitch" and data.get("kpt_shape") != [32, 3]:
        raise ValueError("Pitch labels must contain the 32 landmarks in SoccerPitchConfiguration order.")
    names = data["names"]
    names = [names[key] for key in sorted(names)] if isinstance(names, dict) else names
    expected = {"pitch": ["pitch"], "ball": ["ball"],
                "players": ["ball", "goalkeeper", "player", "referee"]}[task]
    if names != expected:
        raise ValueError(f"Expected classes {expected}, found {names}.")
    manifest, hashes, errors = {}, {}, []
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        raw = Path(data[split])
        base = Path(data.get("path", source.parent))
        if not base.is_absolute():
            base = source.parent / base
        candidates = [base / raw, source.parent / raw,
                      source.parent / ("valid" if split == "val" else split) / "images"]
        folder = next((p.resolve() for p in candidates if p.is_dir()), None)
        if folder is None:
            raise ValueError(f"Missing image directory for {split}: {raw}")
        data[split] = str(folder)
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"Empty split: {split}")
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                errors.append(f"Exact image duplicated across {hashes[digest]}/{split}: {image.name}")
            hashes[digest] = split
            label = folder.parent / "labels" / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"Missing label: {label.name}")
                continue
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue  # Explicit empty labels are background images.
                values = np.fromstring(line, sep=" ")
                size = 101 if task == "pitch" else 5
                if len(values) != size or not np.isfinite(values).all():
                    errors.append(f"Invalid label shape/values: {label.name}")
                    continue
                if values[0] != int(values[0]) or not 0 <= values[0] < len(names):
                    errors.append(f"Invalid class: {label.name}")
                if np.any((values[1:5] < 0) | (values[1:5] > 1)):
                    errors.append(f"Invalid normalized box: {label.name}")
                if task == "pitch":
                    kpts = values[5:].reshape(32, 3)
                    visible = kpts[:, 2] > 0
                    if not np.isin(kpts[:, 2], [0, 1, 2]).all() or np.any(
                        (kpts[visible, :2] < 0) | (kpts[visible, :2] > 1)
                    ):
                        errors.append(f"Invalid keypoints: {label.name}")
        manifest[split] = len(images)
    output.mkdir(parents=True, exist_ok=True)
    audit = {"task": task, "counts": manifest, "errors": errors,
             "source": source.name, "license": data.get("roboflow", {}).get("license"),
             "dataset_fingerprint": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest(),
             "limitations": "Exact-byte duplicates only; split by match/camera before claiming generalisation."}
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if errors:
        raise ValueError(f"Dataset audit failed ({len(errors)} errors); see {output / 'dataset_audit.json'}")
    data.pop("path", None)
    normalized = output / "dataset.yaml"
    normalized.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return normalized, audit


def evaluate(model, data, args, name):
    metrics = model.val(data=str(data), split="val", imgsz=args.imgsz,
                        batch=args.batch, device=args.device, workers=0,
                        project=str(Path(args.output).resolve()), name=name,
                        plots=False, verbose=False)
    return {"metrics": {k: float(v) for k, v in metrics.results_dict.items()},
            "speed_ms_per_image": metrics.speed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--task", choices=["pitch", "ball", "players"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--model", help="YOLO11 pretrained weights or a checkpoint to fine-tune")
    parser.add_argument("--baseline", help="Existing checkpoint evaluated on the same validation split")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.audit_only and (Path(args.output) / 'comparison.json').exists():
        parser.error('This output already contains an experiment. Choose another directory.')
    data, audit = prepare_dataset(args.data, args.output, args.task)
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return
    from ultralytics import YOLO, __version__
    import torch
    model_name = args.model or ("yolo11n-pose.pt" if args.task == "pitch" else "yolo11n.pt")
    model = YOLO(model_name)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=0,
                seed=args.seed, deterministic=True, project=str(Path(args.output).resolve()),
                name="train", exist_ok=False, patience=20, cache=False,
                fliplr=0.0, flipud=0.0, mosaic=0.0 if args.task == "pitch" else 1.0,
                degrees=0.0, plots=False)
    best = Path(model.trainer.best)
    results = {"status": "experimental_not_promoted", "task": args.task,
               "epochs_requested": args.epochs, "seed": args.seed,
               "imgsz": args.imgsz, "model": model_name, "checkpoint": str(best),
               "dataset": audit, "ultralytics": __version__, "torch": torch.__version__,
               "candidate": evaluate(YOLO(best), data, args, "candidate_val")}
    if args.baseline:
        results["baseline"] = evaluate(YOLO(args.baseline), data, args, "baseline_val")
    (Path(args.output) / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
