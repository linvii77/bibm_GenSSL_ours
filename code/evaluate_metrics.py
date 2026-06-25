"""Evaluate test set with DSC, ASD, HD95, NSD and export CSV/Markdown tables."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from medpy import metric
from tqdm import tqdm

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
from utils import read_list, read_nifti, config

SYNAPSE_ORGANS = [
    "spleen", "rkid", "lkid", "gall", "eso", "liver", "sto", "aorta",
    "ivc", "vein", "pancreas", "rad", "lad",
]


def pad_label(label, patch_size):
    if label.shape[0] < patch_size[0] or label.shape[1] < patch_size[1] or label.shape[2] < patch_size[2]:
        pw = max((patch_size[0] - label.shape[0]) // 2 + 1, 0)
        ph = max((patch_size[1] - label.shape[1]) // 2 + 1, 0)
        pd = max((patch_size[2] - label.shape[2]) // 2 + 1, 0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode="constant", constant_values=0)
    return label


def cal_metrics(pred_i, label_i, spacing=(1.0, 1.0, 1.0)):
    if pred_i.sum() > 0 and label_i.sum() > 0:
        dsc = metric.binary.dc(pred_i, label_i) * 100.0
        hd95 = metric.binary.hd95(pred_i, label_i)
        asd = metric.binary.asd(pred_i, label_i)
        sf = compute_surface_distances(label_i.astype(bool), pred_i.astype(bool), spacing_mm=spacing)
        nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.0) * 100.0
    elif pred_i.sum() > 0 and label_i.sum() == 0:
        dsc, hd95, asd, nsd = 0.0, 128.0, 128.0, 0.0
    elif pred_i.sum() == 0 and label_i.sum() > 0:
        dsc, hd95, asd, nsd = 0.0, 128.0, 128.0, 0.0
    else:
        dsc, hd95, asd, nsd = 100.0, 0.0, 0.0, 100.0
    return dsc, asd, hd95, nsd


def evaluate(args):
    cfg = config.Config(args.task)
    ids_list = read_list(args.split, task=args.task)
    test_cls = list(range(1, cfg.num_cls))
    organ_names = SYNAPSE_ORGANS if cfg.num_cls == 14 else [f"class_{i}" for i in test_cls]

    results_all_folds = []
    out_dir = Path("logs") / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, args.folds + 1):
        values = np.zeros((len(ids_list), len(test_cls), 4), dtype=np.float64)
        for idx, data_id in enumerate(tqdm(ids_list, desc=f"fold{fold}")):
            pred = read_nifti(str(out_dir / f"fold{fold}" / "predictions" / f"{data_id}.nii.gz"))
            label = np.load(os.path.join(cfg.save_dir, "npy", f"{data_id}_label.npy"))
            label = pad_label(label, cfg.patch_size)
            for i in test_cls:
                values[idx, i - 1] = cal_metrics(pred == i, label == i)
        results_all_folds.append(values)

    results_all_folds = np.array(results_all_folds)
    per_class_mean = results_all_folds.mean(axis=0).mean(axis=0)
    per_class_std = results_all_folds.mean(axis=0).std(axis=0)
    overall_mean = per_class_mean.mean(axis=0)
    fold_means = results_all_folds.mean(axis=1).mean(axis=1)
    overall_std = fold_means.std(axis=0)

    txt_path = out_dir / "evaluation_metrics.txt"
    csv_path = out_dir / "evaluation_metrics.csv"
    md_path = out_dir / "evaluation_metrics.md"

    rows = []
    for i, name in enumerate(organ_names):
        rows.append({
            "organ": name,
            "DSC": per_class_mean[i, 0],
            "DSC_std": per_class_std[i, 0],
            "ASD": per_class_mean[i, 1],
            "ASD_std": per_class_std[i, 1],
            "HD95": per_class_mean[i, 2],
            "HD95_std": per_class_std[i, 2],
            "NSD": per_class_mean[i, 3],
            "NSD_std": per_class_std[i, 3],
        })
    rows.append({
        "organ": "Average",
        "DSC": overall_mean[0],
        "DSC_std": overall_std[0],
        "ASD": overall_mean[1],
        "ASD_std": overall_std[1],
        "HD95": overall_mean[2],
        "HD95_std": overall_std[2],
        "NSD": overall_mean[3],
        "NSD_std": overall_std[3],
    })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w") as f:
        f.write("| Organ | DSC (%) | ASD (mm) | HD95 (mm) | NSD (%) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            if r["organ"] == "Average":
                f.write(
                    f"| **{r['organ']}** | **{r['DSC']:.2f}±{r['DSC_std']:.2f}** | "
                    f"**{r['ASD']:.2f}±{r['ASD_std']:.2f}** | "
                    f"**{r['HD95']:.2f}±{r['HD95_std']:.2f}** | "
                    f"**{r['NSD']:.2f}±{r['NSD_std']:.2f}** |\n"
                )
            else:
                f.write(
                    f"| {r['organ']} | {r['DSC']:.2f}±{r['DSC_std']:.2f} | "
                    f"{r['ASD']:.2f}±{r['ASD_std']:.2f} | "
                    f"{r['HD95']:.2f}±{r['HD95_std']:.2f} | "
                    f"{r['NSD']:.2f}±{r['NSD_std']:.2f} |\n"
                )

    with open(txt_path, "w") as f:
        f.write(f"Task: {args.task}\nExp: {args.exp}\nFolds: {args.folds}\n\n")
        for r in rows:
            f.write(
                f"{r['organ']}: DSC={r['DSC']:.2f}±{r['DSC_std']:.2f}, "
                f"ASD={r['ASD']:.2f}±{r['ASD_std']:.2f}, "
                f"HD95={r['HD95']:.2f}±{r['HD95_std']:.2f}, "
                f"NSD={r['NSD']:.2f}±{r['NSD_std']:.2f}\n"
            )

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(open(md_path).read())
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="synapse_0.2")
    parser.add_argument("--exp", default="Exp_IBSSL_Synapse_0.2/diffusion")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--split", default="test")
    evaluate(parser.parse_args())
