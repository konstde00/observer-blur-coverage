"""Turn results/gt_geometry.json into small CSV tables.

Usage: python3 make_tables.py results/gt_geometry.json results/
Standard library only.
"""
import csv
import json
import os
import sys


def pct(x):
    return round(100 * x, 2)


def main(src, out_dir):
    d = json.load(open(src, encoding="utf-8"))
    rows, cams, people = [], [], []
    for key, s in d["summary"].items():
        if key.startswith("paired_"):
            seq, view = "paired 01_tagging 001 to 007", key.split("_", 1)[1]
        else:
            seq, view = key.split("|")
        view = {"ego": "head-worn (Aria, 1408 x 1408)", "exo": "room (GoPro, 3840 x 2160)"}[view]
        nf, ko, u90, wide = s["not_fully_covered"], s["keypoint_outside_circle"], s["under_90pct"], s["wider_than_circle"]
        p10, p50, p90 = s["span_px_p10_p50_p90"]
        rows.append({
            "set": seq, "view": view, "face_instances": s["n"],
            "not_fully_covered_pct": pct(nf[0]), "ci95_low": pct(nf[1]), "ci95_high": pct(nf[2]),
            "keypoint_outside_circle_pct": pct(ko[0]), "under_90pct_covered_pct": pct(u90[0]),
            "face_wider_than_circle_pct": pct(wide[0]),
            "face_span_px_p10": p10, "face_span_px_p50": p50, "face_span_px_p90": p90,
        })
        for c, v in s["per_camera_not_fully"].items():
            cams.append({"set": seq, "view": view, "camera": c, "not_fully_covered_pct": pct(v)})
        for p, v in s["per_person_not_fully"].items():
            people.append({"set": seq, "view": view, "person": p, "not_fully_covered_pct": pct(v)})
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("summary.csv", rows), ("per_camera.csv", cams), ("per_person.csv", people)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    lo, hi = d["paired_diff_not_fully_ego_minus_exo_camera_bootstrap_95"]
    with open(os.path.join(out_dir, "bootstrap_diff.txt"), "w", encoding="utf-8") as f:
        f.write("Paired tagging 001 to 007, share of faces not fully covered by the 100 px circle.\n")
        f.write("Head-worn minus room, 95% interval from 2000 bootstrap resamples over cameras (seed 1):\n")
        f.write(f"{pct(lo)} to {pct(hi)} percentage points\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
