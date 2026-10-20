"""Fallback analysis (no images): how much of each ground-truth face could video_deid's fixed blur circle cover,
assuming PERFECT detection (circle centred on the mean of the 5 GT facial keypoints, radius 100 px as
calculate_bounding_box(single point, margin=50) + apply_circular_blur produce).

Inputs: EgoHumans COCO keypoint JSONs (ego complete, exo truncated by the download quota: parsed up to the
last complete annotation). Output: JSON with per-camera-type, per-sequence and per-person numbers.

Usage: python3 gt_geometry.py person_keypoints_ego_rgb.json person_keypoints_exo.json results/gt_geometry.json
Standard library only.
"""
import json, math, re, sys, collections, random

R = 100  # px, video_deid blur radius for one face point (FACE_MARGIN 50 -> 100 x 100 box -> radius 2 * 50)


def load_truncated(path):
    t = open(path, encoding="utf-8").read()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i = t.index('"annotations"')
    head = json.loads(t[:i].rstrip().rstrip(",") + "}")
    body = t[i:]
    # cut after the last complete annotation object: objects at depth 2 end with "\n        }"
    j = body.rfind("\n        }")
    anns = json.loads("{" + body[: j + len("\n        }")] + "]}")["annotations"]
    head["annotations"] = anns
    head["truncated"] = True
    return head


def face_metrics(kp, w, h):
    f = [(kp[3 * k], kp[3 * k + 1]) for k in range(5)]
    inside = [0 <= x < w and 0 <= y < h for x, y in f]
    if not all(inside[:3]):  # nose and both eyes in frame
        return None
    pv = [p for p, ok in zip(f, inside) if ok]
    xs, ys = [p[0] for p in pv], [p[1] for p in pv]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    # face box (anthropometric, around the keypoint mean near the eyes): ear-to-ear span wide,
    # 0.6 span above (forehead) to 0.8 span below (chin); 9 x 9 grid. Keypoint-only check needs no box.
    pts = [(cx - 0.5 * span + span * a / 8, cy - 0.6 * span + 1.4 * span * b / 8) for a in range(9) for b in range(9)]
    pts = [(x, y) for x, y in pts if 0 <= x < w and 0 <= y < h]
    cov = sum((x - cx) ** 2 + (y - cy) ** 2 <= R * R for x, y in pts) / len(pts)
    kcov = all((x - cx) ** 2 + (y - cy) ** 2 <= R * R for x, y in pv)
    return span, cov, kcov


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - m, c + m


def main(ego_path, exo_path, out_path):
    rows = []
    for kind, path in (("ego", ego_path), ("exo", exo_path)):
        d = load_truncated(path)
        im = {i["id"]: i for i in d["images"]}
        for a in d["annotations"]:
            i = im[a["image_id"]]
            seq = "/".join(i["file_name"].split("/")[:2])
            r = face_metrics(a["keypoints"], i["width"], i["height"])
            if r is None:
                continue
            rows.append(dict(kind=kind, seq=seq, cam=a["camera_name"], person=a["human_name"],
                             frame=a.get("time_stamp"), span=r[0], cov=r[1], kcov=r[2], res=i["width"]))
        print(kind, "annotations", len(d["annotations"]), "truncated" if d.get("truncated") else "complete", file=sys.stderr)
    # restrict to sequences present in both views for the paired comparison
    seqs = {k: {r["seq"] for r in rows if r["kind"] == k} for k in ("ego", "exo")}
    paired = seqs["ego"] & seqs["exo"]
    # the last exo sequence may be cut mid-way: keep only (seq, frame) pairs present in both views
    frames = {k: {(r["seq"], r["frame"]) for r in rows if r["kind"] == k and r["seq"] in paired} for k in ("ego", "exo")}
    both = frames["ego"] & frames["exo"]
    res = dict(radius_px=R, paired_sequences=sorted(paired), summary={})

    def summ(sel):
        n = len(sel); full = sum(r["cov"] >= 0.999 for r in sel); under = sum(r["cov"] < 0.9 for r in sel)
        spans = sorted(r["span"] for r in sel)
        q = lambda p: spans[min(n - 1, int(p * n))] if n else float("nan")
        # cluster spread: share not fully covered per camera
        per_cam = collections.defaultdict(list)
        for r in sel:
            per_cam[r["cam"]].append(r["cov"] < 0.999)
        cams = {c: round(sum(v) / len(v), 3) for c, v in sorted(per_cam.items())}
        per_person = collections.defaultdict(list)
        for r in sel:
            per_person[r["person"]].append(r["cov"] < 0.999)
        return dict(n=n, keypoint_outside_circle=wilson(sum(not r["kcov"] for r in sel), n), not_fully_covered=wilson(n - full, n), under_90pct=wilson(under, n),
                    mean_cov=sum(r["cov"] for r in sel) / n if n else None,
                    span_px_p10_p50_p90=(q(.1), q(.5), q(.9)), wider_than_circle=wilson(sum(r["span"] > 2 * R for r in sel), n),
                    per_camera_not_fully=cams, per_person_not_fully={p: round(sum(v) / len(v), 3) for p, v in sorted(per_person.items())})

    for kind in ("ego", "exo"):
        sel = [r for r in rows if r["kind"] == kind and (r["seq"], r["frame"]) in both]
        res["summary"][f"paired_{kind}"] = summ(sel)
    for seq in sorted({r["seq"] for r in rows}):
        for kind in ("ego", "exo"):
            sel = [r for r in rows if r["kind"] == kind and r["seq"] == seq]
            if sel:
                res["summary"][f"{seq}|{kind}"] = summ(sel)
    # bootstrap over cameras for the paired ego minus exo difference in "not fully covered"
    rnd = random.Random(1)
    by = {k: collections.defaultdict(list) for k in ("ego", "exo")}
    for r in rows:
        if (r["seq"], r["frame"]) in both:
            by[r["kind"]][(r["seq"], r["cam"])].append(r["cov"] < 0.999)
    diffs = []
    for _ in range(2000):
        v = {}
        for k in ("ego", "exo"):
            keys = list(by[k]); s = [rnd.choice(keys) for _ in keys]
            xs = [x for kk in s for x in by[k][kk]]; v[k] = sum(xs) / len(xs)
        diffs.append(v["ego"] - v["exo"])
    diffs.sort()
    res["paired_diff_not_fully_ego_minus_exo_camera_bootstrap_95"] = (diffs[50], diffs[1949])
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_")} for k, v in res["summary"].items()}, indent=1, default=float))
    print("diff CI", res["paired_diff_not_fully_ego_minus_exo_camera_bootstrap_95"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
