#!/usr/bin/env python3
"""Fit a Nikon Flexible Color NP3 so a NEF rendered through NX Studio matches a reference image of the same chart.

Subcommands (all take a config JSON; see SKILL.md):
  preview  render each NEF with the start NP3 and write ruler previews of NEFs and targets, for reading off chart corners
  grid     refine chart corners, write overlays to check, save the refined corners into a new config file
  score    score the start NP3 (or --np3 FILE), and any camera JPEGs, against the targets
  fit      fit a new NP3 from the start NP3 and write it, versioned, never overwriting
"""
import argparse, concurrent.futures as cf, json, os, sys, time
import numpy as np, cv2
from scipy.optimize import lsq_linear

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np3lib as L

RENDER_W = 3024  # half size: same render time, faster measuring, patch medians unchanged


def load_config(path):
    cfg = json.load(open(path))
    cfg["_path"] = path
    os.makedirs(cfg["work_dir"], exist_ok=True)
    for p in cfg["pairs"]:
        for k in ("nef", "target"):
            if not os.path.exists(p[k]):
                sys.exit(f"pair {p['id']}: {k} not found: {p[k]}")
    return cfg


def wb_mode(cfg):
    """neutralize_wb: false (off), "nikon" (balance only the Nikon render), "both" or true (balance both images)."""
    v = cfg.get("neutralize_wb", False)
    if v in (False, None, "off", "none"):
        return None
    if v is True or v == "both":
        return "both"
    if v == "nikon":
        return "nikon"
    sys.exit(f"neutralize_wb must be false, \"nikon\" or \"both\", not {v!r}")


def log(cfg, *a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    open(os.path.join(cfg["work_dir"], "log.txt"), "a").write(s + "\n")


def new_path(path):
    """path, or path_rev2, _rev3 ... whichever does not exist yet."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{root}_rev{n}{ext}"):
        n += 1
    return f"{root}_rev{n}{ext}"


def cmd_preview(cfg, a):
    tpl = L.Template(cfg["start_np3"])
    d = os.path.join(cfg["work_dir"], "previews")
    os.makedirs(d, exist_ok=True)
    for p in cfg["pairs"]:
        img = L.render(p["nef"], tpl.base, 6048, cfg["work_dir"])
        L.preview_with_ruler(img, f"{d}/{p['id']}_nef.jpg")
        t, desc = L.load_reference(p["target"], cfg["work_dir"])
        L.preview_with_ruler(t, f"{d}/{p['id']}_target.jpg")
        print(f"{p['id']}: nef render {img.shape[1]}x{img.shape[0]}, target {t.shape[1]}x{t.shape[0]} ({desc}) -> {d}/{p['id']}_nef.jpg, _target.jpg")
    print("Read the A1, A38, Z1, Z38 patch centres (full-resolution pixels) off each preview and put them in the config as nef_corners / target_corners.")


def cmd_grid(cfg, a):
    tpl = L.Template(cfg["start_np3"])
    d = os.path.join(cfg["work_dir"], "overlays")
    os.makedirs(d, exist_ok=True)
    for p in cfg["pairs"]:
        img = L.render(p["nef"], tpl.base, 6048, cfg["work_dir"])
        c, cost = L.locate(img, p["nef_corners"])
        p["nef_corners"] = [round(float(v), 1) for v in c]
        L.overlay(img, c, None, f"{d}/{p['id']}_nef.jpg")
        t, _ = L.load_reference(p["target"], cfg["work_dir"])
        ct, cost_t = L.locate(t, p["target_corners"])
        p["target_corners"] = [round(float(v), 1) for v in ct]
        L.overlay(t, ct, None, f"{d}/{p['id']}_target.jpg")
        if p.get("camera_jpeg"):
            j, _ = L.load_reference(p["camera_jpeg"], cfg["work_dir"])
            cj, _ = L.locate(j, c)
            p["camera_jpeg_corners"] = [round(float(v), 1) for v in cj]
        print(f"{p['id']}: grid fit nef {cost:.4f}, target {cost_t:.4f} (lower is better; well-aligned charts are about 0.003-0.005)")
    out = new_path(os.path.splitext(cfg["_path"])[0] + "_grid.json")
    json.dump({k: v for k, v in cfg.items() if not k.startswith("_")}, open(out, "w"), indent=1)
    print("overlays in", d, "- every box should sit inside one patch. Refined config:", out)


class Data:
    """Measured targets for every pair, plus rendering and scoring helpers."""

    def __init__(self, cfg, tpl):
        self.cfg, self.tpl = cfg, tpl
        self.pairs = {p["id"]: p for p in cfg["pairs"]}
        self.train = [p["id"] for p in cfg["pairs"] if p.get("role", "train") == "train"]
        self.test = [p["id"] for p in cfg["pairs"] if p.get("role") == "test"]
        self.d = {}
        half = RENDER_W / 6048
        for pid, p in self.pairs.items():
            first = L.render(p["nef"], tpl.base, RENDER_W, cfg["work_dir"])
            cn = np.array(p["nef_corners"], float) * half
            t, desc = L.load_reference(p["target"], cfg["work_dir"])
            mt = L.measure(t, np.array(p["target_corners"], float))
            m0 = L.measure(first, cn)
            valid = (mt.max(-1) < .985) & (mt.min(-1) > .012) & (m0.max(-1) < .985) & (m0.min(-1) > .012)
            gains = np.ones(3)
            mode = wb_mode(cfg)
            if mode:
                # balance the Nikon render (and, in "both" mode, the target) to the chart's grey ring.
                # "nikon" keeps any deliberate warmth in the reference, so the profile learns it as part of the look.
                gains = L.wb_gains(m0)
                note = f"Nikon render {L.describe_gains(gains)}"
                if mode == "both":
                    tg = L.wb_gains(mt)
                    mt = L.apply_gains(mt, tg)
                    note += f"; target {L.describe_gains(tg)}"
                else:
                    note += f"; target left as shot (its grey ring would need {L.describe_gains(L.wb_gains(mt))})"
                log(cfg, f"{pid}: white balance neutralised ({mode}) - {note}")
            tl = L.lab(mt)
            if cfg.get("match_exposure"):
                # scale the target so its mid-grey patches match the NEF render's exposure; keeps the look's tone shape
                n = valid & (np.hypot(tl[..., 1], tl[..., 2]) < 5) & (tl[..., 0] > 30) & (tl[..., 0] < 70)
                lin = lambda v: np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055) ** 2.4)
                gain = np.median(lin(m0[n]).mean(-1)) / np.median(lin(mt[n]).mean(-1))
                enc = lambda v: np.where(v <= .0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - .055)
                mt = np.clip(enc(lin(mt) * gain), 0, 1)
                tl = L.lab(mt)
                log(cfg, f"{pid}: target exposure scaled by {np.log2(gain):+.2f} EV")
            ch = np.hypot(tl[..., 1], tl[..., 2])[valid]
            w = np.where(ch < 45, 1.0, np.where(ch < 60, .6, .3)) if cfg.get("weighting", "everyday") == "everyday" else np.ones(len(ch))
            self.d[pid] = dict(nef=p["nef"], corners=cn, valid=valid, target=tl[valid], chroma=ch, weight=w, desc=desc, gains=gains)
            log(cfg, f"{pid} ({p.get('role', 'train')}): {int(valid.sum())} usable patches; target {desc}")
        if not self.train:
            sys.exit("no pairs with role train")
        self.W = np.concatenate([self.d[s]["weight"] for s in self.train])

    def labs(self, np3, ids):
        # the Nikon render keeps one fixed correction per shot, so profile tints can't stand in for white balance
        return {s: L.lab(L.apply_gains(L.measure(L.render(self.d[s]["nef"], np3, RENDER_W, self.cfg["work_dir"]), self.d[s]["corners"]), self.d[s]["gains"]))[self.d[s]["valid"]]
                for s in ids}

    def vec(self, l):
        return np.concatenate([np.stack(L.de00_parts(l[s], self.d[s]["target"])[:3], -1) for s in self.train])

    def score(self, l):
        d = np.concatenate([L.de00(l[s], self.d[s]["target"]) for s in self.train])
        return float((d * self.W).sum() / self.W.sum())

    def report(self, l, tag):
        rows = []
        for s in self.train + self.test:
            t, ch = self.d[s]["target"], self.d[s]["chroma"]
            d0, d7 = L.de00(l[s], t), L.de76(l[s], t)
            ev = ch < 45
            nl = ch < 6
            rows.append(f"{tag} {s} ({self.pairs[s].get('role', 'train')}): ΔE2000 mean {d0.mean():.2f} median {np.median(d0):.2f} p95 {np.percentile(d0, 95):.2f} max {d0.max():.2f}"
                        f" | everyday {d0[ev].mean():.2f} | neutrals {d0[nl].mean() if nl.any() else float('nan'):.2f} | ΔE76 mean {d7.mean():.2f}"
                        f" | lightness offset (target-render) {(t[:, 0] - l[s][:, 0]).mean():+.2f}")
        for r in rows:
            log(self.cfg, r)
        return rows


def cmd_score(cfg, a):
    tpl = L.Template(a.np3 or cfg["start_np3"])
    data = Data(cfg, tpl)
    data.report(data.labs(tpl.base, data.train + data.test), f"NX render of '{L.np3_name(tpl.base)}'")
    for p in cfg["pairs"]:
        if p.get("camera_jpeg"):
            j, _ = L.load_reference(p["camera_jpeg"], cfg["work_dir"])
            cj = np.array(p.get("camera_jpeg_corners") or p["nef_corners"], float)
            mj = L.measure(j, cj)
            if wb_mode(cfg):
                mj = L.apply_gains(mj, L.wb_gains(mj))
            lj = L.lab(mj)[data.d[p["id"]]["valid"]]
            d0 = L.de00(lj, data.d[p["id"]]["target"])
            log(cfg, f"camera JPEG {p['id']}: ΔE2000 mean {d0.mean():.2f} median {np.median(d0):.2f} p95 {np.percentile(d0, 95):.2f}")


def gauss_newton(data, x, iters, label, pool):
    tpl, cfg = data.tpl, data.cfg
    l = data.labs(tpl.make(x), data.train)
    best = data.score(l)
    log(cfg, f"start {label}: weighted ΔE2000 {best:.3f}")
    for it in range(1, iters + 1):
        t = time.time()
        r = data.vec(l)

        def column(i):
            dx = np.zeros_like(x)
            step = tpl.step[i] if x[i] + tpl.step[i] <= tpl.hi[i] else -tpl.step[i]
            dx[i] = step
            return (data.vec(data.labs(tpl.make(np.clip(x + dx, tpl.lo, tpl.hi)), data.train)) - r).ravel() / step

        J = np.stack(list(pool.map(column, range(len(x)))), axis=1)
        de = np.concatenate([L.de00(l[s], data.d[s]["target"]) for s in data.train])
        wv = np.repeat(np.sqrt(data.W) / np.sqrt(de + .5), 3)  # leans toward the average rather than the few worst patches
        lam = .05 * np.median(np.linalg.norm(J * wv[:, None], axis=0))
        A = np.vstack([J * wv[:, None], lam * np.eye(len(x))])
        b = np.r_[-r.ravel() * wv, np.zeros(len(x))]
        sol = lsq_linear(A, b, bounds=(tpl.lo - x, tpl.hi - x), max_iter=200).x
        improved = False
        for alpha in (1.0, .6, .3):
            trial = np.clip(x + alpha * sol, tpl.lo, tpl.hi)
            lt = data.labs(tpl.make(trial), data.train)
            st = data.score(lt)
            log(cfg, f"{label} round {it} step {alpha}: weighted ΔE2000 {st:.3f}")
            if st < best - .005:
                x, l, best, improved = trial, lt, st, True
                break
        log(cfg, f"{label} round {it}: best {best:.3f} ({time.time() - t:.0f}s)")
        if not improved:
            break
    return best, x


def cmd_fit(cfg, a):
    tpl = L.Template(cfg["start_np3"])
    name = cfg["name"]
    tpl.make(tpl.start(), name)  # validates the name early
    data = Data(cfg, tpl)
    pool = cf.ThreadPoolExecutor(3)
    data.report(data.labs(tpl.base, data.train + data.test), "START")
    starts = {"as-is": tpl.start()}
    if cfg.get("also_neutral_start", True):
        starts["neutral colour"] = tpl.start(reset_colour=True)
    results = {k: gauss_newton(data, v, cfg.get("rounds", 3), k, pool) for k, v in starts.items()}
    winner = min(results, key=lambda k: results[k][0])
    x = results[winner][1]
    log(cfg, "winner (chosen on fitted pairs only):", winner, {k: round(v[0], 3) for k, v in results.items()})
    out = tpl.make(x, name)
    slot = int(cfg.get("slot", 1))
    d = new_path(os.path.join(cfg["output_dir"], name))
    os.makedirs(os.path.join(d, "NIKON", "CUSTOMPC"))
    np3_path = os.path.join(d, "NIKON", "CUSTOMPC", f"PICCON{slot:02d}.NP3")
    with open(np3_path, "xb") as f:
        f.write(out)
    rows = data.report(data.labs(out, data.train + data.test), "FINAL")
    limits = [n for n, v, lo, hi in zip(tpl.names, x, tpl.lo, tpl.hi) if v <= lo + .5 or v >= hi - .5]
    with open(os.path.join(d, "settings.json"), "x") as f:
        json.dump(dict(name=name, start_np3=cfg["start_np3"], start=winner, pairs=cfg["pairs"], at_limits=limits,
                       settings={k: round(float(v), 2) for k, v in zip(tpl.names, x)}, results=rows), f, indent=1)
    log(cfg, "controls at their limits:", limits)
    log(cfg, "wrote", np3_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["preview", "grid", "score", "fit"])
    ap.add_argument("config")
    ap.add_argument("--np3", help="score this NP3 instead of the config's start_np3")
    a = ap.parse_args()
    if not os.path.exists(L.SAVEPROCESS):
        sys.exit("NX Studio's SaveProcess not found at " + L.SAVEPROCESS)
    cfg = load_config(a.config)
    dict(preview=cmd_preview, grid=cmd_grid, score=cmd_score, fit=cmd_fit)[a.command](cfg, a)


if __name__ == "__main__":
    main()
