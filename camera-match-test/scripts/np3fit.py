#!/usr/bin/env python3
"""Fit a Nikon Flexible Color NP3 so a NEF rendered through NX Studio matches a reference image of the same chart(s).

Subcommands (all take a config JSON; see SKILL.md):
  preview  render each NEF with the start NP3 and write ruler previews of NEFs and targets, for reading off chart corners
  grid     refine chart corners, write overlays to check, save the refined corners into a new config file
  score    score the start NP3 (or --np3 FILE), and any camera JPEGs, against the targets
  fit      fit a new NP3 from the start NP3 and write it, versioned, never overwriting
"""
import argparse, concurrent.futures as cf, json, os, sys, time
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np3lib as L

RENDER_W = 3024  # half size: same render time, faster measuring, clean patch sampling
DEFAULT_CHARTS = {"cmp": {"rows": L.ROWS, "cols": L.COLS, "fr": 0.5}}


def _resolve_path(val, cfg_dir):
    if not val or os.path.isabs(val) or os.path.exists(val):
        return val
    for base in (cfg_dir, os.path.dirname(cfg_dir)):
        cand = os.path.abspath(os.path.join(base, val))
        if os.path.exists(cand):
            return cand
    return val


def load_config(path):
    cfg = json.load(open(path))
    cfg["_path"] = path
    cfg_dir = os.path.dirname(os.path.abspath(path))
    cfg["start_np3"] = _resolve_path(cfg["start_np3"], cfg_dir)
    os.makedirs(cfg["work_dir"], exist_ok=True)
    for p in cfg["pairs"]:
        for k in ("nef", "target", "camera_jpeg"):
            if p.get(k):
                p[k] = _resolve_path(p[k], cfg_dir)
        for k in ("nef", "target"):
            if not os.path.exists(p[k]):
                sys.exit(f"pair {p['id']}: {k} not found: {p[k]}")
    return cfg


def chart_defs(cfg):
    return cfg.get("charts") or DEFAULT_CHARTS


def pair_charts(p, role_key, defs):
    """Return {chart_name: corners} for 'nef' or 'target', supporting both multi-chart and legacy single-CMP configs."""
    multi = p.get(f"{role_key}_charts")
    if multi:
        return {k: np.asarray(v, float) for k, v in multi.items() if k in defs}
    legacy = p.get(f"{role_key}_corners")
    if legacy is not None:
        first = next(iter(defs))
        return {first: np.asarray(legacy, float)}
    sys.exit(f"pair {p['id']}: missing {role_key}_charts or {role_key}_corners")


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
    print("Read the corner patch centres (full-resolution pixels) off each preview and put them in the config.")


def cmd_grid(cfg, a):
    tpl = L.Template(cfg["start_np3"])
    defs = chart_defs(cfg)
    d = os.path.join(cfg["work_dir"], "overlays")
    os.makedirs(d, exist_ok=True)
    for p in cfg["pairs"]:
        img = L.render(p["nef"], tpl.base, 6048, cfg["work_dir"])
        nc = pair_charts(p, "nef", defs)
        n_over, n_costs = [], {}
        for cname, guess in nc.items():
            cd = defs[cname]
            c, cost = L.locate(img, guess, cd["rows"], cd["cols"], cd.get("fr", 0.5))
            nc[cname] = [round(float(v), 1) for v in c]
            n_costs[cname] = round(cost, 4)
            n_over.append((c, cd["rows"], cd["cols"], cd.get("fr", 0.5)))
        if "nef_charts" in p:
            p["nef_charts"] = nc
        else:
            p["nef_corners"] = next(iter(nc.values()))
        L.overlay(img, n_over, None, f"{d}/{p['id']}_nef.jpg")

        t, _ = L.load_reference(p["target"], cfg["work_dir"])
        tc = pair_charts(p, "target", defs)
        t_over, t_costs = [], {}
        for cname, guess in tc.items():
            cd = defs[cname]
            ct, cost_t = L.locate(t, guess, cd["rows"], cd["cols"], cd.get("fr", 0.5))
            tc[cname] = [round(float(v), 1) for v in ct]
            t_costs[cname] = round(cost_t, 4)
            t_over.append((ct, cd["rows"], cd["cols"], cd.get("fr", 0.5)))
        if "target_charts" in p:
            p["target_charts"] = tc
        else:
            p["target_corners"] = next(iter(tc.values()))
        L.overlay(t, t_over, None, f"{d}/{p['id']}_target.jpg")

        if p.get("camera_jpeg"):
            j, _ = L.load_reference(p["camera_jpeg"], cfg["work_dir"])
            first_c = next(iter(nc.values()))
            first_cd = defs[next(iter(nc))]
            cj, _ = L.locate(j, first_c, first_cd["rows"], first_cd["cols"], first_cd.get("fr", 0.5))
            p["camera_jpeg_corners"] = [round(float(v), 1) for v in cj]
        print(f"{p['id']}: grid fit nef {n_costs}, target {t_costs} (lower is better)")
    out = new_path(os.path.splitext(cfg["_path"])[0] + "_grid.json")
    json.dump({k: v for k, v in cfg.items() if not k.startswith("_")}, open(out, "w"), indent=1)
    print("overlays in", d, "- every box should sit inside one patch. Refined config:", out)


def _chart_wb_gains(m_dict, wb_src):
    """Compute linear R/B white-balance gains from neutral patches, preferring ColorChecker Classic greys if present."""
    if wb_src == "cc24" or (wb_src is None and "cc24" in m_dict):
        lin = L._lin(m_dict["cc24"][1:5, 3])
        return np.array([np.median(lin[:, 1] / lin[:, 0]), 1.0, np.median(lin[:, 1] / lin[:, 2])])
    key = wb_src if wb_src in m_dict else next(iter(m_dict))
    return L.wb_gains(m_dict[key])


def _skin_locus(lab_arr):
    """Boolean mask for patches inside the natural human skin-tone Lab locus."""
    Ls, a, b = lab_arr[..., 0], lab_arr[..., 1], lab_arr[..., 2]
    ch = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    return (Ls >= 20) & (Ls <= 85) & (a >= 4) & (a <= 28) & (b >= 6) & (b <= 35) & (ch >= 8) & (ch <= 40) & (hue >= 15) & (hue <= 68)


class Data:
    """Measured targets across all configured charts for every pair, plus rendering and scoring helpers."""

    def __init__(self, cfg, tpl):
        self.cfg, self.tpl = cfg, tpl
        self.defs = chart_defs(cfg)
        self.pairs = {p["id"]: p for p in cfg["pairs"]}
        self.train = [p["id"] for p in cfg["pairs"] if p.get("role", "train") == "train"]
        self.test = [p["id"] for p in cfg["pairs"] if p.get("role") == "test"]
        self.d = {}
        half = RENDER_W / 6048
        neutral_np3 = tpl.neutral_base()
        wb_src = cfg.get("wb_chart")
        weighting = cfg.get("weighting", "skin_magic" if "cc24" in self.defs else "everyday")

        for pid, p in self.pairs.items():
            # Neutral render removes template color grading/curve bias from WB & exposure estimation
            neut_img = L.render(p["nef"], neutral_np3, RENDER_W, cfg["work_dir"])
            t_img, desc = L.load_reference(p["target"], cfg["work_dir"])
            nc = pair_charts(p, "nef", self.defs)
            tc = pair_charts(p, "target", self.defs)
            common = [k for k in self.defs if k in nc and k in tc]

            m0_dict, mt_dict = {}, {}
            for cname in common:
                cd = self.defs[cname]
                m0_dict[cname] = L.measure(neut_img, nc[cname] * half, cd["rows"], cd["cols"], cd.get("fr", 0.5))
                mt_dict[cname] = L.measure(t_img, tc[cname], cd["rows"], cd["cols"], cd.get("fr", 0.5))

            gains = np.ones(3)
            mode = wb_mode(cfg)
            if mode:
                gains = _chart_wb_gains(m0_dict, wb_src)
                note = f"Nikon neutral render {L.describe_gains(gains)}"
                if mode == "both":
                    tg = _chart_wb_gains(mt_dict, wb_src)
                    mt_dict = {k: L.apply_gains(v, tg) for k, v in mt_dict.items()}
                    note += f"; target {L.describe_gains(tg)}"
                else:
                    note += "; target left as shot"
                log(cfg, f"{pid}: white balance neutralised ({mode}) - {note}")

            if cfg.get("match_exposure") or p.get("match_exposure"):
                # Match mid-grey linear luminance per pair (useful for exposure-mismatched brackets)
                m0_all = np.concatenate([m0_dict[k].reshape(-1, 3) for k in common])
                mt_all = np.concatenate([mt_dict[k].reshape(-1, 3) for k in common])
                tl_all = L.lab(mt_all)
                n = (np.hypot(tl_all[:, 1], tl_all[:, 2]) < 5) & (tl_all[:, 0] > 30) & (tl_all[:, 0] < 70)
                if n.sum() >= 3:
                    gain = np.median(L._lin(L.apply_gains(m0_all, gains)[n]).mean(-1)) / np.median(L._lin(mt_all[n]).mean(-1))
                    mt_dict = {k: np.clip(L._enc(L._lin(v) * gain), 0, 1) for k, v in mt_dict.items()}
                    log(cfg, f"{pid}: target exposure scaled by {np.log2(gain):+.2f} EV")

            # Assemble flat patch arrays + per-patch metadata across all charts
            neut_rgb, targ_rgb, valid_l, skin_l, neut_flag_l, chart_l = [], [], [], [], [], []
            for cname in common:
                m0 = L.apply_gains(m0_dict[cname], gains)
                mt = mt_dict[cname]
                tl = L.lab(mt)
                rows, cols = mt.shape[:2]
                v_mask = (mt.max(-1) < .995) & (mt.min(-1) > .002) & (m0.max(-1) < .992) & (m0.min(-1) > .004)
                sk = _skin_locus(tl)
                if cname == "cc24" and rows == 6 and cols == 4:
                    sk[4, 0] = True  # ColorChecker Light Skin
                    sk[5, 0] = True  # ColorChecker Dark Skin
                nf = np.hypot(tl[..., 1], tl[..., 2]) < 6
                if cname == "cc24" and rows == 6 and cols == 4:
                    nf[:, 3] = True
                elif cname == "tiffen_g":
                    nf[:] = True

                neut_rgb.append(m0[v_mask])
                targ_rgb.append(mt[v_mask])
                valid_l.append((cname, nc[cname] * half, v_mask))
                skin_l.append(sk[v_mask])
                neut_flag_l.append(nf[v_mask])
                chart_l.append(np.full(int(v_mask.sum()), cname, dtype=object))

            neut_rgb = np.concatenate(neut_rgb)
            targ_rgb = np.concatenate(targ_rgb)
            tl = L.lab(targ_rgb)
            ch = np.hypot(tl[:, 1], tl[:, 2])
            skin = np.concatenate(skin_l)
            is_neut = np.concatenate(neut_flag_l)
            charts_arr = np.concatenate(chart_l)

            if weighting == "uniform":
                w = np.ones(len(ch))
            elif weighting == "everyday":
                w = np.where(ch < 45, 1.0, np.where(ch < 60, .6, .3))
            else:  # "skin_magic": prioritize skin tones, neutrals/split-tone, and everyday colors
                w = np.where(ch < 45, 1.2, np.where(ch < 60, .5, .25))
                w = np.where(is_neut, np.maximum(w, 2.5), w)
                w = np.where(skin, 3.0, w)
                if "cc24" in common:
                    cc_skin = (charts_arr == "cc24") & skin
                    w = np.where(cc_skin, 4.5, w)

            self.d[pid] = dict(
                nef=p["nef"], charts=valid_l, neut_rgb=neut_rgb, target=tl, target_srgb=targ_rgb,
                chroma=ch, skin=skin, is_neut=is_neut, chart_names=charts_arr, weight=w, desc=desc, gains=gains,
            )
            log(cfg, f"{pid} ({p.get('role', 'train')}): {len(tl)} usable patches ({int(skin.sum())} skin, {int(is_neut.sum())} neutral); target {desc}")
        if not self.train:
            sys.exit("no pairs with role train")
        self.W = np.concatenate([self.d[s]["weight"] for s in self.train])

    def _measure_pair(self, pid, np3):
        info = self.d[pid]
        img = L.render(info["nef"], np3, RENDER_W, self.cfg["work_dir"])
        chunks = []
        for cname, corners, v_mask in info["charts"]:
            cd = self.defs[cname]
            m = L.measure(img, corners, cd["rows"], cd["cols"], cd.get("fr", 0.5))
            chunks.append(L.apply_gains(m, info["gains"])[v_mask])
        return L.lab(np.concatenate(chunks))

    def labs(self, np3, ids):
        return {s: self._measure_pair(s, np3) for s in ids}

    def vec(self, l):
        return np.concatenate([L.de00_vec(l[s], self.d[s]["target"]) for s in self.train])

    def score(self, l):
        d = np.concatenate([L.de00(l[s], self.d[s]["target"]) for s in self.train])
        return float((d * self.W).sum() / self.W.sum())

    def report(self, l, tag):
        rows = []
        for s in self.train + self.test:
            info = self.d[s]
            t, ch = info["target"], info["chroma"]
            d0, d7 = L.de00(l[s], t), L.de76(l[s], t)
            ev = ch < 45
            nl = info["is_neut"]
            sk = info["skin"]
            msg = (
                f"{tag} {s} ({self.pairs[s].get('role', 'train')}): ΔE2000 mean {d0.mean():.2f} median {np.median(d0):.2f} "
                f"p95 {np.percentile(d0, 95):.2f} max {d0.max():.2f} | everyday {d0[ev].mean():.2f} "
                f"| skin {d0[sk].mean() if sk.any() else float('nan'):.2f} "
                f"| neutrals {d0[nl].mean() if nl.any() else float('nan'):.2f} | ΔE76 mean {d7.mean():.2f} "
                f"| lightness offset (target-render) {(t[:, 0] - l[s][:, 0]).mean():+.2f}"
            )
            rows.append(msg)
        for r in rows:
            log(self.cfg, r)
        return rows


def fit_initial_curve(data):
    """Analytically fit a smooth 1D tone curve from neutral patches across all training pairs."""
    tpl = data.tpl
    xs, ys = [], []
    for s in data.train:
        info = data.d[s]
        # Neutrals in both render and target
        nl_r = L.lab(info["neut_rgb"])
        m = (np.hypot(nl_r[:, 1], nl_r[:, 2]) < 8) & (info["chroma"] < 8)
        if m.sum() >= 3:
            xs.append(info["neut_rgb"][m].mean(-1))
            ys.append(info["target_srgb"][m].mean(-1))
    if not xs:
        return tpl.start(reset_colour=True)
    x_all = np.concatenate(xs).astype(float)
    y_all = np.concatenate(ys).astype(float)
    knots = L.KNOTS

    def loss(vals):
        pred = PchipInterpolator(knots, vals)(x_all)
        d2 = np.diff(vals, 2)
        return float(np.mean((pred - y_all) ** 2) + 0.02 * np.mean(d2 ** 2))

    cons = [{"type": "ineq", "fun": lambda v, i=i: v[i + 1] - v[i] - 0.002} for i in range(len(knots) - 1)]
    bnds = [(0.0, 20 / 255)] + [(0.0, 1.0)] * (len(knots) - 2) + [(1.0 - 20 / 255, 1.0)]
    res = minimize(loss, knots.copy(), method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    lut = np.maximum.accumulate(np.clip(PchipInterpolator(knots, res.x)(np.arange(257) / 256), 0, 1))
    x0 = tpl.start(reset_colour=True)
    cd = tpl.curve_params_for_lut(lut)
    for i, k in enumerate(knots):
        idx = tpl.names.index(f"curve_{k:.2f}")
        x0[idx] = cd[i]
    return np.clip(x0, tpl.lo, tpl.hi)


def cmd_score(cfg, a):
    tpl = L.Template(a.np3 or cfg["start_np3"])
    data = Data(cfg, tpl)
    data.report(data.labs(tpl.base, data.train + data.test), f"NX render of '{L.np3_name(tpl.base)}'")
    for p in cfg["pairs"]:
        if p.get("camera_jpeg"):
            j, _ = L.load_reference(p["camera_jpeg"], cfg["work_dir"])
            cj = np.array(p.get("camera_jpeg_corners") or p["nef_corners"], float)
            first_name = next(iter(data.defs))
            cd = data.defs[first_name]
            mj = L.measure(j, cj, cd["rows"], cd["cols"], cd.get("fr", 0.5))
            if wb_mode(cfg):
                mj = L.apply_gains(mj, L.wb_gains(mj))
            _, _, v_mask = data.d[p["id"]]["charts"][0]
            lj = L.lab(mj)[v_mask]
            # Compare against first chart's target slice
            n_first = int(v_mask.sum())
            d0 = L.de00(lj, data.d[p["id"]]["target"][:n_first])
            log(cfg, f"camera JPEG {p['id']}: ΔE2000 mean {d0.mean():.2f} median {np.median(d0):.2f} p95 {np.percentile(d0, 95):.2f}")


def _fmt_eta(sec):
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


class Progress:
    """Short, human-readable progress tracker leading with overall percentage and live ETA."""

    def __init__(self, cfg, total_evals):
        self.cfg = cfg
        self.total = max(1, total_evals)
        self.done = 0
        self.t0 = time.time()

    def pct(self):
        return min(99, max(1, int(round(5 + 92 * self.done / self.total))))

    def eta(self):
        if self.done < 8:
            return ""
        rem = (time.time() - self.t0) * (self.total - self.done) / max(self.done, 1)
        return f" · ETA ~{_fmt_eta(rem)}"

    def say(self, pct, msg):
        log(self.cfg, f"[{pct:3d}%] {msg}")


def _gn_stage(data, x, iters, label, pool, prog, step_scale=1.0, min_improve=0.002, reg=2e-4):
    tpl, cfg = data.tpl, data.cfg
    l = data.labs(tpl.make(x), data.train)
    best = data.score(l)
    steps = np.where(tpl.step >= 8, np.maximum(np.round(tpl.step * step_scale), 3.0), np.maximum(np.round(tpl.step * step_scale), 2.0))
    prior = tpl.start(reset_colour=True, reset_curve=False)
    curve_idx = [i for i, n in enumerate(tpl.names) if n.startswith("curve_")]
    base_lut_knots = np.interp(L.KNOTS, np.arange(257) / 256, tpl.lut) * 255
    # Second-difference smoothness matrix on the curve knots prevents kinks or flat shadow shelves
    n_k = len(curve_idx)
    D2 = np.zeros((n_k - 2, len(x)))
    for r_i in range(n_k - 2):
        dk1 = L.KNOTS[r_i + 1] - L.KNOTS[r_i]
        dk2 = L.KNOTS[r_i + 2] - L.KNOTS[r_i + 1]
        D2[r_i, curve_idx[r_i]] = 1.0 / dk1
        D2[r_i, curve_idx[r_i + 1]] = -(1.0 / dk1 + 1.0 / dk2)
        D2[r_i, curve_idx[r_i + 2]] = 1.0 / dk2
    smooth_w = 0.008 * np.sqrt(data.W.sum())
    reg_scales = np.array([
        0.0 if n.startswith("curve_") else (3.0 * reg if ("bright" in n and n.startswith("grade_")) or n == "grade_balance" else reg)
        for n in tpl.names
    ]) * np.sqrt(data.W.sum())

    n_params = len(x)
    marks = {max(1, n_params // 3), max(2, (2 * n_params) // 3), n_params}

    for it in range(1, iters + 1):
        t = time.time()
        r = data.vec(l)
        prog.say(prog.pct(), f"{label} {it}/{iters}: starting ΔE2000 {best:.3f}{prog.eta()}")

        def column(i):
            dx = np.zeros_like(x)
            s = steps[i] if x[i] + steps[i] <= tpl.hi[i] else -steps[i]
            if x[i] + s < tpl.lo[i]:
                s = tpl.hi[i] - x[i]
            dx[i] = s
            return i, (data.vec(data.labs(tpl.make(np.clip(x + dx, tpl.lo, tpl.hi)), data.train)) - r).ravel() / s

        cols = [None] * n_params
        futs = [pool.submit(column, i) for i in range(n_params)]
        finished = 0
        for fut in cf.as_completed(futs):
            idx, col = fut.result()
            cols[idx] = col
            finished += 1
            prog.done += 1
            if finished in marks:
                prog.say(prog.pct(), f"{label} {it}/{iters}: testing controls ({finished}/{n_params}){prog.eta()}")

        J = np.stack(cols, axis=1)
        de = np.concatenate([L.de00(l[s], data.d[s]["target"]) for s in data.train])
        wv = np.repeat(np.sqrt(data.W) / np.sqrt(de + .5), 3)
        lam = .05 * np.median(np.linalg.norm(J * wv[:, None], axis=0))
        cur_knots = base_lut_knots + x[curve_idx]
        d2_rhs = -smooth_w * (D2[:, curve_idx] @ cur_knots)
        A = np.vstack([J * wv[:, None], lam * np.eye(n_params), np.diag(reg_scales), smooth_w * D2])
        b = np.r_[-r.ravel() * wv, np.zeros(n_params), -reg_scales * (x - prior), d2_rhs]
        sol = lsq_linear(A, b, bounds=(tpl.lo - x, tpl.hi - x), max_iter=300).x
        improved = False
        for alpha in (1.0, .65, .35, .15):
            trial = np.clip(x + alpha * sol, tpl.lo, tpl.hi)
            # Project curve knots to monotonic values so no flat shelves or inversions occur
            tk = np.maximum.accumulate(base_lut_knots + trial[curve_idx])
            trial[curve_idx] = np.clip(tk - base_lut_knots, tpl.lo[curve_idx], tpl.hi[curve_idx])
            lt = data.labs(tpl.make(trial), data.train)
            st = data.score(lt)
            if st < best - min_improve:
                x, l, best, improved = trial, lt, st, True
                break
        status = f"ΔE2000 {best:.3f}" if improved else f"converged at {best:.3f}"
        prog.say(prog.pct(), f"{label} {it}/{iters} done ({time.time() - t:.0f}s) -> {status}{prog.eta()}")
        if not improved:
            prog.done += (iters - it) * n_params
            break
    return best, x


def gauss_newton(data, x, iters, label, pool, prog):
    best, x = _gn_stage(data, x, iters, f"{label} coarse", pool, prog, step_scale=1.0, min_improve=0.002)
    fine_iters = data.cfg.get("fine_rounds", max(2, iters - 1))
    if fine_iters > 0:
        best, x = _gn_stage(data, x, fine_iters, f"{label} fine", pool, prog, step_scale=0.4, min_improve=0.001)
    return best, x


def cmd_fit(cfg, a):
    t_start = time.time()
    tpl = L.Template(cfg["start_np3"])
    name = cfg["name"]
    tpl.make(tpl.start(), name)
    log(cfg, f"[  1%] Loading {len(cfg['pairs'])} pairs & neutralizing Nikon WB...")
    data = Data(cfg, tpl)
    workers = int(cfg.get("workers", min(8, os.cpu_count() or 4)))
    pool = cf.ThreadPoolExecutor(workers)
    log(cfg, "[  3%] Scoring starting profile...")
    data.report(data.labs(tpl.base, data.train + data.test), "START")

    starts = {}
    if cfg.get("also_fitted_curve", True):
        log(cfg, "[  4%] Fitting initial 1D tone curve from neutral patches...")
        starts["fitted curve"] = fit_initial_curve(data)
    if cfg.get("also_template_start", True):
        starts["as-is"] = tpl.start()
    if cfg.get("also_neutral_start", False):
        starts["neutral colour"] = tpl.start(reset_colour=True)

    rounds = int(cfg.get("rounds", 4))
    fine_iters = int(cfg.get("fine_rounds", max(2, rounds - 1)))
    total_evals = len(starts) * (rounds + fine_iters) * len(tpl.names)
    prog = Progress(cfg, total_evals)
    results = {k: gauss_newton(data, v, rounds, k, pool, prog) for k, v in starts.items()}
    winner = min(results, key=lambda k: results[k][0])
    x = results[winner][1]
    prog.say(98, f"Writing '{name}' & scoring all train/test pairs...")

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
    log(cfg, f"[100%] Done in {_fmt_eta(time.time() - t_start)} -> {np3_path}")


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
