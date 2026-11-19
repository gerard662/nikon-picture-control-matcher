#!/usr/bin/env python3
"""Simple one-command runner for cloning any Fuji film simulation folder (files/fuji_<sim>/) to a Nikon .NP3."""
import argparse, json, os, re, shutil, struct, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")


def _ensure_venv():
    try:
        import numpy, scipy, cv2, tifffile  # noqa: F401
    except ImportError:
        if os.path.exists(VENV_PY) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PY):
            os.execv(VENV_PY, [VENV_PY] + sys.argv)
        sys.exit("Missing Python dependencies (numpy, scipy, opencv-python, tifffile) and .venv not found.")


_ensure_venv()
import cv2  # noqa: E402

REF_DIR = os.path.join(ROOT, "reference")
NP3FIT = os.path.join(ROOT, "camera-match-test", "scripts", "np3fit.py")
TEMPLATE_NP3 = os.path.join(ROOT, "camera-match-test", "assets", "flexible_color_template.NP3")
CORNERS_FILE = os.path.join(REF_DIR, "chart_corners.json")

PAIR_SPECS = [
    ("1_80_m1EV", "NZF_7428.NEF", "DSCF0067", "train", "fuji65", False, 80.0),
    ("1_40_0EV",  "NZF_7427.NEF", "DSCF0065", "train", "fuji65", False, 40.0),
    ("1_30_p033", "NZF_7431.NEF", "DSCF0064", "train", "fuji65", False, 30.0),
    ("1_50_m033", "NZF_7430.NEF", "DSCF0066", "test",  "fuji65", False, 51.0),
    ("1_60_m067", "NZF_7429.NEF", "DSCF0080", "test",  "fuji80", False, 60.0),
    ("1_20_sub",  "NZF_7433.NEF", "DSCF0064_+0.2EV", "test", "fuji65", True, None),
]
IMG_EXTS = (".jpg", ".jpeg", ".hif", ".heif", ".tif", ".tiff")


def available_sims():
    files_dir = os.path.join(ROOT, "files")
    if not os.path.isdir(files_dir):
        return []
    return sorted(d[5:] for d in os.listdir(files_dir) if d.startswith("fuji_") and os.path.isdir(os.path.join(files_dir, d)))


def resolve_fuji_dir(arg):
    clean = arg.rstrip("/\\")
    if os.path.basename(clean).startswith("fuji_"):
        sim = os.path.basename(clean)[5:]
    else:
        sim = os.path.basename(clean)
    folder = os.path.join(ROOT, "files", f"fuji_{sim}")
    if not os.path.isdir(folder):
        sims = available_sims()
        avail = ", ".join(sims) if sims else "(none found in files/)"
        sys.exit(f"Folder not found: {folder}\nAvailable Fuji folders in files/: {avail}")
    return sim, folder


def read_exif_shutter(path):
    """Best-effort shutter speed (1/t) from JPEG/TIFF EXIF without external tools."""
    try:
        data = open(path, "rb").read(262144)
        if data[:2] != b"\xff\xd8":
            return None
        pos = 2
        while pos + 4 < len(data):
            m, ln = struct.unpack_from(">HH", data, pos)
            if m == 0xFFE1 and data[pos + 4:pos + 10] == b"Exif\0\0":
                base = pos + 10
                e = "<" if data[base:base + 2] == b"II" else ">"

                def ifd(off):
                    n = struct.unpack_from(e + "H", data, base + off)[0]
                    out = {}
                    for i in range(n):
                        p = base + off + 2 + i * 12
                        tag, typ, cnt = struct.unpack_from(e + "HHI", data, p)
                        vb = data[p + 8:p + 12]
                        sz = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}.get(typ, 1) * cnt
                        raw = vb[:sz] if sz <= 4 else data[base + struct.unpack_from(e + "I", vb, 0)[0]:][:sz]
                        out[tag] = raw
                    return out

                t0 = ifd(struct.unpack_from(e + "I", data, base + 4)[0])
                tex = ifd(struct.unpack_from(e + "I", t0[0x8769], 0)[0])
                n, d = struct.unpack_from(e + "II", tex[0x829A], 0)
                return d / n if n else None
            pos += 2 + ln
    except Exception:
        return None
    return None


def find_fuji_files(folder):
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(IMG_EXTS) and not f.startswith(".")]
    by_stem = {os.path.splitext(os.path.basename(f))[0].lower(): f for f in files}
    matched = {}
    used = set()
    # Pass 1: exact stem match (e.g. DSCF0065, DSCF0064_+0.2EV)
    for pid, _, stem, *_ in PAIR_SPECS:
        f = by_stem.get(stem.lower())
        if f:
            matched[pid] = f
            used.add(f)
    # Pass 2: fallback by EXIF shutter speed for renamed exports
    remaining = [(f, read_exif_shutter(f)) for f in files if f not in used]
    for pid, _, _, _, _, _, inv_t in PAIR_SPECS:
        if pid in matched or inv_t is None:
            continue
        for f, s in remaining:
            if f not in used and s and abs(s - inv_t) <= 3.0 and "+0.2" not in os.path.basename(f):
                matched[pid] = f
                used.add(f)
                break
    return matched


def scale_charts(charts, target_path):
    img = cv2.imread(target_path)
    if img is None:
        return charts
    h, w = img.shape[:2]
    if (w, h) == (6048, 4032):
        return charts
    sx, sy = w / 6048.0, h / 4032.0
    out = {}
    for k, pts in charts.items():
        out[k] = [round(v * (sx if i % 2 == 0 else sy), 1) for i, v in enumerate(pts)]
    return out


def next_profile_name(sim, out_dir):
    base = re.sub(r"[^A-Za-z0-9 ]+", " ", sim.replace("_", " ")).strip().title()[:15].strip() or "Fuji Sim"
    rev = 1
    while os.path.exists(os.path.join(out_dir, f"{base} R{rev:02d}")):
        rev += 1
    return f"{base} R{rev:02d}"


def main():
    ap = argparse.ArgumentParser(
        prog="./match",
        description="Clone any Fuji film simulation folder in files/fuji_<filmsim>/ to a Nikon .NP3 profile.",
    )
    ap.add_argument("filmsim", nargs="?", help="Film sim name (e.g. 'classic_neg' or 'astia' for files/fuji_<filmsim>/)")
    ap.add_argument("--name", help="Custom Nikon camera profile name (max 19 ASCII chars)")
    ap.add_argument("--slot", type=int, default=1, help="Nikon PICCON slot number 1-99 (default: 1)")
    ap.add_argument("--quick", action="store_true", help="Faster run (3 coarse + 2 fine rounds)")
    ap.add_argument("--score", metavar="NP3", help="Score an existing .NP3 against files/fuji_<filmsim>/ instead of fitting")
    a = ap.parse_args()

    if not a.filmsim:
        sims = available_sims()
        print("Available Fuji folders in files/:")
        for s in sims:
            print(f"  - {s:20s} -> ./match {s}")
        print("\nUsage:\n  ./match <filmsim>         Fit a new Nikon .NP3 from files/fuji_<filmsim>/\n  ./match <filmsim> --quick Faster fit (~8 min)\n  ./match <filmsim> --score PICCON01.NP3   Score an existing .NP3")
        return

    sim, fuji_dir = resolve_fuji_dir(a.filmsim)
    if not os.path.exists(CORNERS_FILE):
        sys.exit(
            f"Chart corners file not found: {CORNERS_FILE}\n"
            "Run `camera-match-test/scripts/np3fit.py preview` and `grid` once to calibrate corners for your shoot, "
            "or place `chart_corners.json` in reference/."
        )
    matched = find_fuji_files(fuji_dir)
    corners = json.load(open(CORNERS_FILE))
    out_dir = os.path.join(ROOT, "profiles")
    work_dir = os.path.join(ROOT, "work", sim)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    pairs = []
    for pid, nef_name, _, role, ckey, me, _ in PAIR_SPECS:
        t_path = matched.get(pid)
        nef_path = os.path.join(ROOT, "files", "nikon", nef_name)
        if not t_path or not os.path.exists(nef_path):
            continue
        entry = {
            "id": pid,
            "nef": nef_path,
            "target": t_path,
            "role": role,
            "nef_charts": corners["nikon"],
            "target_charts": scale_charts(corners[ckey], t_path),
        }
        if me:
            entry["match_exposure"] = True
        pairs.append(entry)

    if not any(p["role"] == "train" for p in pairs):
        sys.exit(f"Could not match training Fuji JPEGs in {fuji_dir}. Expected DSCF0064/65/67 (or matching EXIF shutter speeds).")

    name = a.name or next_profile_name(sim, out_dir)
    if len(name) > 19 or not name.isascii():
        sys.exit(f"Profile name must be <= 19 ASCII characters: {name!r}")

    cfg = {
        "work_dir": work_dir,
        "output_dir": out_dir,
        "name": name,
        "slot": a.slot,
        "start_np3": TEMPLATE_NP3,
        "match_exposure": False,
        "neutralize_wb": "nikon",
        "wb_chart": "cc24",
        "weighting": "skin_magic",
        "rounds": 3 if a.quick else 4,
        "fine_rounds": 2 if a.quick else 3,
        "workers": min(8, os.cpu_count() or 4),
        "also_fitted_curve": True,
        "also_template_start": False,
        "also_neutral_start": False,
        "charts": {
            "cmp": {"rows": 26, "cols": 38, "fr": 0.5},
            "cc24": {"rows": 6, "cols": 4, "fr": 0.45},
            "mp28": {"rows": 28, "cols": 28, "fr": 0.45},
            "tiffen_c": {"rows": 2, "cols": 9, "fr": 0.35},
            "tiffen_g": {"rows": 1, "cols": 20, "fr": 0.35},
        },
        "pairs": pairs,
    }
    cfg_path = os.path.join(work_dir, "config.json")
    json.dump(cfg, open(cfg_path, "w"), indent=1)

    print(f"Film sim : fuji_{sim} ({len(pairs)} pairs matched)")
    for p in pairs:
        print(f"  {p['id']:10s} [{p['role']:5s}]: {os.path.basename(p['nef'])} <-> {os.path.basename(p['target'])}")

    if a.score:
        subprocess.run([sys.executable, NP3FIT, "score", cfg_path, "--np3", os.path.abspath(a.score)], check=True)
        return

    print(f"Profile  : {name} (slot {a.slot}, {'quick' if a.quick else 'full'} mode)")
    subprocess.run([sys.executable, NP3FIT, "fit", cfg_path], check=True)

    # Copy final .NP3 into root as <filmsim>_YYYY-MM-DD.NP3 (auto-versioned if run again same day)
    np3_src = os.path.join(out_dir, name, "NIKON", "CUSTOMPC", f"PICCON{a.slot:02d}.NP3")
    if os.path.exists(np3_src):
        stamp = time.strftime("%Y-%m-%d")
        dest = os.path.join(ROOT, f"{sim}_{stamp}.NP3")
        n = 2
        while os.path.exists(dest):
            dest = os.path.join(ROOT, f"{sim}_{stamp}_{n}.NP3")
            n += 1
        shutil.copy2(np3_src, dest)
        print(f"\nDone!\n  Profile file  : {dest}\n  Camera folder : {os.path.join(out_dir, name, 'NIKON')}")


if __name__ == "__main__":
    main()
