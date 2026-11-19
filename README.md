# Nikon Picture Control Matcher

A Claude Code skill and CLI workflow that builds a Nikon Flexible Color Picture Control (`.NP3`) to match a reference look, such as a Fujifilm film simulation.

## Download the profile, LUT and RAW files

Get the free Nikon profile, LUT and RAW chart files here:
https://gerardneedham.com/en-au/products/fujifilm-astia-for-nikon-zf-free-profile-lut-and-raw-chart-files
It works by measurement:

1. Shoot the same colour chart on both cameras, in the same light.
2. The skill renders your Nikon RAW (`.NEF`) through NX Studio's own rendering engine, run from the command line.
3. It compares the chart patches with the reference and tunes the Nikon profile settings until the match stops improving.
4. It writes a versioned `PICCONnn.NP3` you can copy to the camera's card (and a dated `<filmsim>_YYYY-MM-DD.NP3` in the project root when run via `./match`).

---

## Quick Start (`./match`)

Once your chart corners are calibrated in `reference/chart_corners.json`, matching any new Fuji film simulation requires only dropping the Fuji JPEGs into `files/fuji_<filmsim>/` and running one command:

```zsh
./match <filmsim>
```

For example:

```zsh
./match classic_neg          # reads files/fuji_classic_neg/ -> writes classic_neg_YYYY-MM-DD.NP3
./match astia --quick        # faster 3+2 round fit (~8 min)
./match astia --slot 2       # writes PICCON02.NP3 in profiles/<Name>/NIKON/CUSTOMPC/
./match astia --name "Astia R01"
./match classic_neg --score ClassicNeg_PyAdaption.NP3
./match                      # lists available files/fuji_* folders and usage
```

`./match` (symlinked to `match.py`) automatically uses the local `.venv` if present.

---

## Needed Input Files (Fuji & Nikon)

### Nikon Source (`files/nikon/*.NEF`)

- Shoot bracketed `.NEF` RAWs of your chart setup (e.g. `-1.0`, `-0.67`, `-0.33`, `0.0`, `+0.33 EV`) with **Active D-Lighting Off**, fixed aperture (e.g. `f/5.6`), base ISO (`ISO 100`), and fixed preset/Kelvin white balance.
- No manual pre-conversion is needed — `SaveProcess` renders the `.NEF` files directly.

### Fuji Target (`files/fuji_<filmsim>/*.jpg` or `.hif`)

- Export directly from the matching Fuji `.RAF` files using **Fujifilm X RAW Studio** or **RAW File Converter EX** (or shoot straight-out-of-camera JPEG/HEIF) so Fuji's genuine film simulation color science is baked in:
  - **Film Simulation**: target look (e.g. Classic Neg, Astia, Classic Chrome, Provia).
  - **Dynamic Range**: `DR100`, **Grain**: `Off`, **Color Chrome / FX Blue**: `Off` (unless intentionally part of the look), **Sharpness / NR**: lowest.
  - **Exposure brackets**: match the same shutter speeds as the Nikon `.NEF` frames (e.g. `1/80s`, `1/60s`, `1/50s`, `1/40s`, `1/30s`). If an extreme shutter speed isn't available on Fuji, an exposure-adjusted substitute (e.g. `DSCF0064_+0.2EV.jpg`) can be held out as a test pair with `match_exposure: true`.
  - **Resolution & Color Space**: export in `sRGB` at `6048×4032` to match Nikon Z f geometry directly (if another resolution is exported, `match.py` scales calibrated chart corners proportionally).
- Put each film simulation in its own folder: `files/fuji_<filmsim>/` (for example `files/fuji_classic_neg/`, `files/fuji_astia/`).

---

## Folder & File Structure

```text
nikon-picture-control-matcher-main/
├── match -> match.py                  # One-command CLI launcher (auto-uses .venv)
├── match.py                           # Auto-pairs files/nikon + files/fuji_<sim> & runs fit/score
├── <filmsim>_YYYY-MM-DD.NP3           # Final dated profile written to project root
├── camera-match-test/
│   ├── SKILL.md                       # Full Claude Code skill workflow & guidelines
│   ├── assets/
│   │   ├── flexible_color_template.NP3
│   │   └── flexible_color_template_alt.NP3
│   └── scripts/
│       ├── np3lib.py                  # NP3 binary codec, SaveProcess runner, multi-chart & CIEDE2000 math
│       └── np3fit.py                  # Multi-chart fitter, analytical 1D curve init, coarse+fine Gauss-Newton
├── reference/
│   ├── chart_corners.json             # Calibrated full-res corner coordinates for all 5 charts
│   ├── classic_neg_config.json        # Example reproducible config
│   ├── fuji_sim_template_config.json  # Template config for manual runs
│   └── overlays/                      # Visual box-alignment overlays for verification
├── files/                             # Local RAW/JPEG inputs (gitignored)
│   ├── nikon/                         # Nikon .NEF chart brackets
│   └── fuji_<filmsim>/                # Exported Fuji film simulation JPEGs/HEIFs
├── profiles/                          # Full versioned output (<Name>/NIKON/CUSTOMPC/PICCONnn.NP3 + settings.json)
└── work/                              # Temporary renders & run logs (gitignored)
```

---

## Key Improvements in This Version

1. **Multi-Chart & Skin-Tone Prioritization (`skin_magic`)**:
   - Generalized `np3lib.py` and `np3fit.py` from single-CMP fitting to arbitrary multi-chart sets (`CMP Digital Target Ultimate` 26×38 + `ColorChecker Classic 24` + `Magic Palette` 28×28 + `Tiffen Color Control` 2×9 + `Tiffen Gray Scale` 1×20 — 1,834 usable patches per frame).
   - `skin_magic` weighting prioritizes ColorChecker Light/Dark Skin, patches inside the human skin-tone Lab locus, neutral ramps (tone & split-toning), and everyday colors while down-weighting extreme neon inks.
2. **Unbiased White Balance & Clean Masking**:
   - Computes Nikon WB neutralization gains on a true linear-neutral render (`Template.neutral_base()`) rather than the tinted starting template, preferring ColorChecker neutral greys when present (avoiding CMP's ~9% magenta paper bias).
3. **Expanded Flexible Color `.NP3` Controls & Smooth 13-Knot Tone Curve**:
   - Unlocks verified `SaveProcess` parameter ranges (`±100` Color Blender hue/brightness, `±100` Saturation, `±100` Color Grading brightness/balance, `±80` grading wheels) and a 13-knot tone curve including black/white endpoints with monotonic projection and second-difference smoothness regularization.
4. **Analytical Tone-Curve Pre-Fit + Exact CIEDE2000 Residual**:
   - Fits an initial 1D tone curve via SLSQP on neutral patches before running 8-worker coarse + fine Gauss-Newton optimization using `de00_vec()` (including the CIEDE2000 `RT` blue-purple rotation term).
5. **Zero External Dependencies Beyond Python + macOS Built-ins**:
   - `profile_of()` falls back to macOS built-in `sips` when `exiftool` is not installed.
   - `locate()` uses coarse affine pre-alignment (and prevents thin 1–2 row strips from collapsing).

## Limits

- The NP3 format has a limited set of controls (1D tone curve + 8-band Color Blender + 3-zone Color Grading, not a 3D LUT), so the match is close but not exact.
- Results depend on your chart, light and cameras. Always test on real-world photos and portraits.
- This is an experiment shared as-is. No guarantees.

## Requirements

- macOS with Nikon NX Studio installed. Nothing from Nikon is included here.
- Python 3 with `numpy`, `scipy`, `opencv-python`, `tifffile` (stored in `.venv/` for `./match`), and optionally Claude Code.
- A colour chart (or multi-chart board) and reference camera shots of the same chart.
- The command-line rendering used here (`SaveProcess`) is not an official NX Studio feature and may stop working after an update.

## Install (Claude Code Skill)

Copy the `camera-match-test` folder into `~/.claude/skills/`, then ask Claude Code something like:

> "Use the camera-match-test skill to make my Nikon match my Fuji Astia."

`camera-match-test/SKILL.md` has the skill workflow.

## Licence

MIT. Free to use and modify. Not affiliated with Nikon or Fujifilm.
