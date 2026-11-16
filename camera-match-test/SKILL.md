---
name: camera-match-test
description: Test how closely a Nikon Z camera (Zf, Z6III, Z5II, Z50II, Z8, ZR) can match a reference look in-camera, and build a Nikon Flexible Color Picture Control (.NP3) for it — a Fujifilm JPEG/HEIF film simulation, a Lightroom profile or preset export, or a LUT'd image. It fits the NP3 against a colour chart shot by rendering NEFs headlessly through NX Studio's own engine (SaveProcess), measuring the 988-patch CMP chart, and tuning Color Blender, Color Grading, saturation and tone curve to minimise ΔE2000, then writes a versioned PICCONnn.NP3 and can copy it to the SD card. Use this whenever the user wants to match a Nikon to a Fuji, turn a look, LUT or film simulation into an in-camera Nikon profile, make a "Nikon recipe" from a reference image, or check how close a profile is in Delta E — even if they just say "make my Zf look like my X100" or "build the profile from this chart".
---

# Camera Match Test

This skill turns a reference look into a Nikon in-camera profile by measurement, not guesswork. The core idea: NX Studio's `SaveProcess` binary renders a NEF with any NP3 in about 2 seconds, and its renders closely match the camera's own JPEGs. So every candidate profile can be tested on real Nikon processing, on the Mac, with no camera in the loop until the end.

**Input** = the Nikon NEF of the chart, rendered through NX Studio with the profile being fitted.
**Reference (target)** = the look to match, for example a Fuji HEIF/JPEG of the same chart in the same light, or a Lightroom TIFF export of the Nikon NEF.

## Before shooting (tell the user this up front)

For a cross-camera reference (Fuji to Nikon), the chart pair must differ only in camera:

- Same chart (CMP Digital Target Ultimate, 988 patches), same light, same position. Even, diffuse light; avoid glare on the chart.
- White balance: a fixed, matching Kelvin on both cameras is cleanest, but auto WB is fine because the fitter can correct it (see `neutralize_wb` below). An NP3 has no white balance setting of its own, so whatever cast is left in after that gets learned as part of the look.
- Expose so the white patches aren't clipped on either camera. Matching exposure values is less important than both being clean, because the fitter can level exposure (see `match_exposure`).
- Fuji: chosen film simulation, Grain off, Color Chrome off unless it's part of the look, DR100, Highlight/Shadow at the values you want baked into the look, Sharpness and NR at their lowest, lens corrections at default. HEIF needs a recent Fuji (X-H2/X-T5/X100VI era); an older X100-series body only writes JPEG, and JPEG is fine.
- Nikon: Flexible Color Picture Control as the base, Active D-Lighting off, RAW + JPEG.
- Shoot at least three pairs, moving the chart or exposure slightly between them, so some pairs can be held back as a fair test.
- Exposure brackets work well as pairs: match each Nikon frame to the Fuji frame with the same exposure offset (check `ExposureCompensation` and shutter speed with exiftool rather than trusting file order). With five brackets, fit on the three middle pairs and hold back the two ends.
- The two cameras can frame differently (different focal lengths, lens corrections, crop). That's fine, because each image gets its own chart corners and only patch averages are compared. The whole chart just needs to be in frame and in focus on both.
- A ColorChecker beside the CMP chart is handy for eyeballing white balance and accuracy on screen, but the fitter only measures the CMP chart.
- HEIF targets are converted to sRGB with macOS `sips`, which writes 8 bits. That's fine, because each patch averages thousands of pixels.
- Use the Fuji's own JPEG or HEIF as the target, not its RAF. That file is the film simulation; a RAF would need a raw converter, which brings in that converter's look instead of Fuji's.

## Workflow

Scripts live in `scripts/` next to this file. Run them with `python3 <skill>/scripts/np3fit.py <command> <config.json>`. Keep everything — copied NEFs, renders, configs and the fitted profile — in the session scratchpad, and ask the user where the finished profile should be saved before copying it anywhere else.

1. **Copy NEFs off the card** into a private work folder first. Never edit or delete anything on the card, and check the card isn't read-only before copying back later (`mount | grep -i volumes`).
2. **Write the config.** Template:

   ```json
   {
     "work_dir": "<scratchpad>/np3fit",
     "output_dir": "<scratchpad>/profiles",
     "name": "Fuji Provia R01",
     "slot": 1,
     "start_np3": "<skill>/assets/flexible_color_template.NP3",
     "match_exposure": false,
     "neutralize_wb": "nikon",
     "weighting": "everyday",
     "rounds": 3,
     "pairs": [
       {"id": "A", "nef": ".../NZF_0001.NEF", "target": ".../DSCF0001.HIF", "role": "train", "camera_jpeg": ".../NZF_0001.JPG",
        "nef_corners": [0,0, 0,0, 0,0, 0,0], "target_corners": [0,0, 0,0, 0,0, 0,0]},
       {"id": "B", "...": "...", "role": "test"}
     ]
   }
   ```

   - `name` is what appears on the camera: ASCII, 19 characters at most. Use the rev numbering (`R01`, `R02` …) so nothing is ever overwritten.
   - `start_np3` must be a Flexible Color NP3 with a custom tone curve. The bundled `flexible_color_template.NP3` works as a template for any look: `fit` also tries a neutral-colour start from the same file and keeps whichever fits better.
   - `neutralize_wb` corrects white balance using the chart's grey patches (outer grey ring plus every patch that reads near-neutral, refined over a few passes):
     - `"nikon"`: balance only the Nikon render to neutral and keep the reference exactly as shot. Use this when the reference's colour cast is part of the look, for example a Fuji recipe with a WB shift (check `WhiteBalance` and `WhiteBalanceFineTune` in the Fuji's metadata). The profile then learns that warmth or coolness as a tint, and the Nikon can shoot on auto WB in the field.
     - `"both"`: balance both images to neutral. Use this when any cast is unwanted (different light, accidental mismatch), so the profile learns colour only.
     - `false`: compare as shot. Use this when both cameras used the same fixed Kelvin.
     The log shows the correction each image needed. A few percent is normal; much more means the cameras saw different light or one was set badly.
   - Corners are the centres of patches **A1, A38, Z1, Z38** in full-resolution pixels of that image, as `[x1,y1, x2,y2, x3,y3, x4,y4]`. Go by the printed row letters and column numbers, not by screen position, because the chart is often shot rotated or upside down; A1 can be at the bottom right. NEF corners are in the camera-JPEG geometry (NX renders apply the camera's lens corrections, so they line up with the camera JPEG, not with a Lightroom export).
3. **`preview`** writes ruler previews of each NEF render and each target. Look at them and read off the four corner patch centres; rough is fine (within half a patch).
4. **`grid`** refines the corners, writes overlay images, and saves a `_grid.json` config. Look at every overlay: each yellow box must sit inside a single patch. Use the `_grid.json` from here on.
5. **`score`** gives the starting point and, if `camera_jpeg` is set, what the camera currently produces. Report this baseline before fitting.
6. **`fit`** (runs 10–40 minutes, so start it in the background and tell the user roughly how long). It writes `<output_dir>/<name>/NIKON/CUSTOMPC/PICCONnn.NP3` plus `settings.json`, and never overwrites: if the folder exists it adds `_rev2`, `_rev3`.
7. **Report honestly.** Lead with held-out ΔE2000 (pairs with `role: test`), then fitted pairs, everyday colours and neutrals. Mention controls stuck at their limits, because they mean the format ran out of room. 
8. **Copy to the card only when the user says so**: `cp -n` into `<card>/NIKON/CUSTOMPC/`, creating that folder if needed, then compare checksums. Camera menu: Photo Shooting Menu → Manage Picture Control → Load/save → Copy to camera, then Set Picture Control.
9. **Verify on camera.** Before scoring any camera JPEG, read its `PictureControlName` with exiftool. Then run `score` with that JPEG as `camera_jpeg`.

## Reading results

- ΔE2000 is the headline metric. Lower is closer; small values are hard to see side by side.
- NP3 profiles have limited controls, so repeated rounds stop improving at some point. When that happens, say the format's limit is reached instead of looping.
- Chart scores are a strong guide, not proof. Suggest a few real photos at the end, and say plainly if skin hasn't been tested.

## Things that matter

- Never overwrite files; version everything.
- Nikon's software rebuilds the curve from control points, so `np3lib` writes 20 points that follow the LUT. Don't hand-edit curves without doing the same.
- If SaveProcess output looks unchanged after changing the NP3, check the sidecar is being read: a green-chroma −100 test should desaturate only greens.
