# Nikon Picture Control Matcher

A Claude Code skill that builds a Nikon Flexible Color Picture Control (`.NP3`) to match a reference look, such as a Fujifilm film simulation.

It works by measurement:

1. Shoot the same colour chart on both cameras, in the same light.
2. The skill renders your Nikon RAW (`.NEF`) through NX Studio's own rendering engine, run from the command line.
3. It compares the chart patches with the reference and tunes the Nikon profile settings until the match stops improving.
4. It writes a versioned `PICCONnn.NP3` you can copy to the camera's card.

## Limits

- The NP3 format has a limited set of controls, so the match is close but not exact.
- Results depend on your chart, light and cameras. Check the result on your own photos.
- This is an experiment I shared as-is. No guarantees.

## Requirements

- macOS with Nikon NX Studio installed. Nothing from Nikon is included here.
- Claude Code, and Python 3 with `numpy`, `scipy`, `opencv-python`, `tifffile`.
- A colour chart and a reference camera shot of the same chart.
- The command-line rendering used here is not an official NX Studio feature and may stop working after an update.

## Install

Copy the `camera-match-test` folder into `~/.claude/skills/`, then ask Claude Code something like:

> "Use the camera-match-test skill to make my Nikon match my Fuji Astia."

`camera-match-test/SKILL.md` has the workflow.

## Licence

MIT. Free to use and modify. Not affiliated with Nikon or Fujifilm.
