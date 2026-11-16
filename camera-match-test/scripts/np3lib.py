#!/usr/bin/env python3
"""Shared tools: NP3 read/write, NX Studio SaveProcess rendering, CMP chart measurement, colour maths."""
import base64, math, os, struct, subprocess, tempfile
import numpy as np, cv2, tifffile
from xml.sax.saxutils import escape
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

SAVEPROCESS = "/Applications/Nikon Software/NX Studio/NX Studio.app/Contents/MacOS/SaveProcess"
SRGB_ICC = "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
ROWS, COLS = 26, 38  # CMP Digital Target Ultimate: rows A-Z, columns 1-38, 988 patches
BANDS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
ZONES = ("high", "mid", "shadow")
KNOTS = np.array([.05, .12, .2, .3, .42, .55, .7, .85, .95])


# ---------- NP3 ----------

def walk(b):
    """Record table of an NCP/NP2/NP3 file: {type: (length, data offset)}."""
    if b[:4] != b"NCP\0":
        raise ValueError("not a Picture Control file")
    p, out = 4, {}
    while True:
        t = struct.unpack(">I", b[p:p + 4])[0]
        if t == 0:
            if p + 4 != len(b):
                raise ValueError("trailing bytes after terminator")
            return out
        n = struct.unpack(">I", b[p + 4:p + 8])[0]
        out[t] = (n, p + 8)
        p += 8 + n


class Template:
    """A Flexible Color NP3 with a custom curve; new profiles are this file with controls rewritten."""

    def __init__(self, path):
        self.base = open(path, "rb").read()
        self.rec = walk(self.base)
        for t, need in ((0x100, None), (0x200, 20), (0x1E00, 2), (0x1F00, 28), (0x2000, 20), (0x2, 578)):
            if t not in self.rec or (need and self.rec[t][0] != need):
                raise ValueError(f"{path}: needs a Flexible Color NP3 with a custom tone curve (record {t:#x} missing or wrong size)")
        if self.base[self.rec[0x100][1]:self.rec[0x100][1] + 4] != b"0310":
            raise ValueError(f"{path}: not a Flexible Color (0310) NP3")
        self.BL, self.GR, self.CU, self.SAT = (self.rec[k][1] for k in (0x1F00, 0x2000, 0x2, 0x1E00))
        self.lut = np.array(struct.unpack(">257H", self.base[self.CU + 64:self.CU + 578]), float) / 32767
        self.params = self._params()
        self.names = [p[0] for p in self.params]
        self.lo = np.array([p[3] for p in self.params], float)
        self.hi = np.array([p[4] for p in self.params], float)
        self.step = np.array([p[2] for p in self.params], float)

    def _params(self):
        b, P = self.base, []
        for i, n in enumerate(BANDS):
            P += [(f"{n}_hue", b[self.BL + 3 * i] - 0x80, 10, -70, 70), (f"{n}_chroma", b[self.BL + 3 * i + 1] - 0x80, 12, -100, 100),
                  (f"{n}_bright", b[self.BL + 3 * i + 2] - 0x80, 10, -60, 60)]
        P += [("saturation", b[self.SAT] - 0x80, 10, -50, 50)]
        for j, z in enumerate(ZONES):
            o = self.GR + 4 * j
            hue = ((b[o] & 0x0F) << 8) + b[o + 1]
            ch = b[o + 2] - 0x80
            P += [(f"grade_{z}_x", ch * math.cos(math.radians(hue)), 12, -50, 50), (f"grade_{z}_y", ch * math.sin(math.radians(hue)), 12, -50, 50),
                  (f"grade_{z}_bright", b[o + 3] - 0x80, 10, -40, 40)]
        P += [("grade_blending", b[self.GR + 16] - 0x80, 20, 0, 100), ("grade_balance", b[self.GR + 18] - 0x80, 20, -60, 60)]
        P += [(f"curve_{k:.2f}", 0, 6, -25, 25) for k in KNOTS]
        return P

    def start(self, reset_colour=False):
        x = np.array([p[1] for p in self.params], float)
        if reset_colour:  # neutral colour controls, keep the template's tone curve
            for i, n in enumerate(self.names):
                if not n.startswith("curve_"):
                    x[i] = 50 if n == "grade_blending" else 0
        return np.clip(x, self.lo, self.hi)

    def make(self, x, name=None):
        b = bytearray(self.base)
        v = dict(zip(self.names, x))
        for i, n in enumerate(BANDS):
            for j, k in enumerate(("hue", "chroma", "bright")):
                b[self.BL + 3 * i + j] = 0x80 + int(round(v[f"{n}_{k}"]))
        b[self.SAT] = 0x80 + int(round(v["saturation"]))
        for j, z in enumerate(ZONES):
            gx, gy = v[f"grade_{z}_x"], v[f"grade_{z}_y"]
            chroma = min(100, int(round(math.hypot(gx, gy))))
            hue = int(round(math.degrees(math.atan2(gy, gx)) % 360)) % 360 if chroma else 0
            o = self.GR + 4 * j
            b[o], b[o + 1], b[o + 2], b[o + 3] = 0x80 + (hue >> 8), hue & 0xFF, 0x80 + chroma, 0x80 + int(round(v[f"grade_{z}_bright"]))
        b[self.GR + 16] = 0x80 + int(round(v["grade_blending"]))
        b[self.GR + 18] = 0x80 + int(round(v["grade_balance"]))
        deltas = np.array([v[f"curve_{k:.2f}"] for k in KNOTS]) / 255
        if np.any(np.round(deltas * 255)):
            fix = PchipInterpolator(np.r_[0, KNOTS, 1], np.r_[0, deltas, 0])(np.arange(257) / 256)
            lut = np.maximum.accumulate(np.clip(self.lut + fix, 0, 1))
            lut16 = np.round(lut * 32767).astype(int)
            lut16[0], lut16[-1] = 0, 32767
            # Nikon's software rebuilds the curve from the control points, so write points that follow the LUT
            px = np.round(np.linspace(0, 255, 20)).astype(int)
            py = np.maximum.accumulate(np.round(np.interp(px * 256 / 255, np.arange(257), lut16 / 32767) * 255).astype(int))
            b[self.CU + 8] = 20
            b[self.CU + 9:self.CU + 49] = bytes(int(q) for pair in zip(px, py) for q in pair)
            b[self.CU + 64:self.CU + 578] = struct.pack(">257H", *lut16.tolist())
        if name is not None:
            if len(name) > 19 or not name.isascii():
                raise ValueError("profile name must be ASCII, 19 characters or fewer")
            n, p = self.rec[0x200]
            b[p:p + n] = name.encode("ascii").ljust(n, b"\0")
        out = bytes(b)
        if list(walk(out)) != list(self.rec) or len(out) != len(self.base):
            raise AssertionError("NP3 structure changed while writing")
        return out


def np3_name(b):
    n, p = walk(b)[0x200]
    return b[p:p + n].rstrip(b"\0").decode("ascii", "replace")


# ---------- rendering ----------

def nksc(np3):
    s = base64.b64encode(np3).decode()
    lines = [s[i:i + 60] for i in range(0, len(s), 60)]
    blob = "\n".join(lines) + "\n"
    user = ('<userData><filter id="nikon::PictureControl"><active>true</active><parameters><Export>'
            f'<ExportData>{blob}</ExportData><ExportDataSize>{len(s) + len(lines)}</ExportDataSize></Export>'
            '<SelectedPictureControl>0</SelectedPictureControl><AutoContrast>255</AutoContrast><AutoSaturation>255</AutoSaturation>'
            '<SelectedPictureControlVersion2>0</SelectedPictureControlVersion2><SavedPicConProcess>0</SavedPicConProcess>'
            '<PictureControl>16</PictureControl></parameters></filter></userData>')
    esc = escape(user).replace("\n", "&#xA;")
    return f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.5.0">
   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:sdc="http://ns.nikon.com/sdc/1.0/" xmlns:ast="http://ns.nikon.com/asteroid/1.0/" xmlns:astype="http://ns.nikon.com/asteroid/Types/1.0/" xmlns:nine="http://ns.nikon.com/nine/1.0/">
         <sdc:about>nikon sidecar/1.0</sdc:about>
         <sdc:version>1.1.0</sdc:version>
         <sdc:appversion>1.10 M</sdc:appversion>
         <sdc:appname>NX Studio</sdc:appname>
         <ast:about>core-asteroid-tags</ast:about>
         <ast:version>11.0.0.3000</ast:version>
         <nine:about>nine-tags</nine:about>
         <nine:version>2.0.0</nine:version>
         <nine:NineEdits>{esc}</nine:NineEdits>
      </rdf:Description>
   </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''


def render(nef, np3, width, workdir):
    """Render a NEF through NX Studio's engine with the given NP3. Returns float sRGB in camera-JPEG geometry."""
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        out, sc, job = d + "/o.tif", d + "/s.nksc", d + "/j.hyd"
        open(sc, "w", encoding="utf8").write(nksc(np3))
        lines = [f"InputFile={nef}", f"OutputFile={out}", f"SidecarFile={sc}", "Format=FileType_TIFF", "TiffOption=ColorType_RGB16", "JpegQuality=100",
                 f"ConvertWidth={width}", f"ConvertHeight={width}", "ConvertTosRGB=1", "UsePreview=0", "ApplicationName=NX_Studio", "DevelopColorMode=0",
                 "UpScaleRawSize=0", "DynamicRange=0", "HeifOption=0", "IsUseRGBProfileInstead=0", "TaskTypeSet=0"]
        open(job, "w").write("\n".join(lines) + "\n")
        r = subprocess.run([SAVEPROCESS, "TCJ", job], capture_output=True, text=True, timeout=300)
        if not os.path.exists(out):
            raise RuntimeError("SaveProcess produced no image: " + (r.stdout + r.stderr)[:400])
        return tifffile.imread(out).astype(np.float32) / 65535


# ---------- reference images ----------

def profile_of(path):
    r = subprocess.run(["exiftool", "-s3", "-ProfileDescription", "-BitsPerSample", path], capture_output=True, text=True)
    return r.stdout.strip().replace("\n", " | ")


def load_reference(path, workdir):
    """Load a target image (TIFF, JPEG, HEIF/HEIC) as float sRGB. Anything not already sRGB goes through macOS ColorSync first."""
    desc = profile_of(path)
    ext = os.path.splitext(path)[1].lower()
    srgb = ("srgb" in desc.lower() or desc.split(" | ")[0] in ("", "-")) and ext not in (".heif", ".heic", ".hif")
    if not srgb:
        out = os.path.join(workdir, os.path.basename(path) + ".srgb.tif")
        if not os.path.exists(out):
            subprocess.run(["sips", "-m", SRGB_ICC, "-s", "format", "tiff", path, "--out", out], check=True, capture_output=True)
        path = out
    if path.lower().endswith((".tif", ".tiff")):
        a = tifffile.imread(path)
        if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
            a = np.moveaxis(a, 0, -1)
        a = a[..., :3]
        scale = 65535 if a.dtype == np.uint16 else 255 if a.dtype == np.uint8 else 1
        img = a.astype(np.float32) / scale
    else:
        img = cv2.imdecode(np.frombuffer(open(path, "rb").read(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img = img[..., ::-1].astype(np.float32) / (65535 if img.dtype == np.uint16 else 255)
    return img, desc


# ---------- chart ----------

def grid_points(corners, u, v):
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [0, 1], [1, 1]]), np.float32(np.asarray(corners, float).reshape(4, 2)))
    p = np.stack([u, v, np.ones_like(u)], -1) @ H.T
    return p[..., :2] / p[..., 2:]


def uv(fr, n):
    cc, rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    o = np.linspace(-fr / 2, fr / 2, n)
    ox, oy = np.meshgrid(o, o)
    return (cc[..., None, None] + ox) / (COLS - 1), (rr[..., None, None] + oy) / (ROWS - 1)


def locate(img, guess):
    """Refine the four corner patch centres (A1, A38, Z1, Z38) by making every patch as uniform as possible."""
    g = cv2.GaussianBlur(img.mean(2), (0, 0), 2)
    u, v = uv(.55, 7)

    def cost(c):
        p = grid_points(c, u, v)
        return map_coordinates(g, [p[..., 1].ravel(), p[..., 0].ravel()], order=1).reshape(p.shape[:-1]).std(axis=(2, 3)).mean()

    best = np.asarray(guess, float)
    scale = max(img.shape[:2]) / 6048
    for step in (24, 8, 3):
        best = minimize(cost, best, method="Powell", options=dict(xtol=.2, ftol=1e-7, direc=np.eye(8) * step * scale, maxiter=20)).x
    return best, float(cost(best))


def measure(img, corners):
    u, v = uv(.5, 11)
    p = grid_points(corners, u, v)
    m = np.zeros((ROWS, COLS, 3))
    for k in range(3):
        m[..., k] = np.median(map_coordinates(img[..., k], [p[..., 1].ravel(), p[..., 0].ravel()], order=1).reshape(p.shape[:-1]), axis=(2, 3))
    return m


def overlay(img, corners, valid, path):
    sc = 1512 / img.shape[1]
    small = (cv2.resize(np.clip(img, 0, 1), (1512, int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)[..., ::-1] * 255).astype(np.uint8).copy()
    u, v = uv(.5, 2)
    p = grid_points(corners, u, v) * sc
    for r in range(ROWS):
        for q in range(COLS):
            a = p[r, q]
            cv2.rectangle(small, tuple(a[0, 0].astype(int)), tuple(a[1, 1].astype(int)), (0, 255, 255) if valid is None or valid[r, q] else (0, 0, 255), 1)
    cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 88])


def preview_with_ruler(img, path):
    """Downsized preview with tick labels in full-resolution pixels, for reading off corner positions."""
    sc = 1512 / img.shape[1]
    small = (cv2.resize(np.clip(img, 0, 1), (1512, int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)[..., ::-1] * 255).astype(np.uint8).copy()
    step = 500
    for x in range(0, img.shape[1], step):
        cv2.line(small, (int(x * sc), 0), (int(x * sc), small.shape[0]), (0, 255, 0), 1)
        cv2.putText(small, str(x), (int(x * sc) + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    for y in range(0, img.shape[0], step):
        cv2.line(small, (0, int(y * sc)), (small.shape[1], int(y * sc)), (0, 255, 0), 1)
        cv2.putText(small, str(y), (2, int(y * sc) - 2), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 0), 1)
    cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------- white balance ----------

def _lin(v):
    return np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055) ** 2.4)


def _enc(v):
    return np.where(v <= .0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - .055)


def ring_mask():
    """The chart's outer ring (row A, row Z, columns 1 and 38) is a checker of neutral greys and whites."""
    m = np.zeros((ROWS, COLS), bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def wb_gains(m):
    """Linear R and B gains that make the chart's grey patches neutral, keeping green (and so brightness) fixed.

    Starts from the outer grey ring, then repeatedly adds every patch that reads as near-neutral after the current
    correction, so dozens of greys across the chart vote and a single tinted or unevenly lit patch can't sway it."""
    lin = _lin(m)
    ok = (lin.min(-1) > .03) & (lin.max(-1) < .9)
    use = ring_mask() & ok
    if use.sum() < 10:
        raise RuntimeError("too few clean neutral patches to balance white")
    gains = np.ones(3)
    for _ in range(4):
        r, g, b = np.median(lin[use], axis=0)
        gains = np.array([g / r, 1.0, g / b])
        l = lab(_enc(lin * gains))
        use = ok & (np.hypot(l[..., 1], l[..., 2]) < 4)
        if use.sum() < 10:
            use = ring_mask() & ok
    return gains


def apply_gains(m, gains):
    return np.clip(_enc(_lin(m) * gains), 0, 1)


def describe_gains(g):
    return f"red x{g[0]:.3f}, blue x{g[2]:.3f}" + (" (already neutral)" if abs(g[0] - 1) < .02 and abs(g[2] - 1) < .02 else "")


# ---------- colour ----------

def lab(v):
    l = np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055) ** 2.4)
    M = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .072175], [.0193339, .119192, .9503041]])
    x = l @ M.T / np.array([.95047, 1, 1.08883])
    f = np.where(x > 216 / 24389, np.cbrt(x), (841 / 108) * x + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def de00_parts(a, b):
    """Scaled lightness, chroma and hue differences of CIEDE2000, plus its rotation term."""
    L1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    L2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = .5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1, h2 = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    dh = ((h2 - h1 + 180) % 360) - 180
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hm = (h1 + dh / 2) % 360
    T = 1 - .17 * np.cos(np.radians(hm - 30)) + .24 * np.cos(np.radians(2 * hm)) + .32 * np.cos(np.radians(3 * hm + 6)) - .2 * np.cos(np.radians(4 * hm - 63))
    SL = 1 + .015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2)
    SC, SH = 1 + .045 * Cpm, 1 + .015 * Cpm * T
    RT = -2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7)) * np.sin(np.radians(60 * np.exp(-((hm - 275) / 25) ** 2)))
    return dL / SL, dC / SC, dH / SH, RT


def de00(a, b):
    l, c, h, rt = de00_parts(a, b)
    return np.sqrt(np.maximum(l ** 2 + c ** 2 + h ** 2 + rt * c * h, 0))


def de76(a, b):
    return np.linalg.norm(a - b, axis=-1)
