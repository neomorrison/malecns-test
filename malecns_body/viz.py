"""Video frames: the 3D body next to a live readout of the fly brain."""
from __future__ import annotations

import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(_FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


BG = (18, 20, 26)
FG = (230, 232, 238)
DIM = (140, 146, 160)
COLORS = {
    "GF": (235, 87, 87),
    "STEER": (86, 156, 245),
    "FORWARD": (90, 200, 120),
    "FEED": (245, 180, 60),
    "BACKWARD": (170, 120, 230),
}
BEHAVIOUR_COLORS = {"walk": (90, 200, 120), "run": (60, 220, 160), "jump": (235, 87, 87),
                    "eat": (245, 180, 60), "stand": DIM, "back up": (170, 120, 230)}


class Filmer:
    def __init__(self, agent, width=960, height=540, panel=400, raster_cols=150):
        self.a = agent
        self.W, self.H, self.P = width, height, panel
        self.renderer = mujoco.Renderer(agent.world.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = agent.world.model.body("pelvis").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 6.5, -16, 135
        conn = agent.conn
        a = conn.ann
        dn = np.nonzero((a["superclass"] == "descending_neuron").to_numpy())[0]
        order = np.argsort(a["type"].fillna("").to_numpy()[dn].astype(str), kind="stable")
        self.dn = dn[order]
        self.raster_rows = 180
        self.raster = np.zeros((self.raster_rows, raster_cols), np.uint8)
        self.f_title, self.f_big = _font(17, True), _font(22, True)
        self.f, self.f_small = _font(13), _font(11)

    def _raster_push(self, counts):
        rows = np.array_split(self.dn, self.raster_rows)
        col = np.array([counts[r].sum() for r in rows], float)
        col = np.clip(col * 60, 0, 255).astype(np.uint8)
        self.raster = np.roll(self.raster, -1, axis=1)
        self.raster[:, -1] = col

    def frame(self, counts_since_last):
        a = self.a
        self._raster_push(counts_since_last)
        self.renderer.update_scene(a.world.data, self.cam)
        img = Image.fromarray(self.renderer.render())
        canvas = Image.new("RGB", (self.W + self.P, self.H), BG)
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        x0, y = self.W + 18, 14
        d.text((x0, y), "maleCNS fly brain  →  human body", font=self.f_title, fill=FG)
        y += 24
        d.text((x0, y), f"{a.conn.n:,} spiking neurons   t = {a.t:5.1f} s", font=self.f, fill=DIM)
        y += 26
        beh = a.behaviour.upper()
        d.text((x0, y), beh, font=self.f_big, fill=BEHAVIOUR_COLORS.get(a.behaviour, FG))
        d.text((x0 + 150, y + 6), f"v {a.vx:+.1f} m/s  turn {a.yaw:+.2f}", font=self.f, fill=DIM)
        y += 34
        # hunger
        d.text((x0, y), "hunger", font=self.f, fill=DIM)
        d.rectangle([x0 + 70, y + 3, x0 + 70 + 280, y + 13], outline=(60, 64, 75))
        d.rectangle([x0 + 70, y + 3, x0 + 70 + int(280 * a.hunger), y + 13], fill=(200, 120, 70))
        y += 26
        # sensory drive
        d.text((x0, y), "SENSES → identified input neurons (Hz)", font=self.f_small, fill=DIM)
        y += 16
        inp = a.inputs
        rows = [("LC10 target  L", inp.get("pursuit_l", 0), (86, 156, 245)),
                ("LC10 target  R", inp.get("pursuit_r", 0), (86, 156, 245)),
                ("LPLC2/LC4 loom L", inp.get("loom_l", 0), (235, 87, 87)),
                ("LPLC2/LC4 loom R", inp.get("loom_r", 0), (235, 87, 87)),
                ("sugar GRNs (LB3)", inp.get("sugar", 0), (245, 180, 60))]
        for name, v, col in rows:
            d.text((x0, y), name, font=self.f_small, fill=FG)
            d.rectangle([x0 + 130, y + 2, x0 + 130 + int(220 * min(v, 160) / 160), y + 11], fill=col)
            y += 15
        y += 8
        d.text((x0, y), "COMMAND NEURONS (descending, Hz)", font=self.f_small, fill=DIM)
        y += 16
        r = a.decoder.rates
        nf = max(1, len(a.iface.readout["FORWARD"][0]))
        bars = [("Giant Fiber (jump)", r["GF"], "GF", 400),
                ("DNa01/02, DNg13 (steer)", r["STEER"], "STEER", 300),
                ("P9/oDN1/BDN2 (walk)", r["FORWARD"] / nf, "FORWARD", 120),
                ("MN9 (feed)", r["FEED"], "FEED", 300)]
        for name, lr, key, vmax in bars:
            d.text((x0, y), name, font=self.f_small, fill=FG)
            y += 14
            for side, v in zip("LR", lr):
                d.text((x0 + 4, y), side, font=self.f_small, fill=DIM)
                d.rectangle([x0 + 18, y + 2, x0 + 18 + 330, y + 11], outline=(45, 48, 58))
                d.rectangle([x0 + 18, y + 2, x0 + 18 + int(330 * min(v, vmax) / vmax), y + 11], fill=COLORS[key])
                d.text((x0 + 352, y), f"{v:4.0f}", font=self.f_small, fill=DIM)
                y += 14
            y += 3
        y += 6
        d.text((x0, y), f"all {len(self.dn):,} descending neurons, last {self.raster.shape[1] * 0.04:.0f} s",
               font=self.f_small, fill=DIM)
        y += 15
        rh = min(self.raster_rows, self.H - y - 34)
        ras = Image.fromarray(self.raster[:rh]).resize((360, rh), Image.NEAREST).convert("RGB")
        canvas.paste(ras, (x0, y))
        y += rh + 6
        if a.events:
            t, msg = a.events[-1]
            if a.t - t < 3.0:
                d.text((x0, y), msg[:52], font=self.f_small, fill=(250, 220, 120))
        return np.asarray(canvas)
