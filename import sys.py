import sys
import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageTk

# ---------- Settings ----------
MAX_W, MAX_H = 150, 76      # biggest grid (character columns / rows)
CW, CH = 5, 10              # pixel size of one character cell (CH = 2 * CW)
FONT_SIZE = 8
FPS = 30
LOOP = 15.0                 # animation length in seconds
COLOR_MODE = "photo"        # "photo" = real colors, "orange" = one warm color
GAMMA = 0.9                 # below 1 = brighter picture, above 1 = darker
RAMP_SIZE = 28              # how many different characters to use

ORANGE = np.array([255, 150, 70], np.float32)
WHITE = np.array([235, 235, 255], np.float32)
PINK = np.array([255, 105, 170], np.float32)


def load_font(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "Menlo.ttc", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def glyph(font, c):
    img = Image.new("L", (CW, CH), 0)
    ImageDraw.Draw(img).text((0, 0), c, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255


def build_atlas():
    """Pick characters whose 'ink' goes smoothly from empty to full."""
    font = load_font(FONT_SIZE)
    chars = [chr(i) for i in range(32, 127)]
    bitmaps = np.array([glyph(font, c) for c in chars])
    ink = bitmaps.sum(axis=(1, 2))
    targets = np.linspace(ink.min(), ink.max(), RAMP_SIZE)
    picked = sorted({int(np.argmin(np.abs(ink - tg))) for tg in targets}, key=lambda i: ink[i])
    ramp = bitmaps[picked]
    stars = np.array([glyph(font, c) for c in ".+*"])
    return np.concatenate([ramp, stars]), len(ramp)


atlas, N = build_atlas()
STAR_IDX = np.array([N, N + 1, N + 2])      # star glyphs sit after the ramp


class Scene:
    def __init__(self, path):
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        aspect = img.width / img.height
        rows = MAX_H
        cols = int(rows * 2 * aspect)             # a cell is half as wide as it is tall
        if cols > MAX_W:
            cols = MAX_W
            rows = int(cols / (2 * aspect))
        self.W, self.H = cols, rows

        # sharpen a little, then shrink to one pixel per character cell
        radius = max(img.size) / 150
        img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=120, threshold=2))
        small = img.resize((cols, rows), Image.LANCZOS)

        # brightness 0..1 with stretched contrast
        gray = np.asarray(small.convert("L"), dtype=np.float32) / 255
        lo, hi = np.percentile(gray, (2, 98))
        gray = np.clip((gray - lo) / max(hi - lo, 1e-3), 0, 1) ** GAMMA

        # color of each cell
        if COLOR_MODE == "photo":
            col = np.asarray(small, dtype=np.float32)
            col = col * (255 / np.maximum(col.max(axis=2, keepdims=True), 70))
            col = np.clip(col, 0, 255)
        else:
            col = np.broadcast_to(ORANGE, (rows, cols, 3)).copy()
        self.col = col

        # coordinates, vignette (fades the corners to black)
        ys, xs = np.mgrid[0:rows, 0:cols]
        x = (xs - (cols - 1) / 2) / (cols / 2)
        y = (ys - (rows - 1) / 2) / (rows / 2)
        oval = np.sqrt(x ** 2 + y ** 2)
        vignette = np.clip((1.0 - oval) / 0.3, 0, 1)
        self.b = gray * vignette
        self.on = vignette > 0.02
        self.rr = oval / 1.2
        self.yy = ys / (rows - 1)

        rng = np.random.default_rng(7)
        self.noise = rng.random((rows, cols))
        self.star_mask = rng.random((rows, cols)) < 0.02
        self.phase = rng.random((rows, cols)) * 6.28

    def frame(self, t):
        level = np.interp(t, [0, 4, 9, 13, 15], [0, 1.05, 1.05, 0, 0])
        visible = self.on & ((0.6 * self.noise + 0.4 * np.clip(self.rr, 0, 1)) < level)

        # the picture: a little shimmer + a bright line that sweeps down once
        flick = np.sin(t * 5 + self.phase)
        scan_pos = np.interp(t, [4, 8], [-0.2, 1.2])
        scan = np.exp(-((self.yy - scan_pos) / 0.03) ** 2)
        b = np.clip(self.b + 0.03 * flick + 0.3 * scan * (self.b > 0.05), 0, 1)
        idx = np.clip(b * (N - 1), 0, N - 1).astype(int)
        col = self.col + (WHITE - self.col) * (0.5 * scan)[..., None]

        # background: pink glow + twinkling stars
        glow = np.exp(-(self.rr / 0.5) ** 2) * (1 - min(level / 1.05, 1))
        nb = glow * (0.4 + 0.6 * self.noise)
        neb_idx = np.clip((nb * 6).astype(int), 0, 5)
        twinkle = (np.sin(t * 3 + self.phase) + 1) / 2
        star_idx = STAR_IDX[(twinkle * 2.99).astype(int)]
        g = np.exp(-self.rr ** 2 * 3)[..., None]
        star_col = WHITE * (1 - g * 0.8) + PINK * (g * 0.8)
        is_neb = nb > 0.12
        bg_idx = np.where(is_neb, neb_idx, np.where(self.star_mask, star_idx, 0))
        bg_col = np.where(is_neb[..., None],
                          PINK * np.clip(0.4 + nb[..., None] * 1.5, 0, 1), star_col)

        idx = np.where(visible, idx, bg_idx)
        col = np.where(visible[..., None], col, bg_col)

        # turn character indexes into pixels
        alpha = atlas[idx].transpose(0, 2, 1, 3).reshape(self.H * CH, self.W * CW)
        rgb = np.repeat(np.repeat(col, CH, axis=0), CW, axis=1)
        pixels = np.clip(alpha[..., None] * rgb, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels)


def main():
    root = tk.Tk()
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
    if not path:
        return
    scene = Scene(path)

    root.title("Photo -> Space")
    root.configure(bg="black")
    label = tk.Label(root, bg="black", bd=0)
    label.pack()

    start = time.time()

    def update():
        t = (time.time() - start) % LOOP
        photo = ImageTk.PhotoImage(scene.frame(t))
        label.configure(image=photo)
        label.image = photo            # keep a reference, or Tk deletes it
        root.after(int(1000 / FPS), update)

    update()
    root.mainloop()


if __name__ == "__main__":
    main()
