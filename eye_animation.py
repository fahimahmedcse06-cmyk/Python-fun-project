import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ---------- Settings ----------
W, H = 110, 48          # character columns and rows
CW, CH = 7, 14          # pixel size of one character cell
FONT_SIZE = 12
FPS = 30
LOOP = 15.0             # animation length in seconds
CHARS = " .:-=+*#%@"    # dark -> bright
STAR_IDX = np.array([1, 5, 6])   # indexes of "." "+" "*" inside CHARS

# ---------- Colors ----------
DARK = np.array([95, 38, 16], np.float32)
ORANGE = np.array([255, 140, 52], np.float32)
LIGHT = np.array([255, 225, 170], np.float32)
RED = np.array([175, 40, 14], np.float32)
GOLD = np.array([255, 185, 70], np.float32)
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


# ---------- Glyph atlas: one small bitmap per character ----------
font = load_font(FONT_SIZE)
atlas = np.zeros((len(CHARS), CH, CW), dtype=np.float32)
for i, c in enumerate(CHARS):
    img = Image.new("L", (CW, CH), 0)
    ImageDraw.Draw(img).text((0, 0), c, fill=255, font=font)
    atlas[i] = np.asarray(img, dtype=np.float32) / 255

# ---------- Coordinates (computed once) ----------
ys, xs = np.mgrid[0:H, 0:W]
R = 42.0
x = (xs - W / 2) / R
y = (ys - H / 2) * 2.0 / R        # characters are ~2x taller than wide
r = np.sqrt(x ** 2 + y ** 2)      # distance from center
angle = np.arctan2(y, x)          # angle around center
z = np.sqrt(np.clip(1 - r ** 2, 0, 1))

rng = np.random.default_rng(7)
noise = rng.random((H, W))
star_mask = rng.random((H, W)) < 0.02
star_phase = rng.random((H, W)) * 6.28
pupil_sparkle = rng.random((H, W)) < 0.14
nebula_noise = rng.random((H, W))


def lerp(a, b, t):
    return a + (b - a) * t[..., None]


def eye_brightness(t):
    light = np.clip(-0.4 * x - 0.5 * y + 0.8 * z, 0, 1)
    b = 0.25 + 0.75 * light                                # white of the eye

    streaks = 0.5 + 0.5 * np.sin(angle * 18)
    b = np.where(r < 0.55, 0.4 + 0.45 * streaks, b)        # bright iris
    b = np.where((r > 0.49) & (r < 0.57), 0.12, b)         # dark ring

    pupil = 0.18 + 0.04 * np.sin(t * 2)                    # pupil breathes
    b = np.where(r < pupil, 0.02, b)

    highlight = np.exp(-((x + 0.2) ** 2 + (y + 0.2) ** 2) / 0.01)
    return np.clip(b + highlight, 0, 1)


def make_frame(t):
    # 0 = nothing visible, 1.05 = full eye visible
    level = np.interp(t, [0, 4, 9, 13, 15], [0, 1.05, 1.05, 0, 0])
    show_eye = (r < 1) & ((0.6 * noise + 0.4 * r) < level)

    # --- eye layer ---
    b = eye_brightness(t)
    idx = (b * (len(CHARS) - 1)).astype(int)
    col = np.where((b < 0.5)[..., None],
                   lerp(DARK, ORANGE, np.clip(b / 0.5, 0, 1)),
                   lerp(ORANGE, LIGHT, np.clip((b - 0.5) / 0.5, 0, 1)))

    iris = (r < 0.55) & (r > 0.2)
    col = np.where(iris[..., None],
                   lerp(RED, GOLD, np.clip((b - 0.4) / 0.45, 0, 1)), col)

    # tiny stars inside the pupil
    pupil = 0.18 + 0.04 * np.sin(t * 2)
    inside = (r < pupil + 0.02) & pupil_sparkle
    twinkle = (np.sin(t * 4 + star_phase) + 1) / 2
    idx = np.where(inside, STAR_IDX[(twinkle * 2.99).astype(int)], idx)
    col = np.where(inside[..., None], WHITE, col)

    # --- background layer: pink glow + twinkling stars ---
    glow = np.exp(-(r / 0.5) ** 2) * (1 - min(level / 1.05, 1))
    nb = glow * (0.4 + 0.6 * nebula_noise)
    neb_idx = np.clip((nb * 6).astype(int), 0, 5)
    star_tw = (np.sin(t * 3 + star_phase) + 1) / 2
    star_idx = STAR_IDX[(star_tw * 2.99).astype(int)]
    g = np.exp(-r ** 2 * 3)[..., None]
    star_col = WHITE * (1 - g * 0.8) + PINK * (g * 0.8)

    is_neb = nb > 0.12
    bg_idx = np.where(is_neb, neb_idx, np.where(star_mask, star_idx, 0))
    bg_col = np.where(is_neb[..., None],
                      PINK * np.clip(0.4 + nb[..., None] * 1.5, 0, 1),
                      star_col)

    # --- combine layers ---
    idx = np.where(show_eye, idx, bg_idx)
    col = np.where(show_eye[..., None], col, bg_col)

    # --- turn character indexes into pixels ---
    alpha = atlas[idx].transpose(0, 2, 1, 3).reshape(H * CH, W * CW)
    rgb = np.repeat(np.repeat(col, CH, axis=0), CW, axis=1)
    pixels = np.clip(alpha[..., None] * rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def main():
    root = tk.Tk()
    root.title("Eye -> Space")
    root.configure(bg="black")

    label = tk.Label(root, bg="black", bd=0)
    label.pack()

    start = time.time()

    def update():
        t = (time.time() - start) % LOOP
        photo = ImageTk.PhotoImage(make_frame(t))
        label.configure(image=photo)
        label.image = photo            # keep a reference, or Tk deletes it
        root.after(int(1000 / FPS), update)

    update()
    root.mainloop()


if __name__ == "__main__":
    main()
