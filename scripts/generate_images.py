"""Generate professional course cover images (offline, using Pillow).

Each subject gets a themed illustration on a dark gradient. Files are written to
static/images/courses/<key>.jpg and a default.jpg used when a subject has no image.
Run:  python scripts/generate_images.py
"""
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "static", "images", "courses")
W, H, S = 800, 450, 2          # final size and supersampling factor

THEMES = {
    "dsa":     ((16, 185, 129), "tree"),
    "oop":     ((99, 102, 241), "classes"),
    "dbms":    ((14, 165, 233), "database"),
    "cn":      ((45, 212, 191), "network"),
    "os":      ((245, 158, 11), "chip"),
    "se":      ((168, 85, 247), "flow"),
    "aiml":    ((236, 72, 153), "neural"),
    "math":    ((250, 204, 21), "math"),
    "tc":      ((248, 113, 113), "chat"),
    "default": ((34, 197, 94), "book"),
}


def mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def background(accent):
    w, h = W * S, H * S
    base = Image.new("RGB", (w, h))
    px = base.load()
    top, bottom = (7, 18, 15), mix((7, 18, 15), accent, 0.28)
    for y in range(h):
        row = mix(top, bottom, y / h)
        for x in range(0, w):
            t = x / w * 0.25
            px[x, y] = mix(row, accent, t * 0.25)
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([w * 0.55, -h * 0.3, w * 1.2, h * 0.7], fill=accent + (110,))
    gd.ellipse([-w * 0.2, h * 0.55, w * 0.35, h * 1.3], fill=accent + (60,))
    glow = glow.filter(ImageFilter.GaussianBlur(90 * S))
    base = Image.alpha_composite(base.convert("RGBA"), glow)
    dots = ImageDraw.Draw(base)
    for x in range(0, w, 40 * S):
        for y in range(0, h, 40 * S):
            dots.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255, 22))
    return base


def line(d, p, q, col, width=3):
    d.line([p[0] * S, p[1] * S, q[0] * S, q[1] * S], fill=col, width=int(width * S))


def node(d, x, y, r, fill, outline=None, width=3):
    d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=fill,
              outline=outline, width=int(width * S))


def box(d, x1, y1, x2, y2, fill, outline=None, rad=10, width=3):
    d.rounded_rectangle([x1 * S, y1 * S, x2 * S, y2 * S], radius=rad * S, fill=fill,
                        outline=outline, width=int(width * S))


def draw_tree(d, a):
    pos = {0: (560, 110), 1: (460, 200), 2: (660, 200), 3: (410, 290), 4: (510, 290),
           5: (610, 290), 6: (710, 290), 7: (380, 370), 8: (440, 370)}
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8)]
    for i, j in edges:
        line(d, pos[i], pos[j], a + (170,), 4)
    for k, (x, y) in pos.items():
        node(d, x, y, 24, (10, 30, 24, 255), a + (255,), 4)
        node(d, x, y, 8, a + (255,))
    for k, (x, y) in {"a": (120, 300), "b": (170, 300), "c": (220, 300), "d": (270, 300)}.items():
        box(d, x - 22, y - 22, x + 22, y + 22, (255, 255, 255, 18), a + (200,), 8, 3)


def draw_classes(d, a):
    for (x, y, w, h) in [(470, 70, 220, 110), (380, 250, 190, 110), (610, 250, 190, 110)]:
        box(d, x, y, x + w, y + h, (255, 255, 255, 20), a + (230,), 14, 3)
        line(d, (x, y + 38), (x + w, y + 38), a + (230,), 3)
        for k in range(2):
            line(d, (x + 18, y + 62 + k * 22), (x + w - 30 - k * 30, y + 62 + k * 22), (255, 255, 255, 110), 4)
    line(d, (580, 180), (480, 250), a + (230,), 4)
    line(d, (580, 180), (700, 250), a + (230,), 4)
    d.polygon([(580 * S, 178 * S), (566 * S, 200 * S), (594 * S, 200 * S)], fill=(10, 30, 24, 255), outline=a + (255,))


def draw_database(d, a):
    for cx, cy in [(560, 120), (560, 220), (560, 320)]:
        d.rectangle([(cx - 110) * S, cy * S, (cx + 110) * S, (cy + 70) * S], fill=(12, 34, 30, 255))
        d.ellipse([(cx - 110) * S, (cy + 40) * S, (cx + 110) * S, (cy + 100) * S], fill=(12, 34, 30, 255), outline=a + (240,), width=4 * S)
        d.ellipse([(cx - 110) * S, (cy - 30) * S, (cx + 110) * S, (cy + 30) * S], fill=a + (90,), outline=a + (255,), width=4 * S)
        line(d, (cx - 110, cy), (cx - 110, cy + 70), a + (240,), 4)
        line(d, (cx + 110, cy), (cx + 110, cy + 70), a + (240,), 4)
        node(d, cx + 70, cy + 55, 6, a + (255,))
    for r in range(4):
        for c in range(3):
            box(d, 90 + c * 100, 120 + r * 60, 180 + c * 100, 165 + r * 60, (255, 255, 255, 16 if r else 60), a + (150,), 6, 2)


def draw_network(d, a):
    rnd = random.Random(7)
    pts = [(rnd.randint(380, 740), rnd.randint(50, 400)) for _ in range(14)]
    pts.append((560, 225))
    for i, p in enumerate(pts):
        for q in sorted(pts, key=lambda z: (z[0] - p[0]) ** 2 + (z[1] - p[1]) ** 2)[1:3]:
            line(d, p, q, a + (140,), 3)
    for (x, y) in pts:
        node(d, x, y, 13, (10, 30, 24, 255), a + (255,), 4)
        node(d, x, y, 5, a + (255,))
    node(d, 560, 225, 34, a + (70,), a + (255,), 5)
    for r in (60, 90):
        d.ellipse([(560 - r) * S, (225 - r) * S, (560 + r) * S, (225 + r) * S], outline=a + (70,), width=2 * S)


def draw_chip(d, a):
    box(d, 440, 90, 690, 340, (12, 34, 30, 255), a + (255,), 22, 5)
    box(d, 490, 140, 640, 290, a + (70,), a + (255,), 12, 4)
    for k in range(6):
        o = 470 + k * 38
        line(d, (o, 60), (o, 90), a + (220,), 6)
        line(d, (o, 340), (o, 370), a + (220,), 6)
        line(d, (410, 120 + k * 38), (440, 120 + k * 38), a + (220,), 6)
        line(d, (690, 120 + k * 38), (720, 120 + k * 38), a + (220,), 6)
    for k in range(3):
        line(d, (515, 185 + k * 30), (615, 185 + k * 30), (255, 255, 255, 140), 5)
    for k in range(3):
        box(d, 90 + k * 90, 300 - k * 40, 160 + k * 90, 340, (255, 255, 255, 18), a + (170,), 6, 3)


def draw_flow(d, a):
    steps = [(430, 80), (620, 80), (620, 190), (430, 190), (430, 300), (620, 300)]
    for (x, y) in steps:
        box(d, x, y, x + 140, y + 60, (255, 255, 255, 22), a + (240,), 12, 3)
        line(d, (x + 18, y + 30), (x + 100, y + 30), (255, 255, 255, 120), 4)
    for p, q in [((570, 110), (620, 110)), ((690, 140), (690, 190)), ((620, 220), (570, 220)), ((500, 250), (500, 300)), ((570, 330), (620, 330))]:
        line(d, p, q, a + (255,), 4)
    for k, w in enumerate([200, 140, 260, 100]):
        box(d, 70, 90 + k * 34, 70 + w, 112 + k * 34, a + (150,), None, 5)
    d.text((90 * S, 300 * S), "</>", fill=a + (230,))


def draw_neural(d, a):
    layers = [(430, 3), (530, 5), (630, 5), (730, 2)]
    coords = []
    for x, n in layers:
        gap = 300 / (n + 1)
        coords.append([(x, 75 + gap * (i + 1)) for i in range(n)])
    for L1, L2 in zip(coords, coords[1:]):
        for p in L1:
            for q in L2:
                line(d, p, q, a + (75,), 2)
    for L in coords:
        for (x, y) in L:
            node(d, x, y, 15, (10, 30, 24, 255), a + (255,), 4)
            node(d, x, y, 6, a + (255,))
    for k in range(5):
        box(d, 80 + k * 16, 230 - (k * 25 if k < 3 else (4 - k) * 25), 92 + k * 16, 300, a + (170,), None, 3)


def draw_math(d, a):
    line(d, (400, 380), (760, 380), (255, 255, 255, 150), 3)
    line(d, (430, 60), (430, 400), (255, 255, 255, 150), 3)
    pts = [(430 + i * 4, 230 - 90 * math.sin(i / 14.0)) for i in range(0, 84)]
    for i in range(len(pts) - 1):
        line(d, pts[i], pts[i + 1], a + (255,), 5)
    for (x, y) in pts[::4]:
        line(d, (x, y), (x, 230), a + (60,), 3)
    box(d, 70, 100, 100, 320, None, a + (200,), 4, 4)
    box(d, 250, 100, 280, 320, None, a + (200,), 4, 4)
    for r in range(3):
        for c in range(3):
            d.text(((115 + c * 46) * S, (130 + r * 60) * S), str(r * 3 + c + 1), fill=a + (255,))
            node(d, 128 + c * 46, 150 + r * 60, 10, a + (200,))


def draw_chat(d, a):
    box(d, 420, 70, 690, 170, a + (80,), a + (255,), 22, 4)
    box(d, 500, 200, 760, 300, (255, 255, 255, 25), a + (200,), 22, 4)
    for k in range(3):
        line(d, (450, 100 + k * 22), (660 - k * 40, 100 + k * 22), (255, 255, 255, 150), 5)
        line(d, (530, 230 + k * 22), (730 - k * 40, 230 + k * 22), (255, 255, 255, 120), 5)
    d.polygon([(450 * S, 170 * S), (480 * S, 170 * S), (450 * S, 200 * S)], fill=a + (80,), outline=a + (255,))
    for k in range(4):
        box(d, 80, 100 + k * 55, 330 - k * 30, 130 + k * 55, a + (60 + k * 20,), None, 6)


def draw_book(d, a):
    box(d, 430, 110, 590, 340, (12, 34, 30, 255), a + (255,), 12, 5)
    box(d, 590, 110, 750, 340, (12, 34, 30, 255), a + (255,), 12, 5)
    for k in range(6):
        line(d, (450, 150 + k * 28), (570, 150 + k * 28), a + (160,), 4)
        line(d, (610, 150 + k * 28), (730, 150 + k * 28), a + (160,), 4)
    node(d, 250, 225, 70, a + (60,), a + (255,), 5)
    d.polygon([(250 * S, 175 * S), (300 * S, 210 * S), (250 * S, 245 * S), (200 * S, 210 * S)], outline=a + (255,), width=4 * S)


DRAWERS = {"tree": draw_tree, "classes": draw_classes, "database": draw_database, "network": draw_network,
           "chip": draw_chip, "flow": draw_flow, "neural": draw_neural, "math": draw_math,
           "chat": draw_chat, "book": draw_book}


def render(key):
    accent, motif = THEMES[key]
    img = background(accent)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    DRAWERS[motif](ImageDraw.Draw(layer, "RGBA"), accent)
    img = Image.alpha_composite(img, layer)
    return img.convert("RGB").resize((W, H), Image.LANCZOS)


def main():
    os.makedirs(OUT, exist_ok=True)
    for key in THEMES:
        render(key).save(os.path.join(OUT, f"{key}.jpg"), quality=88, optimize=True)
        print("wrote", key)


if __name__ == "__main__":
    main()
