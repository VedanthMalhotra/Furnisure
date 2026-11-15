"""Draws a small picture of each kit's finished model for the home page gallery.

Usage:  .venv/bin/python server/tools/make_thumbs.py [article_number ...]
Writes web/thumbs/<article_number>.webp. Run it again after adding or replacing a kit model.

The models are too large to load in the gallery (some are over 40 MB), so each one is drawn once here:
flat-shaded triangles, sorted back to front, with Pillow only.
"""
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "server"))
from geometry import kit_geometry  # noqa: E402

KITS = ROOT / "samples" / "ikea-kits"
OUT = ROOT / "web" / "thumbs"
WIDTH, HEIGHT, SCALE = 480, 320, 3          # drawn at 3x and scaled down for smooth edges
VIEW = (0.62, 0.42, 0.66)                   # the camera looks back along this, as in the player
LIGHT = (0.35, 0.85, 0.4)
BASE, DARK = (226, 205, 172), (120, 96, 66)  # lit and shaded tone of the model


def _unit(v):
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / length for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def draw(model_path, out_path):
    geometry = kit_geometry(model_path)
    forward = _unit(VIEW)
    right = _unit(_cross((0, 1, 0), forward))
    up = _cross(forward, right)
    light = _unit(LIGHT)

    faces, xs, ys = [], [], []
    for part in geometry["parts"].values():
        world = part["world"]
        flat = [(sum(p[i] * right[i] for i in range(3)), sum(p[i] * up[i] for i in range(3)),
                 sum(p[i] * forward[i] for i in range(3))) for p in world]
        xs += [p[0] for p in flat]
        ys += [p[1] for p in flat]
        for a, b, c in part["triangles"]:
            pa, pb, pc = world[a], world[b], world[c]
            normal = _cross(tuple(pb[i] - pa[i] for i in range(3)), tuple(pc[i] - pa[i] for i in range(3)))
            size = math.sqrt(sum(n * n for n in normal))
            if size == 0:
                continue
            facing = sum(normal[i] * forward[i] for i in range(3)) / size
            lit = abs(sum(normal[i] * light[i] for i in range(3)) / size)
            if facing < 0:                      # seen from behind: models are not always wound one way
                lit *= 0.8
            faces.append((flat[a][2] + flat[b][2] + flat[c][2], lit, flat[a], flat[b], flat[c]))
    if not faces:
        raise ValueError("the model has no triangles")

    w, h = WIDTH * SCALE, HEIGHT * SCALE
    low_x, high_x, low_y, high_y = min(xs), max(xs), min(ys), max(ys)
    zoom = 0.86 * min(w / (high_x - low_x or 1), h / (high_y - low_y or 1))
    mid_x, mid_y = (low_x + high_x) / 2, (low_y + high_y) / 2

    image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pen = ImageDraw.Draw(image)
    faces.sort(key=lambda f: f[0])              # farthest first
    for _, lit, a, b, c in faces:
        k = 0.25 + 0.75 * lit
        colour = tuple(round(DARK[i] + (BASE[i] - DARK[i]) * k) for i in range(3)) + (255,)
        pen.polygon([(w / 2 + (p[0] - mid_x) * zoom, h / 2 - (p[1] - mid_y) * zoom) for p in (a, b, c)], fill=colour)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((WIDTH, HEIGHT), Image.LANCZOS).save(out_path, "WEBP", quality=82)
    return len(faces)


def main():
    wanted = set(sys.argv[1:])
    for kit in json.loads((KITS / "catalog.json").read_text(encoding="utf-8")):
        number = kit["article_number"]
        if wanted and number not in wanted:
            continue
        try:
            count = draw(KITS / kit["model"], OUT / f"{number}.webp")
            print(f"{number}: {count} triangles")
        except Exception as error:               # one bad model should not stop the rest
            print(f"{number}: skipped ({error})")


if __name__ == "__main__":
    main()
