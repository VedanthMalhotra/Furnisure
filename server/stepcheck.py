"""Checks a LiDAR scan of half-built furniture against the kit model after one step.

The phone places a see-through ghost of the expected state over the real furniture, walks around
it, and sends the scanned points plus that ghost placement. This module:

  ghost - a light mesh of the parts expected after a step, for the phone to draw
  check - lines the scan up with the model (starting from the ghost placement), then looks part by
          part: a board with no scan where it belongs is missing, and scanned shapes where no part
          should be yet are out of place. Screws are too small for LiDAR, so one photo goes to the
          vision model for them.

Everything here works in the kit's model frame (glTF, Y up). `ghost` from the phone maps model units
to ARKit world metres and may include a scale, since the IKEA-Manual kits are not at real size.
"""
import base64
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import ndimage
from scipy.spatial import cKDTree

from kits import ROOT, geometry, get_kit

router = APIRouter(prefix="/api/stepcheck")

SPACING = 0.006      # metres between surface samples on the model
FLOOR = 0.012        # scanned points this close to the floor are floor, not furniture
FLAT = 0.04          # a step whose parts stand lower than this is too flat for the scan
TOLERANCE = 0.02     # a model sample counts as scanned when a scan point is this close (metres)
CONTACT = 0.008      # model samples this close to another part are hidden in a joint
DISTINCT = 0.03     # a part is judged by its surface at least this far from every other part
EXTRA = 0.035       # scan points farther than this from every expected part are unexplained
GHOST_TRIANGLES = 6000
SAVED = ROOT.parent.parent / "server" / "jobs" / "scans"
ADVICE_SECONDS = 50  # Gemini takes 4 to 40 s; the phone waits 90 s for the whole answer


def metres_per_unit(kit):
    """1 for kits modelled at real size; None when the model's scale is unknown (IKEA-Manual kits)."""
    if kit.get("units") == "m" or not kit["article_number"].startswith("im-"):
        return 1.0
    return None


@lru_cache(maxsize=8)
def _arrays(article):
    """Each part's vertices and triangles as numpy arrays, in model units."""
    g = geometry(get_kit(article))
    parts = {}
    for pid, part in g["parts"].items():
        v = np.asarray(part["world"], dtype=np.float64)
        t = np.asarray(part["triangles"], dtype=np.int64).reshape(-1, 3)
        parts[pid] = {"v": v, "t": t, "name": part["name"], "fastener": part["fastener"],
                      "lo": np.asarray(part["lo"]), "hi": np.asarray(part["hi"])}
    return parts, g["diagonal"]


def _simplify(v, t, cell):
    """Vertex clustering: merges vertices that share a grid cell, drops collapsed triangles."""
    if len(t) == 0:
        return v, t
    keys = np.floor((v - v.min(0)) / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    count = inverse.max() + 1
    merged = np.zeros((count, 3))
    np.add.at(merged, inverse, v)
    merged /= np.bincount(inverse, minlength=count)[:, None]
    tri = inverse[t]
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = np.unique(np.sort(tri[keep], axis=1), axis=0) if keep.any() else tri[:0]
    # sorting lost the winding; the ghost is drawn double-sided, so that does not matter
    return merged, tri


@lru_cache(maxsize=64)
def _ghost_part(article, pid):
    parts, diagonal = _arrays(article)
    part = parts[pid]
    v, t = part["v"], part["t"]
    size = float((part["hi"] - part["lo"]).max())
    cell = min(diagonal / 160, size / 10)
    while len(t) > GHOST_TRIANGLES / 2:
        v, t = _simplify(v, t, cell)
        cell *= 1.6
    return v.astype(np.float32), t.astype(np.uint32)


@router.get("/{article}/ghost")
def ghost(article: str, nodes: str, active: str = ""):
    """Mesh of the given parts at their finished positions. Positions and indices are base64
    little-endian float32 / uint32, so a phone can hand them straight to a GPU buffer."""
    parts, diagonal = _arrays(article)
    wanted = [int(n) for n in nodes.split(",") if n.strip()]
    hot = {int(n) for n in active.split(",") if n.strip()}
    out = []
    for pid in wanted:
        if pid not in parts:
            continue
        v, t = _ghost_part(article, pid)
        out.append({"id": pid, "name": parts[pid]["name"], "fastener": bool(parts[pid]["fastener"]),
                    "active": pid in hot,
                    "positions": base64.b64encode(v.tobytes()).decode(),
                    "indices": base64.b64encode(t.tobytes()).decode()})
    all_lo = np.min([p["lo"] for p in parts.values()], axis=0)
    all_hi = np.max([p["hi"] for p in parts.values()], axis=0)
    return {"parts": out, "lo": all_lo.tolist(), "hi": all_hi.tolist(),
            "metres_per_unit": metres_per_unit(get_kit(article)), "diagonal": diagonal}


# ---------- sampling the model ----------

@lru_cache(maxsize=64)
def _samples(article, pid, spacing):
    """Points spread evenly over a part's surface (area weighted), in model units."""
    parts, _ = _arrays(article)
    v, t = parts[pid]["v"], parts[pid]["t"]
    if len(t) == 0:
        return np.zeros((0, 3))
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        return np.zeros((0, 3))
    n = int(np.clip(total / spacing ** 2, 80, 6000))
    rng = np.random.default_rng(pid)
    pick = rng.choice(len(t), size=n, p=area / total)
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[pick] + r1[:, None] * (b[pick] - a[pick]) + r2[:, None] * (c[pick] - a[pick])


# ---------- alignment ----------

def _umeyama(src, dst, with_scale):
    """Similarity (s, R, t) that best maps src onto dst in the least-squares sense."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    s = (d * np.diag(sign)).sum() / xs.var(0).sum() if with_scale else 1.0
    return s, r, mu_d - s * r @ mu_s


def _umeyama_upright(src, dst, axis):
    """Like _umeyama with scale, but the rotation only turns about `axis` (the world's up), so the
    furniture can never be tipped over by the fit."""
    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    a, b, c = xs @ e1, xs @ e2, xs @ axis
    p, q, w = xd @ e1, xd @ e2, xd @ axis
    C, S = float((p * a + q * b).sum()), float((q * a - p * b).sum())
    angle = math.atan2(S, C)
    r = _turn(axis, angle)
    s = (C * math.cos(angle) + S * math.sin(angle) + float((w * c).sum())) / float((xs ** 2).sum())
    return s, r, mu_d - s * r @ mu_s


def _icp(scan, target, unit, with_scale, tree=None, start_gate=0.25, rounds=36, axis=None, sizes=(0.5, 2.0)):
    """Moves scan (model units) onto target samples. Returns the 4x4 similarity and the fit.
    `unit` is metres per model unit at the ghost's scale, so the distance gates stay in metres.
    With `axis`, the scan only turns about that axis (it stays standing the way it was)."""
    tree = tree or cKDTree(target)
    total = np.eye(4)
    moved = scan.copy()
    gates = np.geomspace(start_gate, 0.015, rounds) / unit   # recovers a ghost placed up to ~25 cm off
    rmse, inliers = float("inf"), 0
    size = 1.0
    for gate in list(gates) + [gates[-1]] * 10:
        d, idx = tree.query(moved, distance_upper_bound=gate, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        if axis is not None:
            s, r, t = _umeyama_upright(moved[ok], target[idx[ok]], axis)
        else:
            s, r, t = _umeyama(moved[ok], target[idx[ok]], with_scale)
        # size may only drift a little per round, and at most halve or double overall
        s = float(np.clip(s, 0.97, 1.03))
        s = float(np.clip(size * s, *sizes) / size)
        size *= s
        step = np.eye(4)
        step[:3, :3] = s * r
        step[:3, 3] = t
        moved = moved @ (s * r).T + t
        total = step @ total
        rmse, inliers = float(np.sqrt((d[ok] ** 2).mean())) * unit, int(ok.sum())
        if gate <= gates[-1] and np.abs(step - np.eye(4)).max() < 1e-5:
            break
    return total, moved, rmse, inliers


def _turn(axis, angle):
    """Rotation matrix about a unit axis."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


def _onto(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        other = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        return _turn(np.cross(a, other) / np.linalg.norm(np.cross(a, other)), math.pi)
    return _turn(v / np.linalg.norm(v), math.atan2(np.linalg.norm(v), c))


def _match(points, target_tree, target, tol):
    """How well points sit on the model: share of points on it, share of it covered, and their F1."""
    d, _ = target_tree.query(points, distance_upper_bound=tol, workers=-1)
    precision = float(np.isfinite(d).mean())
    d, _ = cKDTree(points).query(target, distance_upper_bound=tol, workers=-1)
    recall = float(np.isfinite(d).mean())
    return precision, recall, 2 * precision * recall / (precision + recall + 1e-9)


class _Grid:
    """Fast rough version of _match for the coarse search: points and model fall into cubes of side
    `cell`; a point counts as on the model when its cube or a neighbouring one holds model."""
    def __init__(self, model, cell):
        self.cell, self.lo = cell, model.min(0) - 3 * cell
        self.shape = tuple(np.ceil((model.max(0) + 3 * cell - self.lo) / cell).astype(int) + 1)
        self.model_idx = self._index(model)[0]
        occupied = np.zeros(self.shape, dtype=bool)
        occupied[tuple(self.model_idx.T)] = True
        self.near_model = ndimage.binary_dilation(occupied, np.ones((3, 3, 3)))

    def _index(self, points):
        idx = np.floor((points - self.lo) / self.cell).astype(int)
        inside = np.all((idx >= 0) & (idx < self.shape), axis=1)
        return idx[inside], inside

    def score(self, points):
        idx, inside = self._index(points)
        precision = float(self.near_model[tuple(idx.T)].sum()) / max(len(points), 1)
        hit = np.zeros(self.shape, dtype=bool)
        hit[tuple(idx.T)] = True
        recall = float(ndimage.binary_dilation(hit, np.ones((3, 3, 3)))[tuple(self.model_idx.T)].mean())
        return precision, recall, 2 * precision * recall / (precision + recall + 1e-9)


def _footprint_precision(points, model, tree, tol, margin):
    """Share of the scan points over the model's own footprint that lie on it. Walls and objects
    farther away do not count against the fit."""
    lo, hi = model.min(0) - margin, model.max(0) + margin
    inside = points[np.all((points > lo) & (points < hi), axis=1)]
    if len(inside) < 50:
        return 0.0, int(len(inside))
    d, _ = tree.query(inside, distance_upper_bound=tol, workers=-1)
    return float(np.isfinite(d).mean()), int(len(inside))


def _fit(scan, target, unit, up, pivot, near=False, sizes=(0.5, 2.0)):
    """Finds how the furniture sits under the ghost: turned any way about the world's up, up to 40%
    bigger or smaller, and up to ~20 cm off. It is never tipped over: it stands the way the ghost
    stands (the person tips the ghost for furniture built lying down). `scan` is in model units as the
    ghost placed it, `up` the world's up in that frame, `pivot` a floor point under the ghost's middle.
    `sizes` bounds how much bigger the model may get than the scan (scan-to-model scale).
    With `near`, only small changes are tried: the ghost already sits about right (live updates).
    Returns the 4x4 move of the scan onto the model, the moved scan, rmse (m), inliers and a summary."""
    rng = np.random.default_rng(0)
    pick = lambda pts, n: pts[rng.choice(len(pts), min(len(pts), n), replace=False)]
    probe, model = pick(scan, 1500), pick(target, 3000)
    target_tree = cKDTree(target)
    tree = cKDTree(model)
    grid = _Grid(target, 0.04 / unit)
    tol = 0.03 / unit
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    step = 0.15 / unit
    shifts = [np.zeros(3), step * e1, -step * e1, step * e2, -step * e2]
    centre = target.mean(0)
    trials = []
    if near:
        shifts = shifts[:1]
    for yaw in (range(-30, 31, 15) if near else range(0, 360, 15)):
        r = _turn(up, math.radians(yaw))
        tries = (0.92, 1.0, 1.08) if near else (0.7, 0.85, 1.0, 1.18, 1.4)
        for size in sorted({float(np.clip(z, *sizes)) for z in tries}):
            q0 = size * (probe - pivot) @ r.T + pivot
            # also try the scan's middle on the model's middle, for a ghost dragged well off
            centred = centre - q0.mean(0)
            centred -= up * (centred @ up)
            for shift in shifts + [centred]:
                f = grid.score(q0 + shift)[2]
                start = np.eye(4)
                start[:3, :3] = size * r
                start[:3, 3] = pivot - size * r @ pivot + shift
                trials.append((f, yaw, size, start))
    trials.sort(key=lambda x: -x[0])
    # refine the most promising few on a sample, then the winner on the whole scan; on a near tie
    # the turn closest to how the person placed the ghost wins (symmetric furniture fits both ways)
    sample = pick(scan, 2000)
    best, seen = None, []
    for _, yaw, size, start in trials:
        if len(seen) >= 8:
            break
        if any(abs((yaw - y + 180) % 360 - 180) < 10 and abs(size / z - 1) < 0.1 for y, z in seen):
            continue
        seen.append((yaw, size))
        moved = sample @ start[:3, :3].T + start[:3, 3]
        icp = _icp(moved, target, unit / size, True, tree=target_tree, start_gate=0.12, rounds=14, axis=up,
                   sizes=(sizes[0] / size, sizes[1] / size))[0] @ start
        aligned = sample @ icp[:3, :3].T + icp[:3, 3]
        f = _match(pick(aligned, 2000), tree, model, tol)[2]
        rot = icp[:3, :3]
        turned = math.degrees(math.atan2(e2 @ rot @ e1, e1 @ rot @ e1))
        f -= 0.02 * abs(turned) / 180
        if best is None or f > best[0]:
            best = (f, icp)
    _, start = best
    size = float(np.cbrt(abs(np.linalg.det(start[:3, :3]))))
    moved = scan @ start[:3, :3].T + start[:3, 3]
    icp, aligned, rmse, inliers = _icp(moved, target, unit / size, True, tree=target_tree, start_gate=0.05, rounds=10, axis=up,
                                       sizes=(sizes[0] / size, sizes[1] / size))
    total = icp @ start
    size = float(np.cbrt(abs(np.linalg.det(total[:3, :3]))))
    precision, near = _footprint_precision(aligned, target, target_tree, tol, 0.1 / unit)
    recall = _match(pick(aligned, 4000), tree, model, tol)[1]
    rot = total[:3, :3] / size
    yaw = math.degrees(math.atan2(e2 @ rot @ e1, e1 @ rot @ e1))
    summary = {"precision": round(precision, 3), "recall": round(recall, 3), "near": near,
               "yaw": round(yaw), "size": round(size, 3)}
    return total, aligned, rmse, inliers, summary


# ---------- the check ----------

class Hardware(BaseModel):
    name: str
    count: int = 1


class CheckIn(BaseModel):
    step: str
    captions: list[str] = Field(default_factory=list, description="What the manual says to do in this step")
    installed: list[int] = Field(description="Parts fitted in earlier steps")
    active: list[int] = Field(description="Parts fitted in this step")
    future: list[int] = Field(default_factory=list, description="Parts that come in later steps")
    hardware: list[Hardware] = Field(default_factory=list, description="Screws, dowels etc. used in this step")
    ghost: list[float] = Field(description="Model-to-world 4x4, column-major, metres, from the ghost placement")
    points: str = Field(description="base64 float32 x,y,z world points in metres")
    photo: Optional[str] = Field(None, description="base64 JPEG of the furniture")
    fitted: bool = Field(False, description="The live fit already moved the ghost onto the furniture")


class FitIn(BaseModel):
    installed: list[int]
    active: list[int]
    near: bool = Field(False, description="The ghost already sits about right: only small changes are tried")
    ghost: list[float] = Field(description="Model-to-world 4x4, column-major, metres, from the ghost placement")
    points: str = Field(description="base64 float32 x,y,z world points in metres, the scan so far")


def _matrix(values):
    if len(values) != 16:
        raise HTTPException(400, "ghost must be 16 numbers")
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T   # column-major in


def _thin(part, unit):
    """Rods, dowels and narrow rails: too thin for the LiDAR scan to judge reliably (metres)."""
    extent = np.sort(part["hi"] - part["lo"]) * unit
    return bool(extent[1] < 0.04)


def _prepare(article, installed, active, ghost, points, min_points=300):
    """Decodes the scan and finds the floor and the part of the scan around the ghost."""
    parts, _ = _arrays(article)
    raw = np.frombuffer(base64.b64decode(points), dtype="<f4")
    if raw.size < min_points or raw.size % 3:
        raise HTTPException(400, "Too few scanned points. Walk all the way around the furniture and scan again.")
    world = raw.reshape(-1, 3).astype(np.float64)
    ghost_m = _matrix(ghost)
    unit = float(np.cbrt(abs(np.linalg.det(ghost_m[:3, :3]))))   # metres per model unit at the ghost's scale
    to_model = np.linalg.inv(ghost_m)
    expected = [p for p in dict.fromkeys(installed + active) if p in parts]
    solid = [p for p in expected if not parts[p]["fastener"]]
    if not solid:
        raise HTTPException(400, "This step has no parts large enough to scan.")
    spacing = SPACING / unit
    samples = {p: _samples(article, p, spacing) for p in solid}
    target = np.concatenate([samples[p] for p in solid])
    target_world = target @ ghost_m[:3, :3].T + ghost_m[:3, 3]
    floor_y = _floor(world, target_world)
    height = float(target_world[:, 1].max() - target_world[:, 1].min())
    # Keep what is within reach of the ghost in any direction across the floor, so a ghost turned
    # the wrong way does not cut off part of the furniture (a box around the ghost would)
    centre = target_world.mean(0)
    reach = np.linalg.norm(target_world[:, [0, 2]] - centre[[0, 2]], axis=1).max()
    across = np.linalg.norm(world[:, [0, 2]] - centre[[0, 2]], axis=1)
    near = (world[:, 1] > floor_y + FLOOR) & (across < 1.5 * reach + 0.3) & (world[:, 1] < floor_y + 2 * height + 0.3)
    scan = (world @ to_model[:3, :3].T + to_model[:3, 3])[near]
    if len(scan) > 60000:
        scan = scan[np.random.default_rng(0).choice(len(scan), 60000, replace=False)]
    up = to_model[:3, :3] @ np.array([0.0, 1.0, 0.0])
    up /= np.linalg.norm(up)
    pivot = to_model[:3, :3] @ np.array([centre[0], floor_y, centre[2]]) + to_model[:3, 3]
    # Kits modelled at real size only get a little resizing (missing feet make furniture look shorter,
    # which must not shrink the model); kits of unknown size may be any size.
    known = metres_per_unit(get_kit(article))
    sizes = (unit / (1.15 * known), unit / (0.87 * known)) if known else (0.5, 2.0)
    return dict(sizes=sizes, parts=parts, expected=expected, solid=solid, samples=samples, target=target, spacing=spacing,
                ghost_m=ghost_m, unit=unit, to_model=to_model, floor_y=floor_y, height=height, scan=scan,
                up=up, pivot=pivot, world_points=len(world))


GOOD_FIT = 0.6   # share of the scan over the model's footprint that must lie on the model
GOOD_COVER = 0.35  # and share of the model the scan must cover (real LiDAR: right fits 0.65-0.9, wrong ones under 0.5)


@router.post("/{article}/fit")
def live_fit(article: str, body: FitIn):
    """Where the model sits on the scan so far, so the phone can slide, turn and resize the ghost onto
    the real furniture while the person walks around it. `transform` is null until the fit is good."""
    started = time.time()
    c = _prepare(article, body.installed, body.active, body.ghost, body.points, min_points=150)
    if len(c["scan"]) < 150 or c["height"] < FLAT:
        return {"transform": None, "points": len(c["scan"])}
    icp, _, rmse, inliers, match = _fit(c["scan"], c["target"], c["unit"], c["up"], c["pivot"], near=body.near, sizes=c["sizes"])
    good = match["precision"] >= GOOD_FIT and match["recall"] >= GOOD_COVER and inliers >= 100
    print(f"[stepcheck] live fit {time.time() - started:.1f}s: {len(c['scan'])} pts, {match}, "
          f"{'moving the ghost' if good else 'not sure yet'}", flush=True)
    transform = c["ghost_m"] @ np.linalg.inv(icp)
    return {"transform": transform.T.reshape(-1).tolist() if good else None, **match}


def check(article: str, body: CheckIn) -> dict:
    kit = get_kit(article)
    c = _prepare(article, body.installed, body.active, body.ghost, body.points)
    parts, expected, solid, samples, target = c["parts"], c["expected"], c["solid"], c["samples"], c["target"]
    unit, ghost_m, scan, spacing = c["unit"], c["ghost_m"], c["scan"], c["spacing"]

    def all_unsure(message):
        return [{"id": p, "name": parts[p]["name"], "active": p in body.active, "status": "unsure", "coverage": None,
                 "message": message} for p in expected]

    if c["height"] < FLAT:
        return {"verdict": "unsure", "scan": "flat", "summary": "At this step everything lies nearly flat on the floor, "
                "so the 3D scan can't check it; the photo check decides.",
                "parts": all_unsure("Lies too flat on the floor for the 3D scan; checked from the photo."),
                "extras": [], "transform": body.ghost, "fit": {"height_m": round(c["height"], 3)}}
    if len(scan) < 200:
        return {"verdict": "unsure", "scan": "empty", "summary": "The 3D scan barely picked up the furniture, so the "
                "photo check decides. For a better scan, walk slowly all the way around, about an arm's length away.",
                "parts": all_unsure("Not checked by the 3D scan; checked from the photo."),
                "extras": [], "transform": body.ghost, "fit": {"points": int(len(scan))}}

    # a ghost the live fit already put on the furniture only needs small corrections; a wide search
    # over a full scan full of walls and clutter can wander off to a wrong fit
    icp, aligned, rmse, inliers, match = _fit(scan, target, unit, c["up"], c["pivot"], near=body.fitted, sizes=c["sizes"])
    if body.fitted and (match["precision"] < GOOD_FIT or match["recall"] < GOOD_COVER):
        wide = _fit(scan, target, unit, c["up"], c["pivot"], sizes=c["sizes"])
        if wide[4]["precision"] > match["precision"]:
            icp, aligned, rmse, inliers, match = wide
    print(f"[stepcheck] best fit: {match['precision']:.0%} of the scan over the model on it, {match['recall']:.0%} of "
          f"the model scanned, turned {match['yaw']} deg, size x{match['size']:.2f}", flush=True)
    if match["precision"] < GOOD_FIT or match["recall"] < GOOD_COVER or not np.isfinite(rmse):
        return {"verdict": "unsure", "scan": "no_fit", "summary": "The 3D scan couldn't be lined up with the model, so "
                "the photo check decides.",
                "parts": all_unsure("Not checked by the 3D scan; checked from the photo."), "extras": [],
                "transform": body.ghost, "fit": {"points": int(len(scan)), "inliers": inliers, **match}}
    scale = float(np.cbrt(abs(np.linalg.det(icp[:3, :3]))))
    unit_after = unit / scale          # metres per model unit after the fit
    transform = ghost_m @ np.linalg.inv(icp)
    up_fit = icp[:3, :3] @ c["up"]
    up_fit /= np.linalg.norm(up_fit)

    # which model samples the scan covers, ignoring ones under or on the real floor, or inside joints
    labels = np.concatenate([np.full(len(samples[p]), i) for i, p in enumerate(solid)])
    model_tree = cKDTree(target)
    rise = target @ up_fit
    floor_level = float((icp[:3, :3] @ c["pivot"] + icp[:3, 3]) @ up_fit)
    hidden = rise < floor_level + 0.01 / unit_after
    for i in range(len(solid)):   # samples touching another part sit in a joint
        mine, others = labels == i, labels != i
        if others.any():
            d, _ = cKDTree(target[others]).query(target[mine], distance_upper_bound=CONTACT / unit_after)
            hidden[np.flatnonzero(mine)[np.isfinite(d)]] = True
    # Each scan point belongs to the part it lies closest to, later parts included. Otherwise a
    # missing rail would look covered by the scan of the boards it touches.
    future = [p for p in body.future if p in parts and not parts[p]["fastener"] and p not in solid]
    later = [_samples(article, p, spacing) for p in future]
    owners_tree = cKDTree(np.concatenate([target] + later)) if later else model_tree
    owner_labels = np.concatenate([labels] + [np.full(len(s), len(solid) + k) for k, s in enumerate(later)])
    _, nearest_sample = owners_tree.query(aligned)
    owner = owner_labels[nearest_sample]

    everything = np.concatenate([target] + later)
    results = []
    for i, p in enumerate(solid):
        mine = labels == i
        visible = mine & ~hidden
        # Judge a part by its own surface, away from every other part: near a joint, the scan of a
        # neighbour (or of a slot it sits in) can pass for this part even when it is missing.
        others = everything[owner_labels != i]
        if len(others):
            gap, _ = cKDTree(others).query(target[mine])
            for clearance in (DISTINCT, 0.02, 0.013):   # tucked-in rails have little clear surface
                distinct = visible.copy()
                distinct[mine] &= gap > clearance / unit_after
                if distinct.sum() >= 30:
                    visible = distinct
                    break
        nv = int(visible.sum())
        own = aligned[owner == i]
        if nv and len(own):
            d, _ = cKDTree(own).query(target[visible], distance_upper_bound=TOLERANCE / unit_after)
            coverage = float(np.isfinite(d).mean())
        else:
            coverage = 0.0
        results.append({"id": p, "name": parts[p]["name"], "active": p in body.active, "thin": _thin(parts[p], unit_after),
                        "visible_samples": nv, "coverage": round(coverage, 3)})
    measured = [r["coverage"] for r in results if r["visible_samples"] >= 30 and not r["thin"]] or \
               [r["coverage"] for r in results if r["visible_samples"] >= 30]
    reference = float(np.median(measured)) if measured else 0.0
    for r in results:
        if r["visible_samples"] < 30:
            r["status"], r["message"] = "unsure", "Mostly hidden from the 3D scan; checked from the photo."
        elif r["coverage"] >= max(0.25, 0.45 * reference):
            r["status"], r["message"] = "ok", "In place."
        elif r["thin"]:
            # LiDAR often misses rods and dowels that are there, so only the photo may call them missing
            r["status"], r["message"] = "unsure", "Too thin for the 3D scan to be sure; checked from the photo."
        elif r["coverage"] < max(0.1, 0.3 * reference):   # scan noise lends a missing part some coverage from its neighbours
            r["status"] = "missing"
            r["message"] = ("Nothing was scanned where this part goes. It may be missing, in the wrong place, "
                            "or not pushed all the way in.")
        else:
            r["status"] = "unsure"
            r["message"] = "Only partly picked up by the scan. Check it is fully seated, or scan this side again."
    # Slanted or shifted boards: fit a plane to each board's own scan and compare it with the model.
    # On real scans boards that are on right measure up to ~8 deg and ~2.5 cm off, so only more counts.
    seen_boards = [int((owner == i).sum()) for i, r in enumerate(results[:len(solid)]) if r["status"] == "ok"]
    board_points = float(np.median(seen_boards)) if seen_boards else 0.0
    for i, p in enumerate(solid):
        r = results[i]
        extent = np.sort(parts[p]["hi"] - parts[p]["lo"]) * unit_after
        if r["status"] not in ("ok", "missing") or extent[1] < 0.12 or extent[0] > 0.07:   # boards and panels only
            continue
        # a "missing" board may be there but tipped away from its place: look a little farther for it
        reach = 0.05 if r["status"] == "ok" else 0.15
        own = aligned[owner == i]
        if len(own):
            d, _ = cKDTree(target[labels == i]).query(own, distance_upper_bound=reach / unit_after)
            own = own[np.isfinite(d)]
        if len(own) < 300:
            continue
        mc, mn = _plane(target[labels == i])
        for _ in range(2):   # drop points off the main surface (edges, neighbours), refit
            pc, pn = _plane(own)
            keep = np.abs((own - pc) @ pn) < 0.015 / unit_after
            if keep.sum() < 200:
                break
            own = own[keep]
        pc, pn = _plane(own)
        if r["status"] == "missing":
            # only a flat sheet about the board's size counts as the board itself
            spread = np.sort(np.ptp(own - pc, axis=0))[1:] * unit_after
            off_plane = float(np.percentile(np.abs((own - pc) @ pn), 90)) * unit_after
            if (spread.min() < 0.6 * extent[1] or spread.max() > 1.3 * extent[2] + 0.05 or off_plane > 0.02
                    or len(own) < 0.5 * board_points):   # a few stray points are not a whole board
                continue
        tilt = math.degrees(math.acos(min(1.0, abs(float(pn @ mn)))))
        shift = float((pc - mc) @ mn) * unit_after
        if abs(mn @ up_fit) > 0.7:   # a lying board: along its normal is up and down
            shift *= 1 if mn @ up_fit > 0 else -1
            where = f"sits about {abs(shift) * 100:.0f} cm too {'high' if shift > 0 else 'low'}"
        else:
            where = f"stands about {abs(shift) * 100:.0f} cm out of line"
        if r["status"] == "missing" and tilt <= TILT and abs(shift) <= SHIFT:
            continue
        if tilt > TILT:
            r["status"], r["message"] = "misplaced", f"Slanted by about {tilt:.0f} degrees. Check it sits level and is pushed all the way on."
        elif abs(shift) > SHIFT:
            r["status"], r["message"] = "misplaced", f"It {where}. Check it is pushed all the way on."
        r["tilt_deg"], r["shift_cm"] = round(tilt, 1), round(shift * 100, 1)

    # Feet and other parts the furniture stands on: without them it sits lower than the model, so the
    # fitted model sinks into the real floor. The parts that end up under the floor are missing.
    sunk = _sunk_parts(article, expected, unit_after, up_fit, floor_level, spacing)
    for p in expected:
        if p in sunk:
            r = next((r for r in results if r["id"] == p), None)
            if r is None:
                r = {"id": p, "name": parts[p]["name"], "active": p in body.active, "coverage": None}
                results.append(r)
            r["status"] = "missing"
            r["message"] = (f"Your furniture sits about {sunk[p] * 100:.0f} cm lower than it should, as if this part "
                            "isn't on (or isn't screwed in all the way).")
        elif parts[p]["fastener"]:
            results.append({"id": p, "name": parts[p]["name"], "active": p in body.active, "status": "hardware",
                            "message": "Too small for the scan; checked from the photo.", "coverage": None})

    # scanned shapes where no part should be yet
    all_lo = np.min([q["lo"] for q in parts.values()], axis=0) - 0.05 / unit_after
    all_hi = np.max([q["hi"] for q in parts.values()], axis=0) + 0.05 / unit_after
    inside = np.all((aligned > all_lo) & (aligned < all_hi), axis=1) & (aligned @ up_fit > floor_level + 0.02 / unit_after)
    gap, _ = model_tree.query(aligned)
    stray = aligned[inside & (gap > EXTRA / unit_after)]
    extras = _clusters(stray, unit_after, _density(aligned[gap <= TOLERANCE / unit_after], unit_after))
    for e in extras:
        pts = e.pop("points")
        best, share = None, 0.0
        for p in future:
            d, _ = cKDTree(_samples(article, p, spacing)).query(pts, distance_upper_bound=0.03 / unit_after)
            frac = float(np.isfinite(d).mean())
            if frac > share:
                best, share = p, frac
        _, nearest = model_tree.query(e["center"])
        beside = parts[solid[labels[nearest]]]["name"]
        if best is not None and share >= 0.5:
            e["part"] = best
            e["message"] = f"{parts[best]['name']} looks already fitted, but it belongs to a later step."
        else:
            e["message"] = f"Something was scanned next to the {beside} where no part should be."
    # Only a blob shaped like a later part is a sure mistake. Other stray blobs come from where the real
    # furniture differs from the model (rods a little apart, a bag beside it) and would be false alarms.
    extras = [e for e in extras if "part" in e]

    missing = [r for r in results if r["status"] == "missing"]
    misplaced = [r for r in results if r["status"] == "misplaced"]
    thin_unsure = any(r["status"] == "unsure" and r.get("thin") for r in results)
    verdict = "problems" if missing or misplaced or extras else "unsure" if reference < 0.2 or thin_unsure else "ok"
    # how the real furniture compares with the kit model (only meaningful when the model has a real size)
    known = metres_per_unit(kit)
    resized = unit_after / known if known else None
    return {
        "verdict": verdict,
        "scan": "fitted",
        "missing": plain_list(r["name"] for r in missing),
        "misplaced": [f"{plain_name(r['name'])}: {r['message'].split('.')[0].lower()}" for r in misplaced],
        "resized": round(resized, 3) if resized and abs(resized - 1) > 0.05 else None,
        "transform": transform.T.reshape(-1).tolist(),   # column-major, model units -> world metres
        "fit": {"rmse_mm": round(rmse * 1000, 1), "inliers": inliers, "points": int(len(scan)),
                "scale": round(scale, 4), "coverage": round(reference, 3), **match},
        "parts": results,
        "extras": extras,
    }


TILT = 9.0    # degrees a board may lean before it counts as slanted
SHIFT = 0.03  # metres a board may sit off its place before it counts as out of place


def _plane(points):
    """Centre and unit normal of the best plane through points."""
    c = points.mean(0)
    return c, np.linalg.svd(points - c, full_matrices=False)[2][2]


SINK = 0.03   # the fitted furniture this far into the real floor means what it stands on is missing


def _sunk_parts(article, expected, unit, up, floor_level, spacing):
    """Parts that would have to be under the real floor for the scan to fit: {part: how far, metres}.
    Only parts at the very bottom count (feet, legs, plinths), and only when the whole furniture sinks."""
    parts, _ = _arrays(article)
    rises = {p: _samples(article, p, spacing) @ up for p in expected}
    rises = {p: r for p, r in rises.items() if len(r)}
    if not rises:
        return {}
    bottom = min(float(r.min()) for r in rises.values())
    sink = (floor_level - bottom) * unit
    if sink < SINK:
        return {}
    return {p: sink for p, r in rises.items() if float(np.median(r)) < floor_level - 0.3 * SINK / unit}


def _floor(world, ghost_world):
    """Height of the floor in world metres: the lowest dense 5 mm layer of scan points just below
    the ghost's underside, which is where the person placed it. Lowest, not densest, so that a big
    flat board lying above the floor is never taken for it."""
    rough = float(ghost_world[:, 1].min())
    lo, hi = ghost_world[:, [0, 2]].min(0) - 0.5, ghost_world[:, [0, 2]].max(0) + 0.5
    near = world[np.all((world[:, [0, 2]] > lo) & (world[:, [0, 2]] < hi), axis=1)]
    ys = near[(near[:, 1] > rough - 0.1) & (near[:, 1] < rough + 0.1), 1]
    if len(ys) < 200:
        return rough
    counts, edges = np.histogram(ys, bins=np.arange(rough - 0.1, rough + 0.1001, 0.005))
    dense = np.flatnonzero(counts >= 0.3 * counts.max())
    first = int(dense[0])
    return float((edges[first] + edges[first + 1]) / 2)


def _density(points, unit):
    """Typical number of scan points in a 3 cm cube of real surface."""
    keys = np.floor(points / (0.03 / unit)).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return float(np.median(counts)) if len(counts) else 0.0


def _clusters(points, unit, density=0.0):
    """Groups stray points into blobs bigger than a few centimetres. Only cubes holding at least
    half the scan's usual surface density count: real LiDAR leaves a sparse haze of stray depth
    readings around edges, which would otherwise join up into one big false blob."""
    if len(points) < 40:
        return []
    cell = 0.03 / unit
    keys = np.floor((points - points.min(0)) / cell).astype(int)
    if density > 0:
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        solid = counts[inverse.reshape(-1)] >= 0.5 * density
        points, keys = points[solid], keys[solid]
        if len(points) < 40:
            return []
        keys -= keys.min(0)
    grid = np.zeros(keys.max(0) + 1, dtype=bool)
    grid[tuple(keys.T)] = True
    labelled, count = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    ids = labelled[tuple(keys.T)]
    out = []
    for k in range(1, count + 1):
        pts = points[ids == k]
        extent = (pts.max(0) - pts.min(0)) * unit
        if len(pts) >= 40 and extent.max() >= 0.06:
            out.append({"center": pts.mean(0).tolist(), "lo": pts.min(0).tolist(), "hi": pts.max(0).tolist(),
                        "size_m": [round(float(x), 3) for x in extent], "points": pts})
    out.sort(key=lambda e: -len(e["points"]))
    return out[:5]


# ---------- the photo ----------

def plain_name(name: str) -> str:
    """'TOP RAIL_72935' -> 'top rail'; 'part 05: horizontal rail/bar, 35% of ...' -> 'horizontal rail/bar'."""
    import re
    if ":" in name:
        name = name.split(":", 1)[1].split(",")[0]
    name = re.sub(r"[_\s]*\d{3,}(_\d+)?$", "", name)          # catalogue numbers
    name = re.sub(r"([_\s]+\d+)+$", "", name)                  # copy numbers: Rod_2_2 -> Rod
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)            # BottomLeft -> Bottom Left
    return re.sub(r"[_\s]+", " ", name).strip().lower() or "a part"


def plain_list(names) -> list[str]:
    """Plain names with repeats grouped: ['rod', 'rod', 'top rail'] -> ['2 x rod', 'top rail']."""
    from collections import Counter
    counts = Counter(plain_name(n) for n in names)
    return [f"{n} x {name}" if n > 1 else name for name, n in counts.items()]


class Advice(BaseModel):
    missing: list[str] = Field(description="Every part that is missing or not attached yet at this step, each in 2 to 4 "
                                           "simple words a person would use, e.g. 'back panel', 'front left leg', "
                                           "'2 rods on the right'. Group identical ones. Empty if nothing is missing")
    out_of_place: list[str] = Field(default_factory=list, description="Parts that are on but wrong: slanted, crooked, not "
                                    "level, loose, upside down, in the wrong holes, or screw holes left empty. Each as a "
                                    "short phrase, part first, e.g. 'top right panel slanted', '8 screws on top missing'. "
                                    "Empty if none")
    screws: Literal["ok", "problem", "cant_tell"] = Field(description="Whether the screws, bolts and dowels of this step look right")
    summary: str = Field(description="One short sentence for the person assembling, plain words, no part numbers they cannot see")
    tips: list[str] = Field(description="Up to three short, concrete fixes, e.g. 'The right top panel isn't on yet.' Empty if all is well")


ADVICE_PROMPT = """This photo shows furniture part-way through assembly, right after this step of the manual:
{captions}
Small parts and hardware that should be on by now (feet, screws, etc.): {hardware}
A 3D scan already compared the big parts with the model:
{parts}
Look at the photo.
1. List the parts that are missing or not attached yet at this step, in short simple words ("back panel",
   "front left leg", "2 rods on the right"), not the codes above. Include the ones the scan marked missing unless
   the photo clearly shows them in place, and any others you can see are missing.
2. Check every small part listed above, feet included: is each one on, in the right place, and screwed in as far
   as the step says? A missing foot or leg is a missing part: list it in 1.
3. Look for anything that is on but wrong: a panel slanted, crooked or not level, a part loose, upside down or on
   the wrong side, screws or dowels missing from holes that should have them by now. List each in out_of_place.
Speak to the person building it, in plain words, like "the right top panel isn't on yet". Keep everything short.
Do not invent problems you cannot see; say you can't tell when the photo does not show it."""


def advice(body: CheckIn, results: list[dict]) -> Optional[dict]:
    if not body.photo:
        return None
    from PIL import Image, ImageOps
    from pipeline import _png
    from providers import make_provider
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(body.photo)))).convert("RGB")
    image.thumbnail((1600, 1600))
    lines = "\n".join(f"- {r['name']} ({plain_name(r['name'])}): {r['status']}" + (" (fitted in this step)" if r.get("active") else "")
                      for r in results if r["status"] != "hardware") or "- (no large parts)"
    small = plain_list(r["name"] for r in results if r["status"] == "hardware")
    hardware = ", ".join([f"{h.count} x {h.name}" for h in body.hardware] + small) or "none listed"
    prompt = ADVICE_PROMPT.format(captions="\n".join("- " + c for c in body.captions) or f"- Step {body.step}",
                                  hardware=hardware, parts=lines)
    try:
        # A slow model must not hold up the scan result: give up on the photo after ADVICE_SECONDS
        pool = ThreadPoolExecutor(max_workers=1)
        job = pool.submit(make_provider(fail_fast=True).generate, [_png(image)], prompt, Advice, 0.1)
        pool.shutdown(wait=False)
        out = job.result(timeout=ADVICE_SECONDS)
        return {"screws": out.screws, "summary": out.summary.strip(), "tips": [t.strip() for t in out.tips if t.strip()][:3],
                "missing": [m.strip() for m in out.missing if m.strip()][:8],
                "out_of_place": [m.strip() for m in out.out_of_place if m.strip()][:6]}
    except Exception as error:   # the scan result is still useful without the photo check
        return {"screws": "cant_tell", "summary": "The photo check is unavailable right now.",
                "tips": [], "error": f"{type(error).__name__}: {str(error)[:200]}"}


def _stems(text: str) -> set[str]:
    """Words for matching part names loosely: 'Support rods', '8 feet' -> {'support', 'rod', 'foot'}."""
    import re
    irregular = {"feet": "foot", "shelves": "shelf", "legs": "leg"}
    words = re.findall(r"[a-z]+", text.lower())
    return {irregular.get(w, w[:-1] if w.endswith("s") and len(w) > 3 else w) for w in words if len(w) > 2}


def _count(phrase: str) -> Optional[int]:
    """'7 support rods' -> 7, 'two feet' -> 2, 'top panel' -> None."""
    import re
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    m = re.match(r"\s*(\d+|[a-z]+)\b", phrase.lower())
    if not m:
        return None
    return int(m.group(1)) if m.group(1).isdigit() else words.get(m.group(1))


def _mark(result: dict, phrase: str, status: str, message: str):
    """Turns the parts a photo finding names red: 'top right panel slanted' -> Top_Right. With a count
    ('7 rods'), the ones the scan saw least of."""
    said = _stems(phrase)
    named = [r for r in result["parts"] if _stems(plain_name(r["name"])) and _stems(plain_name(r["name"])) <= said
             and r["status"] not in ("missing", "misplaced")]
    if not named:
        return
    # the most specific name wins: 'top right panel' means Top_Right, not every 'top' part
    best = max(len(_stems(plain_name(r["name"]))) for r in named)
    named = [r for r in named if len(_stems(plain_name(r["name"]))) == best]
    n = _count(phrase)
    if n is not None and n < len(named):
        named = sorted(named, key=lambda r: r.get("coverage") or 0)[:n]
    for r in named:
        r["status"], r["message"] = status, message


def combine(result: dict):
    """One answer from both checks. LiDAR judges the big boards and how level they sit; the photo
    judges thin parts (rods, dowels), hardware and anything crooked, and decides on its own when the 3D
    scan could not be used. Everything either one finds wrong is red on the model."""
    photo = result.get("advice")
    seen = bool(photo) and "missing" in photo     # the photo check ran
    photo_missing = photo.get("missing", []) if seen else []
    photo_wrong = photo.get("out_of_place", []) if seen else []
    scan_missing = result.get("missing") or []
    result["missing"] = photo_missing if seen else scan_missing
    result["misplaced"] = list(result.get("misplaced") or [])
    screws_bad = bool(photo) and photo.get("screws") == "problem"
    if seen:
        for phrase in photo_missing:
            _mark(result, phrase, "missing", f"The photo shows this is missing: {phrase}.")
        for phrase in photo_wrong:
            _mark(result, phrase, "misplaced", f"The photo shows: {phrase}.")
        listed = _stems(" ".join(photo_missing))
        result["missing"] += [m for m in scan_missing if not _stems(m.split(" x ")[-1]) <= listed]
        wrong = _stems(" ".join(photo_wrong))
        result["misplaced"] = photo_wrong + [m for m in result["misplaced"] if not _stems(m.split(":")[0]) <= wrong]
        # parts the scan couldn't judge (thin rods, or everything when the scan didn't fit) are there
        # unless the photo named them
        for r in result["parts"]:
            if r["status"] == "unsure" and (r.get("thin") or result.get("scan") != "fitted"):
                r["status"], r["message"] = "ok", "Seen in place in the photo."
    if result["missing"] or result["misplaced"] or screws_bad or result.get("extras"):
        result["verdict"] = "problems"
    elif result["verdict"] == "ok" or (seen and result["verdict"] == "unsure"):
        result["verdict"] = "ok"
    if result["verdict"] == "ok":
        result["summary"] = None   # the scan's own caveats do not matter once everything checks out


@router.post("/{article}")
def check_step(article: str, body: CheckIn):
    started = time.time()
    try:   # every real scan is kept so a bad result can be replayed and fixed (server/jobs is not committed)
        SAVED.mkdir(parents=True, exist_ok=True)
        (SAVED / f"{time.strftime('%Y%m%d-%H%M%S')}-{article}-step{body.step}.json").write_text(body.model_dump_json())
    except OSError:
        pass
    print(f"[stepcheck] {article} step {body.step}: {len(body.points) * 3 // 4 // 12} points, "
          f"photo {'yes' if body.photo else 'no'}", flush=True)
    result = check(article, body)
    print(f"[stepcheck] scan compared in {time.time() - started:.1f}s: {result['verdict']}, fit {result.get('fit')}", flush=True)
    result["advice"] = advice(body, result["parts"]) if result["parts"] else None
    combine(result)
    print(f"[stepcheck] {result['verdict']}, missing: {result['missing']}", flush=True)
    print(f"[stepcheck] done in {time.time() - started:.1f}s", flush=True)
    return result
