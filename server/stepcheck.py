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


def _icp(scan, target, unit, with_scale):
    """Moves scan (model units) onto target samples. Returns the 4x4 similarity and the fit.
    `unit` is metres per model unit at the ghost's scale, so the distance gates stay in metres."""
    tree = cKDTree(target)
    total = np.eye(4)
    moved = scan.copy()
    gates = np.geomspace(0.25, 0.015, 36) / unit   # recovers a ghost placed up to ~25 cm off
    rmse, inliers = float("inf"), 0
    size = 1.0
    for gate in list(gates) + [gates[-1]] * 10:
        d, idx = tree.query(moved, distance_upper_bound=gate)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        s, r, t = _umeyama(moved[ok], target[idx[ok]], with_scale)
        # size may only drift a little per round, and at most halve or double overall
        s = float(np.clip(s, 0.97, 1.03))
        s = float(np.clip(size * s, 0.5, 2.0) / size)
        size *= s
        step = np.eye(4)
        step[:3, :3] = s * r
        step[:3, 3] = t
        moved = moved @ (s * r).T + t
        total = step @ total
        rmse, inliers = float(np.sqrt((d[ok] ** 2).mean())) * unit, int(ok.sum())
        if np.abs(step - np.eye(4)).max() < 1e-6:
            break
    return total, moved, rmse, inliers


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


def _matrix(values):
    if len(values) != 16:
        raise HTTPException(400, "ghost must be 16 numbers")
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T   # column-major in


def check(article: str, body: CheckIn) -> dict:
    kit = get_kit(article)
    parts, diagonal = _arrays(article)
    raw = np.frombuffer(base64.b64decode(body.points), dtype="<f4")
    if raw.size < 300 or raw.size % 3:
        raise HTTPException(400, "Too few scanned points. Walk all the way around the furniture and scan again.")
    world = raw.reshape(-1, 3).astype(np.float64)

    ghost_m = _matrix(body.ghost)
    unit = float(np.cbrt(abs(np.linalg.det(ghost_m[:3, :3]))))   # metres per model unit at the ghost's scale
    to_model = np.linalg.inv(ghost_m)

    expected = [p for p in dict.fromkeys(body.installed + body.active) if p in parts]
    solid = [p for p in expected if not parts[p]["fastener"]]
    if not solid:
        raise HTTPException(400, "This step has no parts large enough to scan.")
    spacing = SPACING / unit
    samples = {p: _samples(article, p, spacing) for p in solid}

    # floor: the ghost stands on the floor, so its lowest point marks it roughly; the scan's own
    # densest flat layer near there marks it exactly
    target = np.concatenate([samples[p] for p in solid])
    target_world = target @ ghost_m[:3, :3].T + ghost_m[:3, 3]
    floor_y = _floor(world, target_world)
    height = float(target_world[:, 1].max() - target_world[:, 1].min())
    if height < FLAT:
        results = [{"id": p, "name": parts[p]["name"], "active": p in body.active, "status": "unsure", "coverage": None,
                    "message": "Lies too flat on the floor for the 3D scan to tell apart; checked from the photo."}
                   for p in expected]
        return {"verdict": "unsure", "summary": "At this step everything lies nearly flat on the floor, so the 3D scan "
                "can't check it. The photo check below is all there is for this step.",
                "parts": results, "extras": [], "transform": body.ghost, "fit": {"height_m": round(height, 3)}}
    above = world[:, 1] > floor_y + FLOOR
    model_pts = world @ to_model[:3, :3].T + to_model[:3, 3]

    lo = target.min(0) - 0.25 / unit
    hi = target.max(0) + 0.25 / unit
    near = above & np.all((model_pts > lo) & (model_pts < hi), axis=1)
    scan = model_pts[near]
    if len(scan) < 200:
        return {"verdict": "unsure", "summary": "The scan did not pick up the furniture under the ghost. "
                "Line the ghost up with your furniture, then walk slowly all the way around it.",
                "parts": [], "extras": [], "transform": body.ghost, "fit": {"points": int(len(scan))}}
    if len(scan) > 60000:
        scan = scan[np.random.default_rng(0).choice(len(scan), 60000, replace=False)]

    # The size is always fitted to the scan, so the model matches the furniture in front of the person
    # even when it is a different size from the kit's model (or the model has no real size at all)
    icp, aligned, rmse, inliers = _icp(scan, target, unit, with_scale=True)
    if inliers < max(100, 0.25 * len(scan)) or not np.isfinite(rmse):
        # the scan under the ghost does not look like this furniture at all
        results = [{"id": p, "name": parts[p]["name"], "active": p in body.active, "status": "unsure", "coverage": None,
                    "message": "Not checked: the scan could not be lined up with the model."} for p in expected]
        return {"verdict": "unsure", "summary": "The scan doesn't match the shape of the furniture at this step, so the "
                "parts couldn't be checked. Make sure the ghost sits right over your furniture (tip it over if it's "
                "lying down), then scan again, walking all the way around.",
                "parts": results, "extras": [], "transform": body.ghost,
                "fit": {"points": int(len(scan)), "inliers": inliers}}
    scale = float(np.cbrt(abs(np.linalg.det(icp[:3, :3]))))
    unit_after = unit / scale          # metres per model unit after the fit
    transform = ghost_m @ np.linalg.inv(icp)

    # which model samples the scan covers, ignoring ones hidden on the floor or inside joints
    labels = np.concatenate([np.full(len(samples[p]), i) for i, p in enumerate(solid)])
    model_tree = cKDTree(target)
    floor_model = target[:, 1].min()
    hidden = target[:, 1] < floor_model + 0.02 / unit_after
    for i, neighbours in enumerate(model_tree.query_ball_point(target, CONTACT / unit_after)):
        if not hidden[i] and any(labels[j] != labels[i] for j in neighbours):
            hidden[i] = True
    # Each scan point belongs to the part it lies closest to, later parts included. Otherwise a
    # missing rail would look covered by the scan of the boards it touches.
    future = [p for p in body.future if p in parts and not parts[p]["fastener"] and p not in solid]
    later =[_samples(article, p, spacing) for p in future]
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
        results.append({"id": p, "name": parts[p]["name"], "active": p in body.active,
                        "visible_samples": nv, "coverage": round(coverage, 3)})
    measured = [r["coverage"] for r in results if r["visible_samples"] >= 30]
    reference = float(np.median(measured)) if measured else 0.0
    for r in results:
        if r["visible_samples"] < 30:
            r["status"], r["message"] = "unsure", "Mostly hidden from view, so the scan cannot check it."
        elif r["coverage"] >= max(0.25, 0.45 * reference):
            r["status"], r["message"] = "ok", "In place."
        elif r["coverage"] < max(0.1, 0.3 * reference):   # scan noise lends a missing part some coverage from its neighbours
            r["status"] = "missing"
            r["message"] = ("Nothing was scanned where this part goes. It may be missing, in the wrong place, "
                            "or not pushed all the way in.")
        else:
            r["status"] = "unsure"
            r["message"] = "Only partly picked up by the scan. Check it is fully seated, or scan this side again."
    for p in expected:
        if parts[p]["fastener"]:
            results.append({"id": p, "name": parts[p]["name"], "active": p in body.active, "status": "hardware",
                            "message": "Too small for the scan; checked from the photo.", "coverage": None})

    # scanned shapes where no part should be yet
    all_lo = np.min([q["lo"] for q in parts.values()], axis=0) - 0.05 / unit_after
    all_hi = np.max([q["hi"] for q in parts.values()], axis=0) + 0.05 / unit_after
    inside = np.all((aligned > all_lo) & (aligned < all_hi), axis=1) & (aligned[:, 1] > floor_model + 0.02 / unit_after)
    gap, _ = model_tree.query(aligned)
    stray = aligned[inside & (gap > EXTRA / unit_after)]
    extras = _clusters(stray, unit_after)
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
            e["message"] = (f"Something was scanned next to the {beside} where no part should be at this step: "
                            "a part in the wrong place, or an object close to the furniture.")

    missing = [r for r in results if r["status"] == "missing"]
    verdict = "problems" if missing or extras else "unsure" if reference < 0.2 else "ok"
    # how the real furniture compares with the kit model (only meaningful when the model has a real size)
    known = metres_per_unit(kit)
    resized = unit_after / known if known else None
    return {
        "verdict": verdict,
        "missing": plain_list(r["name"] for r in missing),
        "resized": round(resized, 3) if resized and abs(resized - 1) > 0.05 else None,
        "transform": transform.T.reshape(-1).tolist(),   # column-major, model units -> world metres
        "fit": {"rmse_mm": round(rmse * 1000, 1), "inliers": inliers, "points": int(len(scan)),
                "scale": round(scale, 4), "coverage": round(reference, 3)},
        "parts": results,
        "extras": extras,
    }


def _floor(world, ghost_world):
    """Height of the floor in world metres: the lowest dense 5 mm layer of scan points just below
    the ghost's underside, which is where the person placed it. Lowest, not densest, so that a big
    flat board lying above the floor is never taken for it."""
    rough = float(ghost_world[:, 1].min())
    lo, hi = ghost_world[:, [0, 2]].min(0) - 0.5, ghost_world[:, [0, 2]].max(0) + 0.5
    near = world[np.all((world[:, [0, 2]] > lo) & (world[:, [0, 2]] < hi), axis=1)]
    ys = near[(near[:, 1] > rough - 0.08) & (near[:, 1] < rough + 0.03), 1]
    if len(ys) < 200:
        return rough
    counts, edges = np.histogram(ys, bins=np.arange(rough - 0.08, rough + 0.0301, 0.005))
    dense = np.flatnonzero(counts >= 0.3 * counts.max())
    first = int(dense[0])
    return float((edges[first] + edges[first + 1]) / 2)


def _clusters(points, unit):
    """Groups stray points into blobs bigger than a few centimetres."""
    if len(points) < 40:
        return []
    cell = 0.03 / unit
    keys = np.floor((points - points.min(0)) / cell).astype(int)
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
    screws: Literal["ok", "problem", "cant_tell"] = Field(description="Whether the screws, bolts and dowels of this step look right")
    summary: str = Field(description="One short sentence for the person assembling, plain words, no part numbers they cannot see")
    tips: list[str] = Field(description="Up to three short, concrete fixes, e.g. 'The right top panel isn't on yet.' Empty if all is well")


ADVICE_PROMPT = """This photo shows furniture part-way through assembly, right after this step of the manual:
{captions}
Hardware used in this step: {hardware}
A 3D scan already compared the big parts with the model:
{parts}
Look at the photo.
1. List the parts that are missing or not attached yet at this step, in short simple words ("back panel",
   "front left leg", "2 rods on the right"), not the codes above. Include the ones the scan marked missing unless
   the photo clearly shows them in place, and any others you can see are missing.
2. Check the screws, bolts, dowels and other small hardware of this step: are they in, in the right holes, and
   screwed in as far as the step says?
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
    hardware = ", ".join(f"{h.count} x {h.name}" for h in body.hardware) or "none listed"
    prompt = ADVICE_PROMPT.format(captions="\n".join("- " + c for c in body.captions) or f"- Step {body.step}",
                                  hardware=hardware, parts=lines)
    try:
        # A slow model must not hold up the scan result: give up on the photo after ADVICE_SECONDS
        pool = ThreadPoolExecutor(max_workers=1)
        job = pool.submit(make_provider(fail_fast=True).generate, [_png(image)], prompt, Advice, 0.1)
        pool.shutdown(wait=False)
        out = job.result(timeout=ADVICE_SECONDS)
        return {"screws": out.screws, "summary": out.summary.strip(), "tips": [t.strip() for t in out.tips if t.strip()][:3],
                "missing": [m.strip() for m in out.missing if m.strip()][:8]}
    except Exception as error:   # the scan result is still useful without the photo check
        return {"screws": "cant_tell", "summary": "The photo check is unavailable right now.",
                "tips": [], "error": f"{type(error).__name__}: {str(error)[:200]}"}


@router.post("/{article}")
def check_step(article: str, body: CheckIn):
    started = time.time()
    print(f"[stepcheck] {article} step {body.step}: {len(body.points) * 3 // 4 // 12} points, "
          f"photo {'yes' if body.photo else 'no'}", flush=True)
    result = check(article, body)
    print(f"[stepcheck] scan compared in {time.time() - started:.1f}s: {result['verdict']}, fit {result.get('fit')}", flush=True)
    result["advice"] = advice(body, result["parts"]) if result["parts"] else None
    # Gemini's plain names win; the scan's own list stands when the photo check is unavailable
    if result["advice"] and "missing" in result["advice"]:
        result["missing"] = result["advice"]["missing"]
    result.setdefault("missing", [])
    print(f"[stepcheck] missing: {result['missing']}", flush=True)
    print(f"[stepcheck] done in {time.time() - started:.1f}s", flush=True)
    if result["advice"] and result["advice"].get("screws") == "problem" and result["verdict"] == "ok":
        result["verdict"] = "problems"
    return result
