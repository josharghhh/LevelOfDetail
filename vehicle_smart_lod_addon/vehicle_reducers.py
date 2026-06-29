"""
Vehicle-specific reducer backend for Vehicle Smart LOD.

This module is pure Python (no Blender dependency) so every routine here can be
unit-tested standalone (see TESTS.py). It implements the vehicle-aware passes
that the generic wedge/QEM reducer cannot express well on its own:

1. ``compute_smooth_normals`` - recompute area-weighted smooth vertex normals on
   a *reduced* mesh, respecting wedge/normal domains. This replaces the old
   behaviour of writing the stale source face normal as the custom normal.

2. ``estimate_radial_axis`` / ``radial_reduce`` - a radial ring/segment reducer
   for tyres and other surfaces of revolution. Instead of collapsing arbitrary
   edges it decimates in the angular direction and preserves the cross-section
   profile, which is what keeps a round tyre round.

3. ``visibility_cull`` - remove interior geometry that is never visible from any
   exterior viewpoint (cockpit guts, hidden chassis, doubled interior walls).
   Uses ray sampling from a Fibonacci sphere with a uniform spatial grid for
   broad-phase occlusion so it stays tractable in pure Python.

4. ``convex_hull_3d`` / ``convex_decompose`` - a convex hull and a clustered
   multi-hull approximate-convex-decomposition collision proxy generator. Far
   tighter than a single bounding box while still cheap and import-safe.

All functions degrade gracefully: bad/degenerate input returns a valid (often
empty or pass-through) result rather than raising.
"""

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# small vector helpers (kept local so this module has no cross-imports)
# ---------------------------------------------------------------------------

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a, fallback=None):
    n = _length(a)
    if n < 1e-12:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def _centroid(points):
    n = len(points)
    if n == 0:
        return (0.0, 0.0, 0.0)
    sx = sy = sz = 0.0
    for p in points:
        sx += p[0]
        sy += p[1]
        sz += p[2]
    return (sx / n, sy / n, sz / n)


def _face_normal(vertices, face):
    """Unit normal of a polygon from its first non-degenerate triangle fan."""
    p0 = vertices[face[0]]
    for i in range(1, len(face) - 1):
        n = _cross(_sub(vertices[face[i]], p0), _sub(vertices[face[i + 1]], p0))
        if _length(n) > 1e-12:
            return _normalize(n)
    return None


VEHICLE_SILHOUETTE_VIEW_DIRECTIONS = (
    (0.0, -1.0, 0.0),   # front
    (0.0, 1.0, 0.0),    # rear
    (-1.0, 0.0, 0.0),   # left side
    (1.0, 0.0, 0.0),    # right side
    (0.6, -0.6, 0.5),   # top-front quarter
    (-0.6, -0.6, 0.5),  # opposite top-front quarter
    (0.6, 0.6, 0.5),    # top-rear quarter
    (-0.6, 0.6, 0.5),   # opposite top-rear quarter
)


def _view_basis(direction):
    forward = _normalize(direction, (0.0, 0.0, 1.0))
    up_hint = (0.0, 0.0, 1.0)
    if abs(_dot(forward, up_hint)) > 0.95:
        up_hint = (0.0, 1.0, 0.0)
    right = _normalize(_cross(up_hint, forward), (1.0, 0.0, 0.0))
    up = _normalize(_cross(forward, right), (0.0, 0.0, 1.0))
    return right, up


def _projected_bbox(vertices, direction):
    if not vertices:
        return {
            "min": (0.0, 0.0),
            "max": (0.0, 0.0),
            "width": 0.0,
            "height": 0.0,
            "area": 0.0,
            "diagonal": 0.0,
        }
    right, up = _view_basis(direction)
    points = [(_dot(v, right), _dot(v, up)) for v in vertices]
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    width = max_x - min_x
    height = max_y - min_y
    return {
        "min": (min_x, min_y),
        "max": (max_x, max_y),
        "width": width,
        "height": height,
        "area": width * height,
        "diagonal": math.sqrt(width * width + height * height),
    }


def sampled_view_silhouette_error(source_vertices, reduced_vertices, directions=None):
    """Compare projected silhouettes from deterministic vehicle view directions.

    This intentionally uses a fast projection-envelope approximation rather
    than rasterization so it can run inside Blender reports and pure-Python
    tests. The returned per-view ``deviation`` is the largest projected bbox edge
    movement normalized by the source projected diagonal; ``area_ratio`` tracks
    gross silhouette shrink/growth for the same view.
    """
    directions = directions or VEHICLE_SILHOUETTE_VIEW_DIRECTIONS
    source_vertices = [tuple(float(c) for c in v) for v in (source_vertices or [])]
    reduced_vertices = [tuple(float(c) for c in v) for v in (reduced_vertices or [])]
    per_view = []
    max_deviation = 0.0
    max_area_delta = 0.0
    for direction in directions:
        src = _projected_bbox(source_vertices, direction)
        red = _projected_bbox(reduced_vertices, direction)
        denom = max(src["diagonal"], 1e-9)
        edge_delta = max(
            abs(red["min"][0] - src["min"][0]),
            abs(red["min"][1] - src["min"][1]),
            abs(red["max"][0] - src["max"][0]),
            abs(red["max"][1] - src["max"][1]),
        )
        deviation = edge_delta / denom
        area_ratio = red["area"] / max(src["area"], 1e-12)
        area_delta = abs(area_ratio - 1.0)
        max_deviation = max(max_deviation, deviation)
        max_area_delta = max(max_area_delta, area_delta)
        per_view.append({
            "direction": [round(float(c), 6) for c in _normalize(direction, (0.0, 0.0, 1.0))],
            "deviation": round(float(deviation), 6),
            "area_ratio": round(float(area_ratio), 6),
            "source_area": round(float(src["area"]), 6),
            "reduced_area": round(float(red["area"]), 6),
        })
    return {
        "max_deviation": round(float(max_deviation), 6),
        "max_area_delta": round(float(max_area_delta), 6),
        "views": per_view,
    }


# ===========================================================================
# 0. Smoothing-group / normal-domain partitioning
# ===========================================================================

def face_smoothing_groups(
    vertices,
    faces,
    sharp_edges=None,
    material_ids=None,
    angle_limit_rad=None,
    face_normals=None,
):
    """Flood-fill faces into smoothing-group ids ("normal domains").

    Two faces belong to the same group iff they share an edge that is:

      * manifold (used by exactly two faces),
      * not flagged sharp,
      * between faces of the same material, and
      * below the dihedral ``angle_limit_rad`` (i.e. the surface is "smooth"
        there, not a hard crease).

    This is the correct basis for wedge/normal domains. Keying domains on the
    raw per-face normal (the previous behaviour) shatters any curved surface
    into one island per triangle, which makes the wedge reducer unable to
    collapse *anything* on curved bodywork - exactly where a vehicle keeps most
    of its triangles. Grouping by hard edges instead keeps smooth regions
    connected (so they reduce) while still protecting genuine hard-surface
    creases, sharp-flagged edges, and material borders.

    Returns one integer group id per face (same order as ``faces``).
    """
    n = len(faces)
    if n == 0:
        return []
    if angle_limit_rad is None:
        angle_limit_rad = math.radians(35.0)
    sharp_edges = sharp_edges or set()
    if material_ids is None:
        material_ids = [0] * n

    if face_normals is None:
        normals = [_face_normal(vertices, f) for f in faces]
    else:
        normals = [
            _normalize(tuple(float(c) for c in fn)) if fn is not None else None
            for fn in face_normals
        ]

    cos_limit = math.cos(max(0.0, min(math.pi, angle_limit_rad)))

    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        m = len(face)
        for i in range(m):
            a = face[i]
            b = face[(i + 1) % m]
            if a != b:
                edge_faces[tuple(sorted((a, b)))].append(fi)

    adjacency = defaultdict(list)
    for edge, incident in edge_faces.items():
        if len(incident) != 2:
            continue  # boundary / non-manifold edge is always a domain border
        if edge in sharp_edges:
            continue
        f0, f1 = incident
        if material_ids[f0] != material_ids[f1]:
            continue
        n0, n1 = normals[f0], normals[f1]
        if n0 is None or n1 is None:
            continue
        if _dot(n0, n1) < cos_limit:
            continue  # dihedral angle exceeds the hard-edge threshold
        adjacency[f0].append(f1)
        adjacency[f1].append(f0)

    group = [-1] * n
    current = 0
    for start in range(n):
        if group[start] != -1:
            continue
        group[start] = current
        stack = [start]
        while stack:
            f = stack.pop()
            for nb in adjacency[f]:
                if group[nb] == -1:
                    group[nb] = current
                    stack.append(nb)
        current += 1
    return group


def allocate_triangle_budget(part_counts, budget, min_per_part=1):
    """Distribute an absolute triangle ``budget`` across parts.

    AAA LOD work is driven by triangle budgets (e.g. "LOD0 <= 90k, LOD1 <= 35k")
    rather than a blanket ratio. Given the current triangle count of each part
    this returns a per-part target count that sums to ``budget`` (or less when
    the parts are already under budget), allocating proportionally to size and
    never asking a part to drop below ``min_per_part`` while it still has
    geometry. Targets are clamped so a part is never asked to *grow*.

    Returns a list of integer targets aligned with ``part_counts``.
    """
    counts = [max(0, int(c)) for c in part_counts]
    total = sum(counts)
    budget = max(0, int(budget))
    if total == 0:
        return [0] * len(counts)
    if budget >= total:
        return list(counts)

    raw = [(budget * c) / total for c in counts]
    targets = [int(math.floor(r)) for r in raw]
    for i, c in enumerate(counts):
        if c > 0:
            targets[i] = max(targets[i], min(min_per_part, c))
        targets[i] = min(targets[i], c)

    def shortfall():
        return budget - sum(targets)

    remainders = sorted(
        range(len(counts)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True
    )
    idx = 0
    guard = 0
    while shortfall() > 0 and guard < len(counts) * 4:
        i = remainders[idx % len(remainders)]
        if targets[i] < counts[i]:
            targets[i] += 1
        idx += 1
        guard += 1
        if all(targets[j] >= counts[j] for j in range(len(counts))):
            break

    over = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    idx = 0
    while shortfall() < 0 and idx < len(over) * 4:
        i = over[idx % len(over)]
        floor_i = min(min_per_part, counts[i]) if counts[i] > 0 else 0
        if targets[i] > floor_i:
            targets[i] -= 1
        idx += 1
    return targets


# ===========================================================================
# 1. Smooth normal recomputation (post-collapse)
# ===========================================================================

def compute_smooth_normals(vertices, faces, vertex_domains=None, fallback=(0.0, 0.0, 1.0)):
    """Area-weighted smooth vertex normals computed from the *current* geometry.

    Because the wedge pipeline already splits vertices across material/normal
    domains, accumulating a normal per vertex automatically stops smoothing
    across a hard-normal seam: the two sides of the seam are distinct vertices.
    The optional ``vertex_domains`` argument is only used to break ties for
    isolated vertices and to keep the signature self-documenting.

    Returns one unit normal per vertex (same order/length as ``vertices``).
    """
    n = len(vertices)
    accum = [(0.0, 0.0, 0.0) for _ in range(n)]

    for face in faces:
        if len(face) < 3:
            continue
        a, b, c = face[0], face[1], face[2]
        if a >= n or b >= n or c >= n:
            continue
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        # un-normalised cross product is area-weighted (2x area magnitude)
        fn = _cross(_sub(pb, pa), _sub(pc, pa))
        for vi in (a, b, c):
            accum[vi] = _add(accum[vi], fn)

    normals = []
    for vi in range(n):
        nrm = _normalize(accum[vi])
        if nrm is None:
            # vertex had no usable incident face area; fall back to stored
            # domain normal if present, else the supplied fallback
            dom = vertex_domains[vi] if vertex_domains and vi < len(vertex_domains) else None
            if dom and len(dom) > 1 and isinstance(dom[1], (tuple, list)) and len(dom[1]) == 3:
                nrm = _normalize(tuple(float(x) for x in dom[1])) or fallback
            else:
                nrm = fallback
        normals.append(nrm)
    return normals


# ===========================================================================
# 2. Radial tyre / surface-of-revolution reducer
# ===========================================================================

def _covariance_axes(points, center):
    """Return the three principal axes (eigenvectors) sorted by ascending spread.

    Uses a small power-iteration / deflation on the 3x3 covariance matrix.
    Good enough to find a tyre's spin axis (the direction of smallest spread).
    """
    # build covariance
    cxx = cxy = cxz = cyy = cyz = czz = 0.0
    for p in points:
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        dz = p[2] - center[2]
        cxx += dx * dx
        cxy += dx * dy
        cxz += dx * dz
        cyy += dy * dy
        cyz += dy * dz
        czz += dz * dz
    n = max(1, len(points))
    cov = [
        [cxx / n, cxy / n, cxz / n],
        [cxy / n, cyy / n, cyz / n],
        [cxz / n, cyz / n, czz / n],
    ]

    def mat_vec(m, v):
        return (
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        )

    def power_iter(m, avoid):
        v = (0.2672, 0.5345, 0.8018)  # arbitrary non-axis-aligned seed
        for _ in range(80):
            # deflate against already-found axes
            for av, lam in avoid:
                d = _dot(v, av)
                v = _sub(v, _scale(av, d))
            w = mat_vec(m, v)
            nw = _length(w)
            if nw < 1e-15:
                break
            v = (w[0] / nw, w[1] / nw, w[2] / nw)
        val = _dot(v, mat_vec(m, v))
        return v, val

    found = []
    v1, l1 = power_iter(cov, [])
    found.append((v1, l1))
    v2, l2 = power_iter(cov, [(v1, l1)])
    found.append((v2, l2))
    # third axis is orthogonal to the first two
    v3 = _normalize(_cross(v1, v2)) or (0.0, 0.0, 1.0)
    l3 = _dot(v3, mat_vec(cov, v3))
    found.append((v3, l3))

    found.sort(key=lambda t: t[1])  # ascending spread
    return [f[0] for f in found], [f[1] for f in found]


def estimate_radial_axis(vertices):
    """Estimate the spin axis of a surface of revolution (tyre/wheel/cylinder).

    Returns ``(center, axis_dir, confidence)`` where confidence in [0, 1] is high
    when the geometry really is radially symmetric (two large, near-equal
    in-plane spreads and one smaller axial spread, with consistent radii).
    """
    if len(vertices) < 4:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0

    center = _centroid(vertices)
    axes, spreads = _covariance_axes(vertices, center)
    axis = axes[0]                  # smallest-spread direction = spin axis
    in_plane = (axes[1], axes[2])
    s_axial, s_a, s_b = spreads[0], spreads[1], spreads[2]

    # two in-plane spreads should be similar for a round part
    plane_balance = 1.0 - abs(s_a - s_b) / (s_a + s_b + 1e-12)
    # axial spread should be the smallest by a clear margin (disc/cylinder)
    axial_ratio = 1.0 - s_axial / (s_b + 1e-12)

    # radius consistency: project to the plane, measure radius variation
    radii = []
    for p in vertices:
        d = _sub(p, center)
        h = _dot(d, axis)
        planar = _sub(d, _scale(axis, h))
        radii.append(_length(planar))
    rmean = sum(radii) / len(radii)
    if rmean < 1e-9:
        return center, axis, 0.0
    rvar = sum((r - rmean) ** 2 for r in radii) / len(radii)
    radius_consistency = 1.0 / (1.0 + (math.sqrt(rvar) / rmean) * 2.0)

    confidence = max(0.0, min(1.0, plane_balance * 0.4 + max(0.0, axial_ratio) * 0.3 + radius_consistency * 0.3))
    return center, axis, confidence


def radial_reduce(
    vertices,
    uvs,
    faces,
    face_materials=None,
    vertex_domains=None,
    group_weights=None,
    axis_point=None,
    axis_dir=None,
    target_segments=16,
    precision=5,
    uv_reconstruction="auto",
):
    """Decimate a surface of revolution in the angular direction.

    Each vertex angle around the axis is snapped to one of ``target_segments``
    evenly spaced angular bins; its radius and axial height are preserved. Mesh
    vertices that snap to the same (position, domain) are welded, which removes
    whole angular rings while keeping the tyre's cross-section profile intact.

    ``uv_reconstruction`` controls UVs assigned to welded vertices:

      * ``"auto"`` (default) detects likely circumferential/tread UV bands and
        reconstructs the angular UV coordinate from the snapped segment.
      * ``"interpolate"`` always reconstructs the detected angular coordinate
        when possible.
      * ``"average"`` only averages the source UVs, but still unwraps seam-crossing
        ranges first so the seam stays continuous.

    Returns a dict shaped like the wedge reducer output (vertices/uvs/faces/...).
    """
    n = len(vertices)
    if face_materials is None:
        face_materials = [0] * len(faces)
    if vertex_domains is None:
        vertex_domains = [("default",)] * n
    if group_weights is None:
        group_weights = [{} for _ in range(n)]

    if n == 0 or not faces:
        return _empty_like(target_segments, len(faces))

    if axis_point is None or axis_dir is None:
        axis_point, axis_dir, _conf = estimate_radial_axis(vertices)
    axis = _normalize(axis_dir) or (0.0, 0.0, 1.0)

    # build an orthonormal frame (axis, u, w)
    seed = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    u = _normalize(_sub(seed, _scale(axis, _dot(seed, axis)))) or (1.0, 0.0, 0.0)
    w = _normalize(_cross(axis, u)) or (0.0, 1.0, 0.0)

    segments = max(3, int(target_segments))
    step = 2.0 * math.pi / segments

    snapped = []
    snap_fracs = []
    angles = []
    for p in vertices:
        d = _sub(p, axis_point)
        h = _dot(d, axis)
        x = _dot(d, u)
        y = _dot(d, w)
        r = math.hypot(x, y)
        ang = math.atan2(y, x)
        snap_idx = int(round(ang / step)) % segments
        snap_ang = snap_idx * step
        sx = math.cos(snap_ang) * r
        sy = math.sin(snap_ang) * r
        new = _add(
            _add(axis_point, _scale(axis, h)),
            _add(_scale(u, sx), _scale(w, sy)),
        )
        snapped.append((float(new[0]), float(new[1]), float(new[2])))
        snap_fracs.append(snap_idx / float(segments))
        angles.append((ang % (2.0 * math.pi)) / (2.0 * math.pi))

    def _unwrap_unit_values(vals):
        if not vals:
            return []
        out = [float(vals[0])]
        for val in vals[1:]:
            x = float(val)
            while x - out[-1] > 0.5:
                x -= 1.0
            while x - out[-1] < -0.5:
                x += 1.0
            out.append(x)
        return out

    def _wrap01(x):
        x = float(x) % 1.0
        return 0.0 if abs(x - 1.0) < 1e-9 else x

    # Find a UV component that behaves like the tyre tread/circumference axis.
    # Such a component has broad 0..1 variation around the ring; if it is later
    # averaged inside snapped angular bins, the tread texture collapses or jumps.
    uv_mode = (uv_reconstruction or "auto").lower()
    uv_fit = None
    if uvs and uv_mode in {"auto", "interpolate"}:
        order = sorted(range(min(n, len(uvs))), key=lambda i: angles[i])
        best = None
        for comp in (0, 1):
            seq = _unwrap_unit_values([uvs[i][comp] for i in order])
            if len(seq) < 2:
                continue
            rng = max(seq) - min(seq)
            if rng < 0.25:
                continue
            xs = [angles[i] for i in order]
            mx = sum(xs) / len(xs)
            my = sum(seq) / len(seq)
            varx = sum((x - mx) ** 2 for x in xs)
            if varx < 1e-12:
                continue
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, seq)) / varx
            intercept = my - slope * mx
            score = abs(slope) * rng
            if best is None or score > best[0]:
                best = (score, comp, slope, intercept, rng)
        if best and (uv_mode == "interpolate" or best[4] > 0.5):
            uv_fit = best[1:]

    def _avg_uv(member_indices, snap_frac):
        vals = [tuple(float(c) for c in uvs[i]) if i < len(uvs) else (0.0, 0.0) for i in member_indices]
        if not vals:
            return (0.0, 0.0)
        out = [0.0, 0.0]
        for comp in (0, 1):
            comp_vals = [v[comp] for v in vals]
            unwrapped = _unwrap_unit_values(comp_vals)
            out[comp] = sum(unwrapped) / len(unwrapped)
        if uv_fit is not None:
            comp, slope, intercept, _rng = uv_fit
            out[comp] = slope * snap_frac + intercept
        return (_wrap01(out[0]), _wrap01(out[1]))

    # weld by (snapped position, domain), then reconstruct/interpolate UVs for
    # each snapped angular segment instead of keeping whichever source UV arrived
    # first. Member UVs are seam-unwrapped before averaging to keep continuity.
    key_to_new = {}
    old_to_new = {}
    members = []
    out_vertices = []
    out_uvs = []
    out_domains = []
    out_weights = []
    for i in range(n):
        pos = snapped[i]
        key = (round(pos[0], precision), round(pos[1], precision), round(pos[2], precision), vertex_domains[i])
        if key in key_to_new:
            target = key_to_new[key]
            old_to_new[i] = target
            members[target].append(i)
            # merge weights conservatively
            merged = dict(out_weights[target])
            for gname, gw in group_weights[i].items():
                merged[gname] = max(float(gw), float(merged.get(gname, 0.0)))
            out_weights[target] = merged
        else:
            new_index = len(out_vertices)
            key_to_new[key] = new_index
            old_to_new[i] = new_index
            members.append([i])
            out_vertices.append(pos)
            out_uvs.append((0.0, 0.0))
            out_domains.append(vertex_domains[i])
            out_weights.append(dict(group_weights[i]))

    for oi, member_indices in enumerate(members):
        snap_frac = sum(snap_fracs[i] for i in member_indices) / len(member_indices)
        out_uvs[oi] = _avg_uv(member_indices, snap_frac)

    out_faces = []
    out_materials = []
    seen_faces = set()
    for face, mat in zip(faces, face_materials):
        remapped = tuple(old_to_new[v] for v in face)
        if len(set(remapped)) < 3:
            continue  # collapsed ring face
        canon = tuple(sorted(remapped))
        if canon in seen_faces:
            continue  # duplicate face produced by the weld
        seen_faces.add(canon)
        out_faces.append(remapped)
        out_materials.append(mat)

    return {
        "vertices": out_vertices,
        "uvs": out_uvs,
        "vertex_domains": out_domains,
        "group_weights": out_weights,
        "faces": out_faces,
        "face_materials": out_materials,
        "stats": {
            "reducer": "radial",
            "target_segments": segments,
            "start_faces": len(faces),
            "end_faces": len(out_faces),
            "start_vertices": n,
            "end_vertices": len(out_vertices),
            "actual_ratio": round(len(out_faces) / max(1, len(faces)), 4),
        },
    }


def _empty_like(segments, start_faces):
    return {
        "vertices": [],
        "uvs": [],
        "vertex_domains": [],
        "group_weights": [],
        "faces": [],
        "face_materials": [],
        "stats": {
            "reducer": "radial",
            "target_segments": segments,
            "start_faces": start_faces,
            "end_faces": 0,
            "start_vertices": 0,
            "end_vertices": 0,
            "actual_ratio": 0.0,
        },
    }


# ===========================================================================
# 3. Visibility / interior-geometry culler
# ===========================================================================

def _fibonacci_sphere(samples):
    """Evenly distributed unit directions on a sphere."""
    points = []
    if samples <= 0:
        return points
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    for i in range(samples):
        y = 1.0 - (i / float(max(1, samples - 1))) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points.append((math.cos(theta) * radius, y, math.sin(theta) * radius))
    return points


def _ray_triangle(orig, direction, v0, v1, v2, eps=1e-9):
    """Moller-Trumbore. Returns t>0 distance along ray or None."""
    e1 = _sub(v1, v0)
    e2 = _sub(v2, v0)
    pvec = _cross(direction, e2)
    det = _dot(e1, pvec)
    if -eps < det < eps:
        return None
    inv = 1.0 / det
    tvec = _sub(orig, v0)
    u = _dot(tvec, pvec) * inv
    if u < -1e-6 or u > 1.0 + 1e-6:
        return None
    qvec = _cross(tvec, e1)
    v = _dot(direction, qvec) * inv
    if v < -1e-6 or u + v > 1.0 + 1e-6:
        return None
    t = _dot(e2, qvec) * inv
    if t <= eps:
        return None
    return t


class _TriGrid:
    """Uniform spatial hash of triangles for ray broad-phase."""

    def __init__(self, vertices, faces, target_per_cell=4.0):
        self.vertices = vertices
        self.faces = faces
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        zs = [p[2] for p in vertices]
        self.min = (min(xs), min(ys), min(zs))
        self.max = (max(xs), max(ys), max(zs))
        extent = _sub(self.max, self.min)
        diag = max(1e-6, _length(extent))
        # aim for ~target_per_cell triangles/cell
        approx_cells = max(1, int((len(faces) / target_per_cell) ** (1.0 / 3.0)))
        self.cell = max(diag / (approx_cells * 3 + 1), diag * 1e-3)
        self.grid = defaultdict(list)
        for fi, face in enumerate(faces):
            self._insert(fi, face)

    def _key(self, p):
        return (
            int((p[0] - self.min[0]) // self.cell),
            int((p[1] - self.min[1]) // self.cell),
            int((p[2] - self.min[2]) // self.cell),
        )

    def _insert(self, fi, face):
        pts = [self.vertices[v] for v in face]
        lo = (min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts))
        hi = (max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts))
        klo = self._key(lo)
        khi = self._key(hi)
        for ix in range(klo[0], khi[0] + 1):
            for iy in range(klo[1], khi[1] + 1):
                for iz in range(klo[2], khi[2] + 1):
                    self.grid[(ix, iy, iz)].append(fi)

    def occluded(self, orig, target, skip_face, max_t):
        """True if any triangle (other than skip_face) blocks orig->target."""
        direction = _sub(target, orig)
        dist = _length(direction)
        if dist < 1e-9:
            return False
        direction = _scale(direction, 1.0 / dist)
        # walk cells along the segment with coarse DDA sampling
        steps = max(1, int(dist / self.cell) + 1)
        seen = set()
        tested = set()
        for s in range(steps + 1):
            point = _add(orig, _scale(direction, dist * s / steps))
            key = self._key(point)
            if key in seen:
                continue
            seen.add(key)
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    for iz in (-1, 0, 1):
                        cell = (key[0] + ix, key[1] + iy, key[2] + iz)
                        for fi in self.grid.get(cell, ()):
                            if fi == skip_face or fi in tested:
                                continue
                            tested.add(fi)
                            face = self.faces[fi]
                            t = _ray_triangle(
                                orig, direction,
                                self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]],
                            )
                            if t is not None and t < max_t - 1e-6:
                                return True
        return False


def visibility_cull(
    vertices,
    faces,
    samples=64,
    margin=1.5,
    return_mask=False,
):
    """Return the subset of ``faces`` visible from at least one exterior viewpoint.

    Viewpoints are placed on a sphere around the mesh (radius = bbox diagonal *
    ``margin``). A face is visible if, from some viewpoint, the straight segment
    to the face centroid is not blocked by another triangle. Interior faces that
    no exterior viewpoint can reach are culled.

    With ``return_mask=True`` returns ``(kept_faces, visible_flags)`` so callers
    can map results back to the original face order.
    """
    n_faces = len(faces)
    if n_faces == 0:
        return ([], []) if return_mask else []

    tris = [f for f in faces if len(f) >= 3 and max(f) < len(vertices)]
    if not tris:
        return ([], [False] * n_faces) if return_mask else []

    grid = _TriGrid(vertices, tris)
    center = _scale(_add(grid.min, grid.max), 0.5)
    radius = max(1e-3, _length(_sub(grid.max, grid.min)) * 0.5 * margin)
    viewpoints = [_add(center, _scale(d, radius)) for d in _fibonacci_sphere(samples)]

    visible = [False] * len(tris)
    centroids = []
    normals = []
    for face in tris:
        a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        centroids.append(_scale(_add(_add(a, b), c), 1.0 / 3.0))
        normals.append(_cross(_sub(b, a), _sub(c, a)))

    for fi, face in enumerate(tris):
        cen = centroids[fi]
        nrm = normals[fi]
        for vp in viewpoints:
            to_view = _sub(vp, cen)
            # only test viewpoints on the front side of the face (two-sided safe:
            # if normal is degenerate we still test)
            if _length(nrm) > 1e-12 and _dot(nrm, to_view) <= 0.0:
                continue
            max_t = _length(to_view)
            if not grid.occluded(cen, vp, fi, max_t):
                visible[fi] = True
                break

    # map back to original face indexing
    flags = []
    kept = []
    ti = 0
    for f in faces:
        if len(f) >= 3 and max(f) < len(vertices):
            vis = visible[ti]
            ti += 1
        else:
            vis = False
        flags.append(vis)
        if vis:
            kept.append(f)

    if return_mask:
        return kept, flags
    return kept


# ===========================================================================
# 4. Convex hull + approximate convex decomposition (collision proxies)
# ===========================================================================

def convex_hull_3d(points, eps=1e-9):
    """Incremental 3D convex hull.

    Returns ``(hull_vertices, hull_faces)`` with outward-facing triangles.
    Falls back to an axis-aligned bounding box for degenerate/coplanar input so
    a usable closed proxy is always produced.
    """
    pts = [tuple(float(c) for c in p) for p in points]
    unique = list({p for p in pts})
    if len(unique) < 4:
        return _bbox_hull(pts)

    # find an initial non-degenerate tetrahedron
    base = _initial_tetra(unique, eps)
    if base is None:
        return _bbox_hull(pts)
    i0, i1, i2, i3 = base

    def outward(a, b, c, apex):
        nrm = _cross(_sub(b, a), _sub(c, a))
        if _dot(nrm, _sub(apex, a)) > 0:  # apex on positive side -> flip
            return (a, c, b)
        return (a, b, c)

    p0, p1, p2, p3 = unique[i0], unique[i1], unique[i2], unique[i3]
    faces = [
        outward(p0, p1, p2, p3),
        outward(p0, p1, p3, p2),
        outward(p0, p2, p3, p1),
        outward(p1, p2, p3, p0),
    ]

    for p in unique:
        if p in (p0, p1, p2, p3):
            continue
        # find faces this point can see
        visible = []
        for f in faces:
            nrm = _cross(_sub(f[1], f[0]), _sub(f[2], f[0]))
            if _dot(nrm, _sub(p, f[0])) > eps:
                visible.append(f)
        if not visible:
            continue
        # collect horizon edges (edges on exactly one visible face)
        edge_count = defaultdict(int)
        edge_dir = {}
        for f in visible:
            for i in range(3):
                a = f[i]
                b = f[(i + 1) % 3]
                key = frozenset((a, b))
                edge_count[key] += 1
                edge_dir[key] = (a, b)
        horizon = [edge_dir[k] for k, c in edge_count.items() if c == 1]
        # remove visible faces, add new faces from horizon to p
        vis_set = set(map(id, visible))
        faces = [f for f in faces if id(f) not in vis_set]
        for (a, b) in horizon:
            faces.append((a, b, p))

    # re-orient all faces outward from hull centroid
    hull_pts = list({v for f in faces for v in f})
    cen = _centroid(hull_pts)
    oriented = []
    for f in faces:
        nrm = _cross(_sub(f[1], f[0]), _sub(f[2], f[0]))
        if _dot(nrm, _sub(_centroid(f), cen)) < 0:
            oriented.append((f[0], f[2], f[1]))
        else:
            oriented.append(f)

    index = {v: i for i, v in enumerate(hull_pts)}
    hull_faces = [(index[a], index[b], index[c]) for (a, b, c) in oriented]
    return hull_pts, hull_faces


def _initial_tetra(points, eps):
    n = len(points)
    i0 = 0
    # i1: farthest from i0
    i1 = max(range(n), key=lambda i: _length(_sub(points[i], points[i0])))
    if _length(_sub(points[i1], points[i0])) < eps:
        return None
    # i2: farthest from line i0-i1
    line = _sub(points[i1], points[i0])
    def line_dist(i):
        return _length(_cross(line, _sub(points[i], points[i0])))
    i2 = max(range(n), key=line_dist)
    if line_dist(i2) < eps:
        return None
    # i3: farthest from plane (i0,i1,i2)
    nrm = _cross(_sub(points[i1], points[i0]), _sub(points[i2], points[i0]))
    def plane_dist(i):
        return abs(_dot(nrm, _sub(points[i], points[i0])))
    i3 = max(range(n), key=plane_dist)
    if plane_dist(i3) < eps:
        return None
    return i0, i1, i2, i3


def _bbox_hull(points):
    if not points:
        return [], []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    verts = [
        (mn[0], mn[1], mn[2]), (mx[0], mn[1], mn[2]), (mx[0], mx[1], mn[2]), (mn[0], mx[1], mn[2]),
        (mn[0], mn[1], mx[2]), (mx[0], mn[1], mx[2]), (mx[0], mx[1], mx[2]), (mn[0], mx[1], mx[2]),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return verts, faces


def _kmeans(points, k, iterations=12, seed=1234):
    """Tiny deterministic k-means for spatial clustering."""
    n = len(points)
    k = max(1, min(k, n))
    # deterministic spread-out initial centroids (farthest-point-ish)
    centroids = [points[0]]
    while len(centroids) < k:
        best_p = None
        best_d = -1.0
        for p in points:
            d = min(_length(_sub(p, c)) for c in centroids)
            if d > best_d:
                best_d = d
                best_p = p
        centroids.append(best_p)

    assign = [0] * n
    for _ in range(iterations):
        changed = False
        for i, p in enumerate(points):
            best = min(range(k), key=lambda ci: _length(_sub(p, centroids[ci])))
            if best != assign[i]:
                assign[i] = best
                changed = True
        sums = [[0.0, 0.0, 0.0] for _ in range(k)]
        counts = [0] * k
        for i, p in enumerate(points):
            c = assign[i]
            sums[c][0] += p[0]
            sums[c][1] += p[1]
            sums[c][2] += p[2]
            counts[c] += 1
        for ci in range(k):
            if counts[ci] > 0:
                centroids[ci] = (sums[ci][0] / counts[ci], sums[ci][1] / counts[ci], sums[ci][2] / counts[ci])
        if not changed:
            break

    clusters = defaultdict(list)
    for i, p in enumerate(points):
        clusters[assign[i]].append(p)
    return [clusters[ci] for ci in range(k) if clusters[ci]]


def convex_decompose(vertices, faces=None, max_hulls=4, min_cluster=8):
    """Approximate convex decomposition into up to ``max_hulls`` convex hulls.

    Vertices are spatially clustered (k-means) and a convex hull is fitted to
    each cluster. This is a VHACD-style proxy: cheap, deterministic, and much
    tighter than a single bounding box for concave parts. Returns a list of
    ``{"vertices": [...], "faces": [...]}`` hulls.
    """
    pts = [tuple(float(c) for c in v) for v in vertices]
    if len(pts) < 4:
        verts, hfaces = _bbox_hull(pts)
        return [{"vertices": verts, "faces": hfaces}] if verts else []

    max_hulls = max(1, int(max_hulls))
    if max_hulls == 1:
        verts, hfaces = convex_hull_3d(pts)
        return [{"vertices": verts, "faces": hfaces}]

    clusters = _kmeans(pts, max_hulls)
    hulls = []
    for cluster in clusters:
        if len(cluster) < min_cluster:
            # too small to hull meaningfully; use its bbox
            verts, hfaces = _bbox_hull(cluster)
        else:
            verts, hfaces = convex_hull_3d(cluster)
        if verts and hfaces:
            hulls.append({"vertices": verts, "faces": hfaces})
    if not hulls:
        verts, hfaces = convex_hull_3d(pts)
        hulls = [{"vertices": verts, "faces": hfaces}]
    return hulls


def hull_volume(vertices, faces):
    """Signed volume magnitude of a closed triangular hull (for QA / ratios)."""
    if not faces:
        return 0.0
    total = 0.0
    for f in faces:
        a, b, c = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        total += _dot(a, _cross(b, c))
    return abs(total) / 6.0
