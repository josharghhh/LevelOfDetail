"""
Quadric Error Metrics (QEM) backend for Vehicle Smart LOD.

This module is pure Python with no Blender dependency so it can be unit tested
standalone (see SMOKE_TEST.py and tests/).

Compared with the original prototype this version adds the two fixes that matter
most for production quality:

1. Bounded optimal-vertex placement. The classic QEM "solve the 3x3 system"
   step can return a point arbitrarily far from the edge when the quadric is
   ill-conditioned (flat or cylindrical regions). That produces spikes and
   doubles bounding boxes. We now reject any solved vertex that lands outside a
   bounded neighbourhood of the edge and fall back to the best endpoint/midpoint.

2. Boundary-preserving virtual planes. Open mesh borders (panel edges, cylinder
   caps) get an extra perpendicular constraint plane added to their endpoint
   quadrics so the border is held in place by cost instead of needing a hard
   lock that stops all reduction.

The public API (build_vertex_quadrics, optimal_target, collapse_candidate,
analyse_candidates) is unchanged so existing callers keep working.
"""

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# small vector helpers
# ---------------------------------------------------------------------------

def _vec_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_len(a):
    return math.sqrt(_vec_dot(a, a))


def _vec_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _normalize(a):
    length = _vec_len(a)
    if length < 1e-12:
        return None
    return (a[0] / length, a[1] / length, a[2] / length)


# ---------------------------------------------------------------------------
# 4x4 quadric stored as a flat list of 16 floats (row-major)
# ---------------------------------------------------------------------------

def _quadric_zero():
    return [0.0] * 16


def _quadric_add(a, b):
    return [a[i] + b[i] for i in range(16)]


def _quadric_scaled_add(a, b, scale):
    return [a[i] + b[i] * scale for i in range(16)]


def _quadric_from_plane(a, b, c, d):
    p = (a, b, c, d)
    return [p[row] * p[col] for row in range(4) for col in range(4)]


def _quadric_eval(q, v):
    x = (v[0], v[1], v[2], 1.0)
    total = 0.0
    for row in range(4):
        base = row * 4
        xr = x[row]
        total += xr * (
            q[base] * x[0] + q[base + 1] * x[1] + q[base + 2] * x[2] + q[base + 3] * x[3]
        )
    return total


# ---------------------------------------------------------------------------
# 3x3 solve via Cramer's rule
# ---------------------------------------------------------------------------

def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _solve3(a, b):
    det = _det3(a)
    if abs(det) < 1e-10:
        return None

    def replaced(col):
        return [
            [b[row] if idx == col else a[row][idx] for idx in range(3)]
            for row in range(3)
        ]

    return (
        _det3(replaced(0)) / det,
        _det3(replaced(1)) / det,
        _det3(replaced(2)) / det,
    )


# ---------------------------------------------------------------------------
# plane / quadric construction
# ---------------------------------------------------------------------------

def _face_plane(vertices, face):
    if len(face) < 3:
        return None
    p0 = vertices[face[0]]
    for i in range(1, len(face) - 1):
        p1 = vertices[face[i]]
        p2 = vertices[face[i + 1]]
        normal = _vec_cross(_vec_sub(p1, p0), _vec_sub(p2, p0))
        length = _vec_len(normal)
        if length > 1e-12:
            n = (normal[0] / length, normal[1] / length, normal[2] / length)
            return (n[0], n[1], n[2], -_vec_dot(n, p0))
    return None


def _boundary_edges(faces):
    """Return the set of edges used by exactly one triangle (open borders)."""
    counts = defaultdict(int)
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            if a != b:
                counts[tuple(sorted((a, b)))] += 1
    return {edge for edge, count in counts.items() if count == 1}


def build_vertex_quadrics(vertices, faces, boundary_weight=0.0):
    """Accumulate face-plane quadrics per vertex.

    If ``boundary_weight`` > 0 we also add a "virtual plane" quadric along every
    open boundary edge. The virtual plane is perpendicular to the incident face
    and contains the edge, so moving a boundary vertex off the border costs
    energy. This keeps panel edges and open caps in place without a hard lock.
    """
    quadrics = [_quadric_zero() for _ in vertices]

    # face-to-plane lookup so boundary virtual planes can reference the face normal
    face_planes = []
    for face in faces:
        plane = _face_plane(vertices, face)
        face_planes.append(plane)
        if not plane:
            continue
        q = _quadric_from_plane(*plane)
        for vi in face:
            quadrics[vi] = _quadric_add(quadrics[vi], q)

    if boundary_weight > 0.0:
        # map each boundary edge to one incident face normal
        edge_face = {}
        for fi, face in enumerate(faces):
            if not face_planes[fi]:
                continue
            n = len(face)
            for i in range(n):
                a = face[i]
                b = face[(i + 1) % n]
                key = tuple(sorted((a, b)))
                edge_face.setdefault(key, fi)

        for edge in _boundary_edges(faces):
            fi = edge_face.get(edge)
            if fi is None or not face_planes[fi]:
                continue
            a, b = edge
            pa, pb = vertices[a], vertices[b]
            edge_dir = _normalize(_vec_sub(pb, pa))
            if edge_dir is None:
                continue
            face_n = (face_planes[fi][0], face_planes[fi][1], face_planes[fi][2])
            # plane perpendicular to the face, containing the edge
            virt_n = _normalize(_vec_cross(edge_dir, face_n))
            if virt_n is None:
                continue
            d = -_vec_dot(virt_n, pa)
            q = _quadric_from_plane(virt_n[0], virt_n[1], virt_n[2], d)
            q = [val * boundary_weight for val in q]
            quadrics[a] = _quadric_add(quadrics[a], q)
            quadrics[b] = _quadric_add(quadrics[b], q)

    return quadrics


# ---------------------------------------------------------------------------
# optimal target placement (now bounded)
# ---------------------------------------------------------------------------

def _within_bounds(point, p0, p1, max_offset_factor):
    """Reject a solved vertex that flies far away from the collapsing edge."""
    mid = _vec_mid(p0, p1)
    edge_len = _vec_len(_vec_sub(p1, p0))
    # allow a generous neighbourhood but not unbounded extrapolation
    limit = max(edge_len * max_offset_factor, 1e-6)
    return _vec_len(_vec_sub(point, mid)) <= limit


def optimal_target(q, p0, p1, max_offset_factor=4.0):
    """Best collapse position for the merged quadric ``q``.

    Tries the analytic QEM minimiser first, but only accepts it when it lands
    in a bounded neighbourhood of the edge. Otherwise it returns whichever of
    {p0, p1, midpoint} has the lowest quadric error. This is the single most
    important correctness fix versus the original prototype.
    """
    a = [
        [q[0], q[1], q[2]],
        [q[4], q[5], q[6]],
        [q[8], q[9], q[10]],
    ]
    b = (-q[3], -q[7], -q[11])
    solved = _solve3(a, b)
    if solved is not None and _within_bounds(solved, p0, p1, max_offset_factor):
        return solved, "optimal"

    candidates = [p0, p1, _vec_mid(p0, p1)]
    best = min(candidates, key=lambda pos: _quadric_eval(q, pos))
    return best, "fallback"


def _vec2_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# candidate evaluation (non-destructive planning + reducer shared core)
# ---------------------------------------------------------------------------

def collapse_candidate(
    vertices,
    quadrics,
    edge,
    protected_vertices,
    protected_edges,
    vertex_uvs=None,
    uv_locked_vertices=None,
    uv_weight=0.0,
    uv_distance_limit=None,
    max_offset_factor=4.0,
):
    a, b = edge
    key = tuple(sorted((a, b)))

    if key in protected_edges:
        return None, "protected edge"

    uv_locked_vertices = uv_locked_vertices or set()
    if a in uv_locked_vertices or b in uv_locked_vertices:
        return None, "uv wedge vertex locked"

    uv_penalty = 0.0
    target_uv = None
    if vertex_uvs is not None:
        uv_a = vertex_uvs.get(a)
        uv_b = vertex_uvs.get(b)
        if uv_a is None or uv_b is None:
            return None, "missing uv"
        uv_distance = _vec2_dist(uv_a, uv_b)
        if uv_distance_limit is not None and uv_distance > uv_distance_limit:
            return None, "uv distance too high"
        target_uv = ((uv_a[0] + uv_b[0]) * 0.5, (uv_a[1] + uv_b[1]) * 0.5)
        uv_penalty = uv_weight * uv_distance * uv_distance

    a_locked = a in protected_vertices
    b_locked = b in protected_vertices
    if a_locked and b_locked:
        return None, "both vertices locked"

    q = _quadric_add(quadrics[a], quadrics[b])

    if a_locked:
        target = vertices[a]
        mode = "target locked vertex a"
    elif b_locked:
        target = vertices[b]
        mode = "target locked vertex b"
    else:
        target, mode = optimal_target(q, vertices[a], vertices[b], max_offset_factor)

    geometry_cost = float(_quadric_eval(q, target))
    # quadric error can be slightly negative from float noise; clamp
    if geometry_cost < 0.0:
        geometry_cost = 0.0

    return {
        "edge": [int(a), int(b)],
        "target": [round(float(v), 6) for v in target],
        "target_raw": tuple(float(v) for v in target),
        "geometry_cost": geometry_cost,
        "uv_penalty": float(uv_penalty),
        "cost": float(geometry_cost + uv_penalty),
        "target_uv": [round(float(v), 6) for v in target_uv] if target_uv else None,
        "target_uv_raw": tuple(float(v) for v in target_uv) if target_uv else None,
        "mode": mode,
        "locked_vertices": int(a_locked) + int(b_locked),
    }, None


def analyse_candidates(
    vertices,
    faces,
    edges,
    protected_vertices=None,
    protected_edges=None,
    limit=100,
    vertex_uvs=None,
    uv_locked_vertices=None,
    uv_weight=0.0,
    uv_distance_limit=None,
    boundary_weight=0.0,
):
    protected_vertices = set(protected_vertices or [])
    protected_edges = {tuple(sorted(edge)) for edge in (protected_edges or [])}
    uv_locked_vertices = set(uv_locked_vertices or [])
    quadrics = build_vertex_quadrics(vertices, faces, boundary_weight=boundary_weight)

    accepted = []
    rejected = {}
    for edge in edges:
        candidate, reason = collapse_candidate(
            vertices,
            quadrics,
            tuple(edge),
            protected_vertices,
            protected_edges,
            vertex_uvs=vertex_uvs,
            uv_locked_vertices=uv_locked_vertices,
            uv_weight=uv_weight,
            uv_distance_limit=uv_distance_limit,
        )
        if candidate:
            accepted.append(candidate)
        else:
            rejected[reason] = rejected.get(reason, 0) + 1

    accepted.sort(key=lambda item: item["cost"])
    # strip the internal raw fields from the reported plan
    for item in accepted[:limit]:
        item.pop("target_raw", None)
        item.pop("target_uv_raw", None)
    return {
        "candidate_count": len(accepted),
        "rejected_count": sum(rejected.values()),
        "rejected_reasons": rejected,
        "best_candidates": accepted[:limit],
    }
