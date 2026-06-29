"""
Wedge-vertex QEM reducer for Vehicle Smart LOD.

A "wedge" (a.k.a. render vertex) is a unique combination of
position + UV + material/normal domain. Splitting the mesh into wedges before
reducing means UV seams, material borders and hard-normal borders are never
merged blindly: a collapse can only happen inside a single domain.

This rewrite replaces the original O(faces x edges) per-collapse loop with an
incremental priority-queue collapser:

    * vertex quadrics are built once (with boundary virtual planes),
    * every collapsible edge is scored once and pushed to a heap,
    * each iteration pops the cheapest valid edge, applies it, and only
      re-scores the handful of edges touched by that collapse.

This takes the reducer from ~quadratic to ~O(E log E) and makes it usable on
real vehicle parts. It also adds a manifold link-condition test so over-reduction
can no longer punch non-manifold holes in the topology.

Public API (triangulate_face, build_edges, simplify_wedge_mesh,
simplify_wedge_mesh_partitioned, plus the weld/compact helpers) is unchanged.
"""

import heapq
from collections import defaultdict

try:
    from . import qem_backend
    from . import vehicle_reducers
except Exception:  # pragma: no cover - standalone import
    import qem_backend
    import vehicle_reducers


# ---------------------------------------------------------------------------
# topology helpers (kept for backward compatibility / partition wrapper)
# ---------------------------------------------------------------------------

def triangulate_face(face):
    if len(face) == 3:
        return [tuple(face)]
    return [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]


def build_edges(faces):
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            if a != b:
                edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross3(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _len_sq3(a):
    return _dot3(a, a)


def face_normal_unscaled(vertices, face):
    a, b, c = (vertices[i] for i in face)
    return _cross3(_sub3(b, a), _sub3(c, a))


def collapse_breaks_faces(vertices, faces, keep, remove, target, min_area_sq=1e-18):
    """Legacy helper: does collapsing remove->keep flip/flatten any incident face?"""
    test_vertices = list(vertices)
    test_vertices[keep] = target
    for face in faces:
        if keep not in face and remove not in face:
            continue
        old_normal = face_normal_unscaled(vertices, face)
        updated = tuple(keep if v == remove else v for v in face)
        if len(set(updated)) < 3:
            continue
        new_normal = face_normal_unscaled(test_vertices, updated)
        if _len_sq3(new_normal) <= min_area_sq:
            return True, "face flattened"
        if _len_sq3(old_normal) > min_area_sq and _dot3(old_normal, new_normal) < 0.0:
            return True, "face flipped"
    return False, None


def border_vertices(faces):
    counts = defaultdict(int)
    for face in faces:
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            counts[tuple(sorted((a, b)))] += 1
    locked = set()
    for edge, count in counts.items():
        if count == 1:
            locked.update(edge)
    return locked


def partition_faces_by_connectivity(faces):
    vertex_to_faces = defaultdict(list)
    for face_index, face in enumerate(faces):
        for vertex in face:
            vertex_to_faces[vertex].append(face_index)

    seen = set()
    partitions = []
    for start in range(len(faces)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        current = []
        while stack:
            face_index = stack.pop()
            current.append(face_index)
            for vertex in faces[face_index]:
                for linked_face in vertex_to_faces[vertex]:
                    if linked_face not in seen:
                        seen.add(linked_face)
                        stack.append(linked_face)
        partitions.append(current)
    return partitions


def remove_degenerate_faces(faces, payloads=None):
    cleaned = []
    cleaned_payloads = [] if payloads is not None else None
    for index, face in enumerate(faces):
        if len(set(face)) == 3:
            cleaned.append(face)
            if cleaned_payloads is not None:
                cleaned_payloads.append(payloads[index])
    if cleaned_payloads is not None:
        return cleaned, cleaned_payloads
    return cleaned


def compact_mesh(vertices, uvs, domains, group_weights, faces, face_materials):
    used = sorted({v for face in faces for v in face})
    remap = {old: new for new, old in enumerate(used)}
    return (
        [vertices[i] for i in used],
        [uvs[i] for i in used],
        [domains[i] for i in used],
        [group_weights[i] for i in used],
        [tuple(remap[v] for v in face) for face in faces],
        list(face_materials),
        remap,
    )


def _rounded(values, precision):
    return tuple(round(float(value), precision) for value in values)


def merge_group_weights(a, b):
    merged = dict(a)
    for group_name, weight in b.items():
        merged[group_name] = max(float(weight), float(merged.get(group_name, 0.0)))
    return merged


def safe_weld_same_domain(vertices, uvs, domains, group_weights, faces, face_materials, precision=6):
    key_to_new = {}
    old_to_new = {}
    new_vertices = []
    new_uvs = []
    new_domains = []
    new_group_weights = []

    for index, (vertex, uv, domain, weights) in enumerate(zip(vertices, uvs, domains, group_weights)):
        key = (_rounded(vertex, precision), _rounded(uv, precision), domain)
        if key in key_to_new:
            target = key_to_new[key]
            old_to_new[index] = target
            new_group_weights[target] = merge_group_weights(new_group_weights[target], weights)
        else:
            old_to_new[index] = len(new_vertices)
            key_to_new[key] = len(new_vertices)
            new_vertices.append(vertex)
            new_uvs.append(uv)
            new_domains.append(domain)
            new_group_weights.append(dict(weights))

    welded_faces = [tuple(old_to_new[v] for v in face) for face in faces]
    welded_faces, welded_materials = remove_degenerate_faces(welded_faces, face_materials)
    return compact_mesh(new_vertices, new_uvs, new_domains, new_group_weights, welded_faces, welded_materials)


# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------

def _normalise_inputs(vertices, uvs, faces, face_materials, vertex_domains, group_weights):
    vertices = [tuple(float(c) for c in v) for v in vertices]
    n = len(vertices)

    # pad/truncate UVs to match vertex count
    uvs = [tuple(float(c) for c in uv) for uv in uvs]
    if len(uvs) < n:
        uvs = uvs + [(0.0, 0.0)] * (n - len(uvs))
    elif len(uvs) > n:
        uvs = uvs[:n]

    if vertex_domains is None:
        vertex_domains = [("default",) for _ in range(n)]
    vertex_domains = [tuple(d) for d in vertex_domains]
    if len(vertex_domains) < n:
        vertex_domains = vertex_domains + [("default",)] * (n - len(vertex_domains))
    elif len(vertex_domains) > n:
        vertex_domains = vertex_domains[:n]

    if group_weights is None:
        group_weights = [{} for _ in range(n)]
    group_weights = [dict(w) for w in group_weights]
    if len(group_weights) < n:
        group_weights = group_weights + [{} for _ in range(n - len(group_weights))]
    elif len(group_weights) > n:
        group_weights = group_weights[:n]

    faces = [tuple(int(v) for v in face) for face in faces]
    if face_materials is None:
        face_materials = [0 for _ in faces]
    face_materials = list(face_materials)
    if len(face_materials) < len(faces):
        face_materials = face_materials + [0] * (len(faces) - len(face_materials))
    elif len(face_materials) > len(faces):
        face_materials = face_materials[:len(faces)]

    # drop faces that reference out-of-range vertices or are degenerate
    clean_faces = []
    clean_materials = []
    for face, mat in zip(faces, face_materials):
        if len(face) != 3:
            for tri in triangulate_face(face):
                if all(0 <= v < n for v in tri) and len(set(tri)) == 3:
                    clean_faces.append(tri)
                    clean_materials.append(mat)
            continue
        if all(0 <= v < n for v in face) and len(set(face)) == 3:
            clean_faces.append(face)
            clean_materials.append(mat)

    return vertices, uvs, clean_faces, clean_materials, vertex_domains, group_weights


# ---------------------------------------------------------------------------
# incremental collapse machinery
# ---------------------------------------------------------------------------

def _link_condition_ok(vert_faces, faces, u, v):
    """True if collapsing edge (u, v) preserves a manifold.

    The link condition: the shared one-ring of u and v must equal exactly the
    set of third vertices of triangles on the edge (1 for a boundary edge,
    2 for an interior edge). Anything more would create a non-manifold fan.
    """
    nu = set()
    for fi in vert_faces[u]:
        face = faces[fi]
        if face is None:
            continue
        nu.update(w for w in face if w != u)
    nv = set()
    for fi in vert_faces[v]:
        face = faces[fi]
        if face is None:
            continue
        nv.update(w for w in face if w != v)

    shared = (nu & nv) - {u, v}

    edge_third = set()
    for fi in vert_faces[u]:
        face = faces[fi]
        if face is None:
            continue
        if v in face:
            edge_third.update(w for w in face if w != u and w != v)

    return shared == edge_third


def _incident_faces_break(faces, vert_faces, positions, keep, remove, target,
                          min_area_sq=1e-16):
    """Flip / flatten test limited to faces incident to keep or remove."""
    touched = vert_faces[keep] | vert_faces[remove]
    for fi in touched:
        face = faces[fi]
        if face is None:
            continue
        updated = tuple(keep if x == remove else x for x in face)
        if len(set(updated)) < 3:
            continue  # this face collapses away, fine
        a, b, c = face
        old_n = _cross3(_sub3(positions[b], positions[a]), _sub3(positions[c], positions[a]))
        pa = target if updated[0] == keep else positions[updated[0]]
        pb = target if updated[1] == keep else positions[updated[1]]
        pc = target if updated[2] == keep else positions[updated[2]]
        new_n = _cross3(_sub3(pb, pa), _sub3(pc, pa))
        if _len_sq3(new_n) <= min_area_sq:
            return True
        if _len_sq3(old_n) > min_area_sq and _dot3(old_n, new_n) < 0.0:
            return True
    return False


def simplify_wedge_mesh(
    vertices,
    uvs,
    faces,
    face_materials=None,
    vertex_domains=None,
    group_weights=None,
    target_ratio=0.7,
    uv_weight=25.0,
    uv_distance_limit=0.08,
    lock_border_vertices=True,
    allow_domain_crossing=False,
    safe_weld=True,
    reject_face_flips=True,
    max_iterations=100000,
    boundary_weight=2.0,
    preserve_manifold=True,
    target_faces=None,
):
    vertices, uvs, faces, face_materials, vertex_domains, group_weights = _normalise_inputs(
        vertices, uvs, faces, face_materials, vertex_domains, group_weights
    )

    start_faces = len(faces)
    if target_faces is not None:
        # absolute triangle-budget target (AAA LOD budgeting) overrides ratio
        target_faces = max(1, min(start_faces, int(target_faces)))
    else:
        target_faces = max(1, int(round(start_faces * target_ratio)))
    rejected_reasons = defaultdict(int)

    if start_faces == 0:
        return _package_result(
            vertices, uvs, vertex_domains, group_weights, [], [],
            start_faces, target_faces, 0, rejected_reasons,
            lock_border_vertices, allow_domain_crossing, safe_weld,
            reject_face_flips, uv_weight, uv_distance_limit, safe_weld,
        )

    positions = list(vertices)
    quadrics = qem_backend.build_vertex_quadrics(positions, faces, boundary_weight=boundary_weight)

    # mutable face table; None means deleted
    face_table = [tuple(f) for f in faces]
    vert_faces = defaultdict(set)
    for fi, face in enumerate(face_table):
        for v in face:
            vert_faces[v].add(fi)
    face_count = len(face_table)

    locked = border_vertices(faces) if lock_border_vertices else set()
    vertex_alive = [True] * len(positions)

    edge_version = defaultdict(int)
    heap = []

    def neighbours(v):
        out = set()
        for fi in vert_faces[v]:
            face = face_table[fi]
            if face is None:
                continue
            out.update(w for w in face if w != v)
        return out

    def score_and_push(a, b):
        if a == b or not vertex_alive[a] or not vertex_alive[b]:
            return
        if not allow_domain_crossing and vertex_domains[a] != vertex_domains[b]:
            return
        candidate, _reason = qem_backend.collapse_candidate(
            positions,
            quadrics,
            (a, b),
            protected_vertices=locked,
            protected_edges=set(),
            vertex_uvs={a: uvs[a], b: uvs[b]},
            uv_locked_vertices=set(),
            uv_weight=uv_weight,
            uv_distance_limit=uv_distance_limit,
        )
        if not candidate:
            return
        key = (a, b) if a < b else (b, a)
        edge_version[key] += 1
        heapq.heappush(heap, (candidate["cost"], edge_version[key], key, candidate))

    # seed the heap
    for edge in build_edges(faces):
        score_and_push(edge[0], edge[1])

    collapsed = 0
    while face_count > target_faces and collapsed < max_iterations and heap:
        cost, ver, key, candidate = heapq.heappop(heap)
        a, b = key
        if not vertex_alive[a] or not vertex_alive[b]:
            continue
        if edge_version[key] != ver:
            continue  # stale entry, a newer score exists

        # decide keep/remove + target honouring border locks
        a_locked = a in locked
        b_locked = b in locked
        if a_locked and b_locked:
            continue
        if a_locked:
            keep, remove = a, b
            target = positions[a]
            target_uv = uvs[a]
        elif b_locked:
            keep, remove = b, a
            target = positions[b]
            target_uv = uvs[b]
        else:
            keep, remove = a, b
            target = candidate.get("target_raw") or tuple(candidate["target"])
            target_uv = candidate.get("target_uv_raw")
            if target_uv is None:
                target_uv = ((uvs[a][0] + uvs[b][0]) * 0.5, (uvs[a][1] + uvs[b][1]) * 0.5)

        if preserve_manifold and not _link_condition_ok(vert_faces, face_table, keep, remove):
            rejected_reasons["non-manifold link"] += 1
            continue

        if reject_face_flips and _incident_faces_break(
            face_table, vert_faces, positions, keep, remove, target
        ):
            rejected_reasons["face flip/flatten"] += 1
            continue

        # ---- apply collapse ----
        positions[keep] = tuple(target)
        uvs[keep] = tuple(target_uv)
        quadrics[keep] = qem_backend._quadric_add(quadrics[keep], quadrics[remove])
        group_weights[keep] = merge_group_weights(group_weights[keep], group_weights[remove])

        affected = set(vert_faces[remove])
        for fi in affected:
            face = face_table[fi]
            if face is None:
                continue
            updated = tuple(keep if x == remove else x for x in face)
            if len(set(updated)) < 3:
                face_table[fi] = None
                for v in face:
                    vert_faces[v].discard(fi)
                face_count -= 1
            else:
                face_table[fi] = updated
                vert_faces[keep].add(fi)
        vert_faces[remove] = set()
        vertex_alive[remove] = False

        # re-score every edge now incident to keep
        for n in neighbours(keep):
            score_and_push(keep, n)
        collapsed += 1

    final_faces = [face_table[fi] for fi in range(len(face_table)) if face_table[fi] is not None]
    final_materials = [face_materials[fi] for fi in range(len(face_table)) if face_table[fi] is not None]

    return _package_result(
        positions, uvs, vertex_domains, group_weights, final_faces, final_materials,
        start_faces, target_faces, collapsed, rejected_reasons,
        lock_border_vertices, allow_domain_crossing, safe_weld,
        reject_face_flips, uv_weight, uv_distance_limit, safe_weld,
    )


def _package_result(positions, uvs, domains, group_weights, faces, materials,
                    start_faces, target_faces, collapsed, rejected_reasons,
                    lock_border_vertices, allow_domain_crossing, safe_weld_flag,
                    reject_face_flips, uv_weight, uv_distance_limit, do_weld):
    pre_weld_vertices = len({v for face in faces for v in face})
    if do_weld:
        out = safe_weld_same_domain(positions, uvs, domains, group_weights, faces, materials)
    else:
        out = compact_mesh(positions, uvs, domains, group_weights, faces, materials)
    o_vertices, o_uvs, o_domains, o_weights, o_faces, o_materials, _ = out

    return {
        "vertices": o_vertices,
        "uvs": o_uvs,
        "vertex_domains": o_domains,
        "group_weights": o_weights,
        "faces": o_faces,
        "face_materials": o_materials,
        "stats": {
            "start_faces": start_faces,
            "target_faces": target_faces,
            "end_faces": len(o_faces),
            "collapsed_edges": collapsed,
            "target_ratio": round(target_faces / max(1, start_faces), 4),
            "actual_ratio": round(len(o_faces) / max(1, start_faces), 4),
            "rejected_reasons": dict(rejected_reasons),
            "lock_border_vertices": lock_border_vertices,
            "allow_domain_crossing": allow_domain_crossing,
            "safe_weld": safe_weld_flag,
            "reject_face_flips": reject_face_flips,
            "pre_weld_vertices": pre_weld_vertices,
            "post_weld_vertices": len(o_vertices),
            "welded_vertices": max(0, pre_weld_vertices - len(o_vertices)),
            "uv_weight": uv_weight,
            "uv_distance_limit": uv_distance_limit,
        },
    }


def simplify_wedge_mesh_partitioned(
    vertices,
    uvs,
    faces,
    face_materials=None,
    vertex_domains=None,
    group_weights=None,
    target_ratio=0.7,
    uv_weight=25.0,
    uv_distance_limit=0.08,
    lock_border_vertices=True,
    allow_domain_crossing=False,
    safe_weld=True,
    reject_face_flips=True,
    max_iterations=100000,
    boundary_weight=2.0,
    preserve_manifold=True,
    target_faces=None,
):
    vertices, uvs, faces, face_materials, vertex_domains, group_weights = _normalise_inputs(
        vertices, uvs, faces, face_materials, vertex_domains, group_weights
    )

    partitions = partition_faces_by_connectivity(faces)

    # When an absolute triangle budget is requested, split it across partitions
    # proportionally to their size so the whole mesh lands on the budget.
    partition_targets = None
    if target_faces is not None:
        partition_targets = vehicle_reducers.allocate_triangle_budget(
            [len(face_indices) for face_indices in partitions],
            int(target_faces),
        )
    out_vertices = []
    out_uvs = []
    out_domains = []
    out_group_weights = []
    out_faces = []
    out_materials = []
    part_stats = []

    for part_index, face_indices in enumerate(partitions):
        used = sorted({vertex for face_index in face_indices for vertex in faces[face_index]})
        old_to_local = {old: new for new, old in enumerate(used)}
        local_vertices = [vertices[i] for i in used]
        local_uvs = [uvs[i] for i in used]
        local_domains = [vertex_domains[i] for i in used]
        local_weights = [group_weights[i] for i in used]
        local_faces = [tuple(old_to_local[v] for v in faces[i]) for i in face_indices]
        local_materials = [face_materials[i] for i in face_indices]

        local_target_faces = (
            partition_targets[part_index] if partition_targets is not None else None
        )
        reduced = simplify_wedge_mesh(
            local_vertices,
            local_uvs,
            local_faces,
            face_materials=local_materials,
            vertex_domains=local_domains,
            group_weights=local_weights,
            target_ratio=target_ratio,
            uv_weight=uv_weight,
            uv_distance_limit=uv_distance_limit,
            lock_border_vertices=lock_border_vertices,
            allow_domain_crossing=allow_domain_crossing,
            safe_weld=safe_weld,
            reject_face_flips=reject_face_flips,
            max_iterations=max_iterations,
            boundary_weight=boundary_weight,
            preserve_manifold=preserve_manifold,
            target_faces=local_target_faces,
        )

        offset = len(out_vertices)
        out_vertices.extend(reduced["vertices"])
        out_uvs.extend(reduced["uvs"])
        out_domains.extend(reduced["vertex_domains"])
        out_group_weights.extend(reduced["group_weights"])
        out_faces.extend([tuple(v + offset for v in face) for face in reduced["faces"]])
        out_materials.extend(reduced["face_materials"])
        stats = dict(reduced["stats"])
        stats["partition_index"] = part_index
        stats["source_face_count"] = len(face_indices)
        part_stats.append(stats)

    pre_weld_vertices = len({v for face in out_faces for v in face})
    if safe_weld:
        out_vertices, out_uvs, out_domains, out_group_weights, out_faces, out_materials, _ = safe_weld_same_domain(
            out_vertices,
            out_uvs,
            out_domains,
            out_group_weights,
            out_faces,
            out_materials,
        )

    return {
        "vertices": out_vertices,
        "uvs": out_uvs,
        "vertex_domains": out_domains,
        "group_weights": out_group_weights,
        "faces": out_faces,
        "face_materials": out_materials,
        "stats": {
            "partitioned": True,
            "partition_count": len(partitions),
            "start_faces": len(faces),
            "target_ratio": target_ratio,
            "end_faces": len(out_faces),
            "actual_ratio": round(len(out_faces) / max(1, len(faces)), 4),
            "pre_weld_vertices": pre_weld_vertices,
            "post_weld_vertices": len(out_vertices),
            "welded_vertices": max(0, pre_weld_vertices - len(out_vertices)),
            "partitions": part_stats[:50],
            "partition_stats_truncated": len(part_stats) > 50,
        },
    }
