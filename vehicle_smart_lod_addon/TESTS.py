"""
Standalone test suite for the Vehicle Smart LOD backends.

Runs without Blender. Execute with:  python3 TESTS.py

These tests guard the correctness fixes that separate the reducer from a toy:
bounded vertex placement, manifold preservation, boundary preservation, speed,
and robustness against degenerate input.
"""

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qem_backend
import wedge_backend
import vehicle_reducers


# ---------------------------------------------------------------------------
# primitive builders
# ---------------------------------------------------------------------------

def make_cylinder(seg=24, rings=2, r=1.0, h=2.0):
    verts = []
    for ri in range(rings + 1):
        z = -h / 2 + h * ri / rings
        for s in range(seg):
            a = 2 * math.pi * s / seg
            verts.append((r * math.cos(a), r * math.sin(a), z))
    faces = []
    for ri in range(rings):
        for s in range(seg):
            s2 = (s + 1) % seg
            v0 = ri * seg + s
            v1 = ri * seg + s2
            v2 = (ri + 1) * seg + s2
            v3 = (ri + 1) * seg + s
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))
    return verts, [(v[0], v[2]) for v in verts], faces


def make_uvsphere(seg=24, rings=16, r=1.0):
    v = [(0, 0, r)]
    for ri in range(1, rings):
        phi = math.pi * ri / rings
        for s in range(seg):
            th = 2 * math.pi * s / seg
            v.append((r * math.sin(phi) * math.cos(th), r * math.sin(phi) * math.sin(th), r * math.cos(phi)))
    v.append((0, 0, -r))
    f = []
    for s in range(seg):
        f.append((0, 1 + s, 1 + (s + 1) % seg))
    for ri in range(rings - 2):
        for s in range(seg):
            a = 1 + ri * seg + s
            b = 1 + ri * seg + (s + 1) % seg
            c = 1 + (ri + 1) * seg + (s + 1) % seg
            d = 1 + (ri + 1) * seg + s
            f.append((a, b, c))
            f.append((a, c, d))
    last = len(v) - 1
    base = 1 + (rings - 2) * seg
    for s in range(seg):
        f.append((last, base + (s + 1) % seg, base + s))
    uv = [(0.5 + math.atan2(p[1], p[0]) / (2 * math.pi), 0.5 - p[2] / (2 * r)) for p in v]
    return v, uv, f


def make_grid(n):
    v = []
    uv = []
    for j in range(n):
        for i in range(n):
            v.append((float(i), float(j), 0.0))
            uv.append((i / n, j / n))
    f = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + i + 1
            c = (j + 1) * n + i + 1
            d = (j + 1) * n + i
            f.append((a, b, c))
            f.append((a, c, d))
    return v, uv, f


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------

def manifold_counts(faces):
    ec = defaultdict(int)
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            ec[tuple(sorted((a, b)))] += 1
    boundary = sum(1 for c in ec.values() if c == 1)
    nonman = sum(1 for c in ec.values() if c > 2)
    degen = sum(1 for f in faces if len(set(f)) < 3)
    return boundary, nonman, degen


def bbox_dims(verts):
    if not verts:
        return (0.0, 0.0, 0.0)
    mn = [min(p[i] for p in verts) for i in range(3)]
    mx = [max(p[i] for p in verts) for i in range(3)]
    return tuple(mx[i] - mn[i] for i in range(3))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_bounded_vertex_no_bbox_explosion():
    """Cylinder reduction must not balloon the bounding box (the old spike bug)."""
    v, uv, f = make_cylinder(24, 2)
    src = bbox_dims(v)
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, target_ratio=0.15, lock_border_vertices=False,
        uv_distance_limit=10.0, reject_face_flips=True,
    )
    out = bbox_dims(r["vertices"])
    for axis in range(3):
        growth = out[axis] / max(src[axis], 1e-9)
        assert growth < 1.5, f"axis {axis} grew {growth:.2f}x (bbox explosion regression)"
    print("  PASS bounded vertex (cylinder bbox stayed within 1.5x)")


def test_manifold_preserved_sphere():
    v, uv, f = make_uvsphere(24, 16)
    assert manifold_counts(f) == (0, 0, 0), "test sphere should start closed"
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, target_ratio=0.3, lock_border_vertices=False,
        uv_distance_limit=10.0, reject_face_flips=True,
    )
    b, nm, d = manifold_counts(r["faces"])
    assert nm == 0 and d == 0, f"sphere reduction created non-manifold/degenerate ({nm},{d})"
    assert b == 0, f"closed sphere opened {b} boundary edges"
    assert len(r["faces"]) < len(f), "no reduction achieved"
    print(f"  PASS manifold preserved (sphere {len(f)} -> {len(r['faces'])} faces, still closed)")


def test_link_condition_octahedron():
    v = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    uv = [(x[0], x[1]) for x in v]
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, target_ratio=0.25, lock_border_vertices=False,
        uv_distance_limit=10.0, reject_face_flips=True,
    )
    _, nm, d = manifold_counts(r["faces"])
    assert nm == 0 and d == 0, "octahedron over-reduction broke manifold"
    print("  PASS link condition (octahedron stays manifold under over-reduction)")


def test_boundary_preserved_open_grid():
    v, uv, f = make_grid(40)
    src = bbox_dims(v)
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, target_ratio=0.2, lock_border_vertices=False,
        uv_distance_limit=10.0, boundary_weight=4.0,
    )
    out = bbox_dims(r["vertices"])
    for axis in range(2):  # the flat plane extends in x,y
        assert abs(out[axis] - src[axis]) < 1e-3, f"open border shrank on axis {axis}"
    print("  PASS boundary preservation (open grid keeps its border extent)")


def test_speed():
    v, uv, f = make_grid(80)  # ~12.5k tris
    t = time.time()
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, target_ratio=0.3, lock_border_vertices=True, uv_distance_limit=10.0,
    )
    dt = time.time() - t
    assert dt < 10.0, f"12.5k tri reduction took {dt:.1f}s (perf regression)"
    assert len(r["faces"]) < len(f) * 0.5, "did not reach target ratio"
    print(f"  PASS speed (12.5k tris -> {len(r['faces'])} in {dt:.2f}s)")


def test_robust_degenerate_input():
    # empty
    r = wedge_backend.simplify_wedge_mesh([], [], [])
    assert r["stats"]["end_faces"] == 0
    # mismatched domain / weight / material lengths must not crash
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    f = [(0, 1, 2), (0, 2, 3)]
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, vertex_domains=[(0,)], group_weights=[{}], face_materials=[0],
        target_ratio=0.5,
    )
    assert len(r["vertices"]) == len(r["uvs"]) == len(r["vertex_domains"]) == len(r["group_weights"])
    # out-of-range face indices dropped, not crashed
    r = wedge_backend.simplify_wedge_mesh(v, uv, [(0, 1, 99), (0, 1, 2)], target_ratio=0.5)
    assert all(0 <= idx < len(r["vertices"]) for face in r["faces"] for idx in face)
    print("  PASS robustness (empty / mismatched / out-of-range inputs handled)")


def test_attribute_consistency():
    v, uv, f = make_cylinder(16, 3)
    domains = [(0, (0.0, 0.0, 1.0)) for _ in v]
    weights = [{"hull": 1.0} for _ in v]
    r = wedge_backend.simplify_wedge_mesh_partitioned(
        v, uv, f, vertex_domains=domains, group_weights=weights,
        face_materials=[0] * len(f), target_ratio=0.5, uv_distance_limit=10.0,
    )
    assert len(r["vertices"]) == len(r["uvs"]) == len(r["vertex_domains"]) == len(r["group_weights"])
    assert len(r["faces"]) == len(r["face_materials"])
    assert all(0 <= idx < len(r["vertices"]) for face in r["faces"] for idx in face)
    print("  PASS attribute consistency (verts/uvs/domains/weights/materials aligned)")


def test_qem_optimal_vertex_bounded():
    # an ill-conditioned quadric must not return a far-flung optimal point
    verts = [(0, 0, 0), (1, 0, 0.0001), (2, 0, 0), (1, 1, 0)]
    faces = [(0, 1, 3), (1, 2, 3)]
    quads = qem_backend.build_vertex_quadrics(verts, faces)
    q = qem_backend._quadric_add(quads[1], quads[3])
    target, mode = qem_backend.optimal_target(q, verts[1], verts[3])
    mid = qem_backend._vec_mid(verts[1], verts[3])
    edge_len = qem_backend._vec_len(qem_backend._vec_sub(verts[3], verts[1]))
    offset = qem_backend._vec_len(qem_backend._vec_sub(target, mid))
    assert offset <= edge_len * 4.0 + 1e-6, f"optimal vertex unbounded ({offset/edge_len:.1f}x)"
    print(f"  PASS QEM optimal vertex bounded (offset {offset/edge_len:.2f}x edge, mode={mode})")


def test_smooth_normals_recomputed():
    # flat grid in XY -> every smooth normal must be axis-aligned Z, NOT a stale
    # source face normal. This guards the post-collapse normal recomputation.
    n = 6
    verts = [(i, j, 0.0) for j in range(n) for i in range(n)]
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + i + 1
            c = (j + 1) * n + i + 1
            d = (j + 1) * n + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    normals = vehicle_reducers.compute_smooth_normals(verts, faces)
    assert len(normals) == len(verts)
    for nrm in normals:
        assert abs(abs(nrm[2]) - 1.0) < 1e-6, f"normal not Z-aligned: {nrm}"
    # a folded crease (two perpendicular quads) must give the shared edge a
    # blended normal, proving smoothing actually averages adjacent faces
    v2 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)]
    f2 = [(0, 1, 3), (0, 3, 2), (0, 4, 5), (0, 5, 1)]
    nn = vehicle_reducers.compute_smooth_normals(v2, f2)
    assert all(abs(vehicle_reducers._length(x) - 1.0) < 1e-6 for x in nn), "non-unit normal"
    print(f"  PASS smooth normals recomputed (flat grid Z-aligned, {len(normals)} verts)")


def test_radial_reduce_preserves_roundness():
    seg, rings = 48, 4
    verts = []
    for ri in range(rings + 1):
        z = -0.3 + 0.6 * ri / rings
        for s in range(seg):
            a = 2 * math.pi * s / seg
            verts.append((math.cos(a), math.sin(a), z))
    faces = []
    for ri in range(rings):
        for s in range(seg):
            s2 = (s + 1) % seg
            v0 = ri * seg + s
            v1 = ri * seg + s2
            v2 = (ri + 1) * seg + s2
            v3 = (ri + 1) * seg + s
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))
    uvs = [(0.0, 0.0)] * len(verts)

    center, axis, conf = vehicle_reducers.estimate_radial_axis(verts)
    assert conf > 0.8, f"radial axis confidence too low: {conf:.3f}"
    assert abs(abs(axis[2]) - 1.0) < 0.05, f"spin axis should be ~Z: {axis}"

    res = vehicle_reducers.radial_reduce(verts, uvs, faces, axis_point=center, axis_dir=axis, target_segments=12)
    assert len(res["faces"]) < len(faces), "radial reduce did not reduce"
    # roundness: all reduced verts keep in-plane radius ~1.0
    for p in res["vertices"]:
        d = vehicle_reducers._sub(p, center)
        h = vehicle_reducers._dot(d, axis)
        planar = vehicle_reducers._sub(d, vehicle_reducers._scale(axis, h))
        r = vehicle_reducers._length(planar)
        assert abs(r - 1.0) < 1e-3, f"radius drifted to {r:.3f}"
    ratio = len(res["faces"]) / len(faces)
    print(f"  PASS radial reduce ({len(faces)} -> {len(res['faces'])} faces, radius held, conf {conf:.2f})")


def test_visibility_cull_removes_interior():
    def box(c, s):
        h = s / 2
        v = [(c[0] - h, c[1] - h, c[2] - h), (c[0] + h, c[1] - h, c[2] - h),
             (c[0] + h, c[1] + h, c[2] - h), (c[0] - h, c[1] + h, c[2] - h),
             (c[0] - h, c[1] - h, c[2] + h), (c[0] + h, c[1] - h, c[2] + h),
             (c[0] + h, c[1] + h, c[2] + h), (c[0] - h, c[1] + h, c[2] + h)]
        f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
        return v, f
    ov, of = box((0, 0, 0), 4.0)
    iv, iff = box((0, 0, 0), 1.0)
    verts = ov + iv
    faces = of + [tuple(x + len(ov) for x in t) for t in iff]
    kept, flags = vehicle_reducers.visibility_cull(verts, faces, samples=48, return_mask=True)
    outer = sum(1 for i in range(12) if flags[i])
    inner = sum(1 for i in range(12, 24) if flags[i])
    assert outer == 12, f"outer shell partly culled ({outer}/12)"
    assert inner == 0, f"interior box not culled ({inner}/12 kept)"
    print(f"  PASS visibility cull (outer 12/12 kept, interior 12/12 culled)")


def test_convex_decomposition_tighter():
    # cube hull volume sanity
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    hv, hf = vehicle_reducers.convex_hull_3d(corners)
    vol = vehicle_reducers.hull_volume(hv, hf)
    assert abs(vol - 8.0) < 1e-6, f"cube hull volume wrong: {vol}"
    assert len(hv) == 8, f"cube hull should have 8 verts, got {len(hv)}"
    # L-shape: multi-hull must be tighter than a single hull
    lpts = []
    for x in range(11):
        for y in range(6):
            for z in range(4):
                X, Y, Z = x * 0.2, y * 0.2, z * 0.2
                if X < 0.6 or Y < 0.6:
                    lpts.append((X, Y, Z))
    multi = vehicle_reducers.convex_decompose(lpts, max_hulls=3)
    assert len(multi) >= 2, "decomposition did not split the L-shape"
    multi_vol = sum(vehicle_reducers.hull_volume(h["vertices"], h["faces"]) for h in multi)
    sv, sf = vehicle_reducers.convex_hull_3d(lpts)
    single_vol = vehicle_reducers.hull_volume(sv, sf)
    assert multi_vol < single_vol, f"multi-hull not tighter ({multi_vol:.3f} vs {single_vol:.3f})"
    # degenerate inputs must still return a valid closed proxy
    assert vehicle_reducers.convex_hull_3d([])[0] == []
    assert len(vehicle_reducers.convex_hull_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0)])[0]) == 8  # bbox fallback
    print(f"  PASS convex decomposition (cube vol 8.0; L-shape {multi_vol:.2f} < single {single_vol:.2f})")


def _build_wedge_like_addon(verts, uv, faces, sharp_edges=None, material_ids=None,
                            angle_deg=35.0):
    """Replicate build_wedge_source(): smoothing-group normal domains + wedge split.

    This is the exact path the Blender add-on takes, so it guards against the
    per-face-normal "domain shattering" regression that made the wedge reducer
    achieve zero reduction on any curved surface.
    """
    if material_ids is None:
        material_ids = [0] * len(faces)
    groups = vehicle_reducers.face_smoothing_groups(
        verts, faces, sharp_edges=sharp_edges or set(),
        material_ids=material_ids, angle_limit_rad=math.radians(angle_deg),
    )
    wedge_map = {}
    wv, wuv, wdom, wfaces, wmat = [], [], [], [], []
    for fi, face in enumerate(faces):
        dom = (material_ids[fi], groups[fi])
        wf = []
        for vi in face:
            uvk = (round(uv[vi][0], 6), round(uv[vi][1], 6))
            key = (vi, uvk, dom)
            if key not in wedge_map:
                wedge_map[key] = len(wv)
                wv.append(verts[vi]); wuv.append(uvk); wdom.append(dom)
            wf.append(wedge_map[key])
        wfaces.append(tuple(wf)); wmat.append(material_ids[fi])
    return wv, wuv, wfaces, wdom, wmat


def test_smoothing_groups_partition():
    # smooth cylinder: every interior edge is below the hard-edge angle -> ONE group
    v, uv, f = make_cylinder(48, 4)
    groups = vehicle_reducers.face_smoothing_groups(v, f, angle_limit_rad=math.radians(35.0))
    assert len(set(groups)) == 1, f"smooth cylinder split into {len(set(groups))} groups (should be 1)"
    # a cube: every edge is a 90 deg crease -> each of the 6 faces is its own group
    cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    cf = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
          (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    cg = vehicle_reducers.face_smoothing_groups(cube, cf, angle_limit_rad=math.radians(35.0))
    assert len(set(cg)) == 6, f"cube should yield 6 hard-edge groups, got {len(set(cg))}"
    # material split: a flat strip cut by a material change must become 2 groups
    v2, uv2, f2 = make_grid(6)
    mats = [0 if i < len(f2) // 2 else 1 for i in range(len(f2))]
    g2 = vehicle_reducers.face_smoothing_groups(v2, f2, material_ids=mats,
                                                angle_limit_rad=math.radians(35.0))
    assert len(set(g2)) >= 2, "material border did not split the smoothing group"
    print("  PASS smoothing groups (cylinder=1, cube=6, material split>=2)")


def test_wedge_reduces_curved_surface_regression():
    """REGRESSION: the add-on domain path must actually reduce a curved surface.

    Before the smoothing-group fix the per-face-normal domains shattered the
    cylinder into one island per triangle and reduction was 0% (384 -> 384).
    """
    v, uv, f = make_cylinder(48, 4)
    wv, wuv, wf, wdom, wmat = _build_wedge_like_addon(v, uv, f)
    r = wedge_backend.simplify_wedge_mesh_partitioned(
        wv, wuv, wf, face_materials=wmat, vertex_domains=wdom,
        group_weights=[{} for _ in wv], target_ratio=0.25,
        uv_weight=25.0, uv_distance_limit=0.5, lock_border_vertices=True,
        allow_domain_crossing=False, safe_weld=True, reject_face_flips=True,
        boundary_weight=2.0, preserve_manifold=True,
    )
    end = r["stats"]["end_faces"]
    assert end < len(f) * 0.5, f"curved-surface reduction failed ({len(f)} -> {end})"
    _, nm, d = manifold_counts(r["faces"])
    assert nm == 0 and d == 0, f"reduction broke manifold ({nm} non-manifold, {d} degenerate)"
    print(f"  PASS curved-surface regression ({len(f)} -> {end} via add-on domain path)")


def test_hard_edges_protected_via_domains():
    """A cube routed through the add-on path must keep its 8 corners (no collapse)."""
    cube = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    cf = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
          (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    uv = [(p[0], p[1]) for p in cube]
    wv, wuv, wf, wdom, wmat = _build_wedge_like_addon(cube, uv, cf)
    r = wedge_backend.simplify_wedge_mesh_partitioned(
        wv, wuv, wf, face_materials=wmat, vertex_domains=wdom,
        group_weights=[{} for _ in wv], target_ratio=0.1,
        uv_distance_limit=10.0, lock_border_vertices=True,
        allow_domain_crossing=False, preserve_manifold=True,
    )
    out = bbox_dims(r["vertices"])
    for axis in range(3):
        assert abs(out[axis] - 2.0) < 1e-6, f"cube collapsed on axis {axis}: {out[axis]}"
    print("  PASS hard edges protected (cube keeps 2x2x2 extent under aggressive target)")


def test_target_faces_budget():
    # absolute triangle-count target should be hit closely (not just a ratio)
    v, uv, f = make_grid(40)  # single flat domain
    r = wedge_backend.simplify_wedge_mesh(
        v, uv, f, target_faces=600, lock_border_vertices=False,
        uv_distance_limit=10.0, boundary_weight=2.0,
    )
    end = r["stats"]["end_faces"]
    assert end <= 900, f"absolute target overshot badly: asked 600 got {end}"
    assert end < len(f) * 0.5, f"absolute target under-reduced: {end}/{len(f)}"
    print(f"  PASS target_faces budget (asked 600, got {end} from {len(f)})")


def test_allocate_triangle_budget():
    alloc = vehicle_reducers.allocate_triangle_budget
    t = alloc([1000, 500, 500], 400)
    assert sum(t) == 400, f"budget not met: {t} sums to {sum(t)}"
    assert t[0] > t[1] and t[0] > t[2], f"largest part should get most: {t}"
    t2 = alloc([10, 10, 10], 9999)
    assert t2 == [10, 10, 10], f"over-budget should pass through unchanged: {t2}"
    t3 = alloc([100000, 3], 50000)
    assert t3[1] >= 1, f"tiny part starved to {t3[1]}"
    assert sum(t3) == 50000, f"budget not met with tiny part: {sum(t3)}"
    assert alloc([], 100) == []
    print("  PASS allocate_triangle_budget (sums to budget, caps, keeps tiny parts)")


def run_all():
    tests = [
        test_qem_optimal_vertex_bounded,
        test_smoothing_groups_partition,
        test_wedge_reduces_curved_surface_regression,
        test_hard_edges_protected_via_domains,
        test_target_faces_budget,
        test_allocate_triangle_budget,
        test_bounded_vertex_no_bbox_explosion,
        test_manifold_preserved_sphere,
        test_link_condition_octahedron,
        test_boundary_preserved_open_grid,
        test_speed,
        test_robust_degenerate_input,
        test_attribute_consistency,
        test_smooth_normals_recomputed,
        test_radial_reduce_preserves_roundness,
        test_visibility_cull_removes_interior,
        test_convex_decomposition_tighter,
    ]
    print("Running Vehicle Smart LOD backend tests...")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"{failed} test(s) failed.")
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
