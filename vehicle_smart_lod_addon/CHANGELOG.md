# Changelog

## 0.6.0 - Domain correctness + LOD budgeting

The headline fix in this release makes the wedge reducer actually work on curved
geometry. See `GAPS.md` for the full AAA readiness analysis.

### Critical fix - wedge normal domains were per-face (zero reduction on curves)

`build_wedge_source()` keyed each wedge vertex's normal domain on the raw
per-face normal. Because the reducer rejects collapses across differing domains,
every triangle on a curved surface became its own one-triangle domain island and
**no collapse was possible** - the smart reducer did nothing on vehicle bodywork.
Measured on a smooth 384-triangle cylinder via the real add-on path: 384 -> 384
before, 384 -> 102 (target 96) after.

Normal domains are now **smoothing groups** computed by
`vehicle_reducers.face_smoothing_groups()`: a flood fill that only breaks across
sharp-flagged edges, material borders, and dihedral angles above the hard-edge
threshold. Smooth regions stay connected and collapse; genuine hard edges stay
protected. UV seams remain protected by the UV-distance limit, not the normal
domain. The hard-edge angle from the panel feeds this grouping.

### Added - absolute triangle-budget targeting

- `simplify_wedge_mesh` / `simplify_wedge_mesh_partitioned` accept `target_faces`
  to hit an exact triangle count instead of a ratio.
- `vehicle_reducers.allocate_triangle_budget()` splits a whole-mesh budget across
  islands proportionally, capping at each part's size and keeping tiny parts >= 1.
- New **Target Triangles (0 = ratio)** setting drives the wedge preview and safe
  test pack.

### Fixed - iteration ceiling silently capped large parts

Each collapse removes ~2 faces, so the default `max_iterations` (5000) capped
reduction at ~10k faces regardless of target. The wedge operators now raise the
ceiling to at least the part's face count so the target is reachable.

### Tests

- `TESTS.py` extended from 12 to 18 tests: smoothing-group partitioning, a
  curved-surface reduction regression run through the exact add-on domain path,
  hard-edge protection via domains, absolute `target_faces`, and
  `allocate_triangle_budget`. All pass standalone (no Blender).

## 0.4.0 - Correctness and performance pass

This release is a bug-fix and hardening pass over the 0.3.0 prototype. The focus
was the reducer core, which had two defects that made it unsafe and unusable on
real vehicle meshes, plus several smaller robustness issues.

### Critical fixes

**Bounded optimal-vertex placement (mesh corruption).**
The QEM "solve the 3x3 system" step returned a vertex position with no bound on
how far it could move. On flat or cylindrical regions the quadric is
ill-conditioned, and the solver placed collapsed vertices many edge-lengths away
from the edge. A 24-segment cylinder reduced to 20% saw its Z dimension grow from
2.0 to 4.18 - a doubled bounding box and visible spikes. The face-flip test did
not catch it because the adjacent faces stretched rather than inverted.
`optimal_target` now rejects any solved vertex outside a bounded neighbourhood of
the edge and falls back to the best of {endpoint a, endpoint b, midpoint}.

**Incremental priority-queue reducer (was quadratic).**
The old loop rebuilt the entire edge list, all vertex quadrics, and all collapse
candidates on every single collapse. A 1,682-triangle grid took ~15 seconds;
extrapolated to the tool's own 30,000-triangle default that is roughly 80
minutes. The reducer is now an incremental heap collapse: quadrics are built
once, edges are scored once, and only the edges touched by a collapse are
re-scored. The same 1,682-triangle grid now reduces in ~0.13 s, and 12,500
triangles reduce in ~1.6 s.

### High-severity fixes

**Manifold preservation (link condition).**
Edge collapses could create non-manifold fans when the two endpoints shared more
neighbours than the edge's own triangles. Non-manifold output is rejected by most
engine importers. A standard topological link-condition test now runs before each
collapse and rejects any that would break the manifold. Controlled by the new
**Preserve Manifold** toggle (on by default).

**Boundary preservation without hard-locking.**
Previously open borders (panel edges, cylinder caps, holes) could only be
protected by fully locking their vertices, which stopped reduction entirely on
heavily seamed parts. The reducer now adds a perpendicular "virtual plane"
quadric along open boundary edges, so borders are held in place by cost while
still allowing interior reduction. Controlled by the new **Boundary Weight**
setting.

**Stall-on-first-exhaustion removed.**
The old loop did `if not best: break`, abandoning the whole mesh the first time a
single iteration found no valid collapse. The heap loop naturally moves on to the
next-cheapest edge instead.

### Robustness fixes

- Input normalisation: mismatched `vertex_domains`, `group_weights`, or
  `face_materials` lengths no longer raise `IndexError`; they are padded or
  truncated. Faces referencing out-of-range vertices are dropped. Non-triangle
  faces are triangulated. Empty meshes return cleanly.
- Custom split normals now also enable `use_auto_smooth` on Blender 4.0 (the flag
  was removed in 4.1+, so this is guarded with `hasattr`), so preserved normals
  actually display.
- `modifier_apply` is now wrapped in a context-safe helper that forces object
  mode and isolates the selection before applying, instead of relying on
  ambient selection/active state inside loops.

### Tests

- New `TESTS.py` standalone suite (no Blender required) covering bounded vertex
  placement, manifold preservation, boundary preservation, speed, robustness to
  degenerate input, and attribute consistency.
- `SMOKE_TEST.py` and the in-Blender self-test operator continue to pass.

### Notes / still open

- Visibility culling is exact-ish but O(faces) per view ray; it is guarded by a
  triangle-count limit and is best run on interior/chassis parts.
- Convex decomposition uses deterministic spatial clustering plus a hull per
  cluster (VHACD-style), not a full volumetric VHACD; it is a tight, cheap
  collision proxy rather than a perfect concavity match.

---

## 0.5.0 - Vehicle-specific reducers

This release implements the four roadmap items from METHOD.md Stage 5 and the
post-collapse normal recomputation noted as open in 0.4.0. All four are pure
Python in `vehicle_reducers.py` and covered by `TESTS.py`.

### Smooth normal recomputation (fixes the stale-normal note from 0.4.0)

The wedge preview previously wrote the original per-domain source face normal as
the custom split normal. After collapses that normal is stale. It now recomputes
area-weighted smooth vertex normals from the reduced geometry. Because wedges are
already split across material/normal domains, per-vertex smoothing does not bleed
across hard-normal seams. Verified: a flat grid yields exactly axis-aligned
normals; a crease keeps unit normals on the shared edge.

### Radial tyre reducer (`radial_reduce` + `estimate_radial_axis`)

Decimates a surface of revolution in the angular direction instead of collapsing
arbitrary edges. The spin axis is estimated via covariance principal axes with a
confidence score (plane balance, axial-spread ratio, radius consistency); parts
below a confidence threshold are skipped rather than damaged. Each vertex angle
is snapped to one of N segments while its radius and axial height are preserved,
then rings are welded. Verified: a 48-segment cylinder reduces to 12 segments
(384 -> 96 faces) with the in-plane radius held at 1.0.

### Visibility / interior-geometry culler (`visibility_cull`)

Removes faces unreachable from any exterior viewpoint (cockpit guts, hidden
chassis, doubled interior walls). Viewpoints are placed on a Fibonacci sphere; a
uniform spatial grid plus Moller-Trumbore ray tests provide broad-phase
occlusion. Guarded by a triangle-count limit. Verified: a box fully enclosing a
smaller box keeps all 12 outer faces and culls all 12 interior faces.

### Convex collision decomposition (`convex_hull_3d` + `convex_decompose`)

Three collision proxy modes: bounding box, single convex hull (incremental 3D
hull with a bounding-box fallback for degenerate/coplanar input), and clustered
multi-hull convex decomposition (deterministic k-means then a hull per cluster).
Verified: a cube point cloud yields a volume-8 hull; an L-shape multi-hull is
~40% tighter than its single hull.

### New operators / settings

- Operators: `vlod.radial_tyre`, `vlod.visibility_cull`, `vlod.collision_hull`.
- Settings: radial segments + confidence threshold, cull samples/margin/max
  faces, collision method (BOX/HULL/DECOMPOSE) + max hulls.
- New "Vehicle Reducers" panel section.

### Tests

- `TESTS.py` extended from 8 to 12 tests; the four new ones cover smooth normals,
  radial roundness preservation, interior culling, and convex decomposition
  tightness (plus hull degenerate-input fallbacks).
