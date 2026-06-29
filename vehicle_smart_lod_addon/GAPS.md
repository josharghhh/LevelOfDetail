# AAA Readiness — Gap Analysis (Arma Reforger / Enfusion vehicles)

Context: source vehicles are ~400–500k triangles and need to come down to a
sane LOD0 plus a coherent LOD chain, while keeping UVs, hard-surface normals,
materials, wheels/glass/turrets, sockets, and the silhouette intact.

This document is the result of a full read-through, a standalone test run, and
some scaling/benchmark probes of the 0.5.0 add-on. It records what was actually
broken, what I fixed in this pass, and what still stands between this tool and
"AAA-worthy" automated reduction. Severity is from the point of view of getting
a real Reforger vehicle through it.

---

## 0. What I fixed in this pass (0.6.0)

### FIXED — CRITICAL: wedge "normal domains" shattered every curved surface

`build_wedge_source()` keyed each wedge vertex's normal domain on the **raw
per-face normal** (`domain_key = (material, round(poly.normal))`). The reducer
refuses any collapse across differing domains (`allow_domain_crossing=False`).
On any curved surface, adjacent triangles have different face normals, so every
single triangle became its own domain island. Measured on a smooth 384-triangle
cylinder through the real add-on path:

```
before fix:  384 -> 384 triangles   (48 single-triangle partitions, 0% reduction)
after  fix:  384 -> 102 triangles   (1 smoothing group, target was 96)
```

The test suite never caught this because every test passed a single `"default"`
domain; the per-face-normal domain logic only existed in the Blender glue
(`build_wedge_source`), which the tests didn't exercise.

**Net effect of the bug:** the "smart" wedge QEM reducer — the headline feature
— did essentially nothing on vehicle bodywork. Only large flat coplanar regions
reduced. The custom reducer would have looked broken/useless on a real asset.

**Fix:** normal domains are now **smoothing groups** computed by
`vehicle_reducers.face_smoothing_groups()` — a flood fill that keeps faces in
the same domain unless they are separated by a sharp-flagged edge, a material
border, or a dihedral angle above the hard-edge threshold. Smooth regions stay
connected (so they collapse); genuine hard edges, sharp flags, and material
borders still hard-block collapses. UV seams remain protected by the existing
UV-distance limit/penalty, not by the normal domain. Covered by three new tests
plus a regression test that runs the exact add-on domain path.

### FIXED — `max_iterations` default silently capped reduction on big parts

Each collapse removes ~2 faces, and the default `max_iterations` is 5000, so the
reducer could remove at most ~10k faces no matter the target. A 100k-triangle
part would stop at ~90k. The wedge operators now auto-raise the ceiling to at
least the part's face count, so the requested target is actually reachable.

### ADDED — absolute triangle-budget targeting (AAA LOD budgeting)

LOD work is specified in triangle budgets ("LOD0 ≤ 90k, LOD1 ≤ 35k…"), not a
blanket ratio. The wedge reducers now accept `target_faces`, and
`allocate_triangle_budget()` splits a whole-mesh budget across islands by size.
Exposed in the UI as **Target Triangles (0 = ratio)**. Tested: a 3042-triangle
grid asked for 600 lands on exactly 600.

---

## 1. CRITICAL gaps still open

### 1.1 Performance ceiling: pure-Python reducer is ~5k tris/s

Measured on this machine through the real domain path:

| Source tris | Reduced to | Time | Throughput |
|---|---|---|---|
| 28k | 5.6k | 4.9 s | ~6k tris/s |
| 114k | 23k | 23 s | ~5k tris/s |
| 258k | 52k | 53 s | ~5k tris/s |

Scaling is roughly linear (the incremental heap works), but a 400–500k vehicle
is **minutes per heavy part** and the whole vehicle is many minutes to tens of
minutes. The current guards (`Max Preview Tris = 30000`, off unless "Allow Heavy
Preview") mean that **by default every heavy part of a real vehicle is skipped**.

What AAA needs, in order of effort:
- **Short term:** raise/clear the triangle guard and document expected runtimes;
  process per-part so the user can walk away.
- **Right answer:** offer a fast native path. Blender's own Decimate (Collapse)
  modifier is C-speed and already wired in `apply_decimate_modifier`, but it is
  *not* attribute/domain-aware. A hybrid — native collapse for the bulk, the
  constraint-aware wedge reducer only where the constraint net is dense — would
  get both speed and safety.
- **Or:** port the hot loop (quadric eval + heap) to `numpy`, or call an
  external meshoptimizer/`pymeshlab` simplifier with seam locking.

### 1.2 The actual LOD generator does NOT use the smart reducer

`VLOD_OT_generate` (the "Generate Conservative LODs" button that produces the
real LOD objects) uses only Blender's **Decimate** modifier (planar dissolve +
collapse). The validated, constraint-aware wedge reducer is used **only** in the
previews and the safe-test pack. So the tool's headline tech never reaches the
shipped LODs. Wire the wedge reducer (now that it works) into LOD generation as
a selectable method, with the absolute-budget path per LOD.

### 1.3 No Enfusion / Reforger engine profile or export (Stage 6, unimplemented)

This is the whole point of the tool and it's the largest missing piece for
*this* engine specifically:
- **LOD packaging:** Enfusion expects LODs authored in the model with explicit
  LOD levels + screen-coverage/distance thresholds. The tool emits loose
  `*.LOD1/.LOD2` objects into a collection with no LOD-level metadata, no
  distance thresholds, and no naming the Workbench importer recognises.
- **Sockets / points:** Reforger vehicles rely on named points and proxies
  (wheels `w_*`, lights, exhausts, damage zones, crew positions, hitpoints).
  None of these are detected, preserved, or carried to LODs. Collapsing near a
  socket without protecting it will desync wheels/lights from the chassis.
- **Geometry/Collision/Roadway/Hitpoint LODs:** Enfusion has special-purpose
  LODs beyond visual ones. The collision proxy generator is a good start but is
  not emitted in the layered form the engine consumes.
- **Material/shader naming + emissive/glass conventions** are not validated.
- **FBX/XOB export** with correct axis, scale, and LOD grouping is absent.

### 1.4 Tangent space is not preserved/validated

AAA vehicles are normal-mapped. The tool recomputes smooth split normals
(good), but tangents are left to re-generate at import. For mirrored UVs and
seams this can flip/shift tangents and break normal-map shading on LOD0/LOD1.
At minimum, validate tangent continuity at seams; ideally preserve handedness.

---

## 2. HIGH-severity gaps

### 2.1 Border locking + dense constraint nets stall reduction on hard-surface

`Lock Wedge Borders` (on in Safe Defaults) locks the perimeter of every
smoothing-group island. A hard-surface vehicle is hundreds of small islands, so
a lot of geometry is frozen and reduction underperforms its target. The
boundary-weight virtual planes already let you turn locking *off* and still hold
borders by cost — that path should become the default once validated on a real
asset, with locking reserved for UV/material seams only.

### 2.2 Radial tyre reducer scrambles tread UVs

`radial_reduce` welds rings by snapped position+domain and keeps the *first UV
seen*. Tyres with a tread normal/diffuse map will get garbled UVs around the
circumference. For AAA tyres the angular snap must also snap/interpolate UV in
the angular direction (or keep the reducer for the sidewall/structure and leave
tread bands alone).

### 2.3 No progressive / coherent LOD chain

Each LOD is reduced independently from the source. LOD2 is not a reduction of
LOD1, so vertices pop incoherently between levels and shared boundaries can
diverge. A progressive-mesh record (reduce once, snapshot at each budget) gives
coherent, pop-minimised LODs and is much faster than three independent passes.

### 2.4 Silhouette preservation is not measured or weighted

The validation gate checks surface area, UV area, bbox, and material set, but
nothing measures **silhouette error from gameplay camera angles** — the thing a
player actually notices on a distant vehicle. METHOD.md lists silhouette
weighting and a shadow-proxy generator; neither exists. QEM curvature/silhouette
weighting would also stop the reducer eating wheel arches and roof curvature
first.

### 2.5 Symmetry is ignored

Vehicles are largely mirrored. Independent reduction of mirrored halves breaks
symmetry (and visibly so on the centerline). A symmetry-aware pass (or reduce
one half + mirror) would improve quality and halve the work.

---

## 3. MEDIUM-severity gaps

- **Visibility culler cost:** exact-ish but O(faces) per ray and guarded at 8k
  faces — unusable on a full interior without pre-reduction. Needs a BVH, or run
  it only on isolated interior parts.
- **Convex decomposition is clustered-hull, not volumetric VHACD:** fine as a
  cheap proxy, but concave parts (roll cages, frames) won't be hugged tightly.
- **Tiny-detail handling is delete-only:** no bake-to-normal / instance path, so
  bolts/rivets/grilles are simply lost rather than represented at distance.
- **Part classifier is name/keyword-driven:** English tokens only; an unnamed or
  foreign-named mesh falls back to `general_body`. Geometry-based detection
  (radial → wheel, planar → panel) exists for wheels but not generally.
- **No automatic UV-stretch/normal-deviation metric in the report** beyond UV
  *area* ratio; a per-collapse stretch and an angular normal-deviation score
  would make the validation gate trustworthy enough to run unattended.
- **n-gon handling:** faces are fan-triangulated; concave n-gons can triangulate
  badly. Use a proper ear-clip/`tessellate` before wedge building.

## 4. LOW-severity / polish

- Operators run on the main thread with no progress bar; a 50s part looks like a
  hang. Add `wm.progress_*` or a modal operator.
- No undo grouping around multi-object generation.
- Reports are JSON in a text block; a small in-panel summary (before/after tris,
  worst offenders, gate pass/fail) would speed iteration.
- No unit on triangle counts vs Blender's "tris vs faces" (quads) in the UI.

---

## 5. Suggested order of attack (post-Blender validation)

1. Validate the domain fix on a real Reforger vehicle (this unblocks everything).
2. Decide the performance path (raise guards + document, vs. native/hybrid).
3. Wire the wedge reducer + absolute budgets into the real LOD generator (1.2).
4. Build the Enfusion profile: socket/point detection + protection, LOD-level
   metadata, layered collision/geometry output, FBX/XOB export (1.3).
5. Progressive LOD chain (2.3) + silhouette weighting & shadow proxy (2.4).
6. Tangent validation (1.4), tyre tread UVs (2.2), symmetry (2.5).

## 6. How to run the standalone tests (no Blender required)

```
cd vehicle_smart_lod_addon
python3 TESTS.py        # 18 backend tests incl. the domain regression
python3 SMOKE_TEST.py   # quick partitioned-reducer smoke test
```
