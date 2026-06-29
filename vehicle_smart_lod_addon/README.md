# Vehicle Smart LOD

Prototype Blender add-on for vehicle-aware, constraint-aware LOD generation.

This is not meant to be a magic one-button remesher. The goal is to build a smarter planner around known reduction methods so a vehicle is treated like a vehicle: UV seams, hard edges, material borders, pivots, named parts, glass, wheels, turrets, and tiny detail all need different handling.

## Current prototype

- Analyses selected mesh objects.
- Counts triangles, vertices, UV seams, material borders, boundary edges, hard-normal edges, and protected vertices.
- Classifies parts using object names, materials, dimensions, and triangle counts.
- Writes a JSON report to a Blender text block named `Vehicle Smart LOD Report`.
- Creates a `VLOD_PROTECTED` vertex group for seam/material/boundary/hard-edge vertices.
- Builds a non-destructive QEM collapse plan showing the safest edge-collapse candidates.
- Supports UV-safe QEM planning by locking UV wedge vertices and adding UV distortion cost.
- Generates experimental wedge QEM preview meshes using split render vertices.
- Uses a fast incremental priority-queue collapser (re-scores only edges touched by each collapse).
- Bounds optimal-vertex placement so collapses cannot fling vertices outside the mesh.
- Rejects collapses that would create non-manifold topology (link condition).
- Holds open mesh borders in place with a boundary cost instead of hard-locking them.
- Supports partitioned mesh-island preview reduction.
- Rejects edge collapses that flip or flatten adjacent faces.
- Writes validation metrics into the report.
- Can create simple bounding-box collision proxy objects for quick testing.
- Recomputes smooth area-weighted normals on reduced wedge previews (domain-aware, no seam bleed).
- Radial tyre reducer: angular decimation that preserves the round cross-section, with spin-axis confidence gating.
- Visibility culler: removes interior faces never visible from any exterior viewpoint.
- Collision proxy generator: bounding box, single convex hull, or clustered multi-hull convex decomposition.
- Includes a backend self-test operator and standalone `SMOKE_TEST.py`.
- Adds `Set Safe Test Defaults` for conservative first-run settings.
- Adds `Build Safe Test Pack` for one-click analysis, protection marking, limited preview generation, collision proxies, and validation.
- Generates conservative duplicated LOD objects into a new `Vehicle Smart LOD Output` collection.
- Uses planar dissolve first for hard-surface objects.
- Uses Blender collapse decimate after planar cleanup for lower LODs.
- Skips glass by default.
- Can delete tiny detail objects in generated LODs when enabled.

## Install

1. Zip the `vehicle_smart_lod_addon` folder, or install the folder directly if your Blender setup supports it.
2. In Blender, go to `Edit > Preferences > Add-ons > Install`.
3. Enable `Vehicle Smart LOD`.
4. Open the viewport sidebar with `N`.
5. Use the `Vehicle LOD` tab.

## Suggested safe workflow

1. Save a copy of the `.blend`.
2. Select the vehicle mesh objects.
3. Click `Set Safe Test Defaults`.
4. Click `Run Backend Self-Test`.
5. Click `Build Safe Test Pack`.
6. Read the report and inspect the generated preview objects.
7. Click `Analyse Selected Vehicle` if you want a broader report.
8. Click `Build QEM Collapse Plan` on one object if you want candidate detail.
9. Click `Generate Wedge QEM Preview` manually on one or two problem parts if needed.
10. Compare the preview mesh against the original UVs/materials/silhouette.
11. Generate only `LOD1` first.
12. Inspect UVs, hard edges, wheels, windows, doors, turret parts, and silhouette.
13. Only then try `LOD2` and `LOD3`.

## Part treatment rules

| Part type | Current treatment | Better future treatment |
|---|---|---|
| General body | planar dissolve then collapse decimate | attribute-aware QEM |
| Flat/trim panels | planar dissolve | constrained planar region solver |
| Wheels/tyres | radial segment reducer (implemented) | tread-aware UV preservation |
| Glass | skipped by default | planar-only validation |
| Turret/weapon | protected planar/collapse | pivot and socket aware QEM |
| Doors/hatches | protected planar | animation-boundary aware reduction |
| Interior | visibility cull then reduce (implemented) | cockpit visibility mask refinement |
| Collision | box / convex hull / multi-hull (implemented) | volumetric VHACD-grade decomposition |
| Bolts/rivets/tiny detail | optional delete | bake/instance classifier |

## Where the method is heading

The target method is a vehicle-specific simplification planner:

```text
analyse mesh
  -> build constraint net
  -> classify each object/island
  -> pick reducer per class
  -> validate UV/normal/silhouette error
  -> output LODs and report
```

The constraint net is the important idea. For terrain papers, the protected topology is built from critical scalar-field lines. For vehicles, the protected topology is:

- UV seams and UV island borders
- Material borders
- Hard/sharp normal edges
- Open mesh boundaries
- Object origins and animation-related parts
- Vertex groups / named selections
- Sockets, memory points, attachment points
- Silhouette edges from common camera angles

## Maths backlog

These are the next reducers worth implementing behind the same Blender UI:

- Quadric Error Metrics (QEM) for geometric simplification. The first non-destructive QEM planning pass is now implemented.
- Attribute-aware QEM for UVs, normals, colour, material ids, and tangent-space preservation.
- Curvature-weighted QEM to keep wheel arches, tyres, rounded panels, and silhouette features.
- Seam-aware edge collapse that rejects UV island crossing unless explicitly allowed.
- Saliency weighting based on curvature contrast and likely viewing angles.
- Visibility and ambient-occlusion sampling to remove internal/hidden geometry.
- Radial simplification for tyres and cylindrical parts.
- Convex hull / VHACD-style proxy generation for collision.
- Progressive mesh records so LOD1/LOD2/LOD3 are coherent rather than independently damaged.
- Distributed/partitioned QEM so large vehicle meshes can be reduced per part or chunk, then reconciled safely.

## AAA gate

The add-on should not claim production-grade automated reduction until it can pass these gates:

| Gate | Required result |
|---|---|
| QEM planning | collapse candidates are sane on real vehicle meshes |
| QEM mutation | custom reducer can collapse edges without invalid geometry |
| Attribute preservation | UVs, normals, material ids, and tangents survive within tolerance |
| Vehicle semantics | wheels, glass, turrets, doors, suspension, sockets, and selections are protected |
| Validation | report proves triangle reduction, UV error, normal error, and silhouette error |
| Engine profile | output matches Enfusion/Reforger limits and naming expectations |

## UV safety rules

For LOD0 and LOD1, the tool should assume UVs are sacred.

Current QEM planning rules:

- UV island borders are protected.
- Explicit Blender seam edges are protected.
- Vertices with multiple loop UV values are treated as UV wedge vertices and locked.
- Collapse candidates can be rejected if endpoint UVs are too far apart.
- Collapse candidates include a UV penalty in the total cost.
- Missing UV data is treated as unsafe in UV-safe mode.

This is intentionally conservative. Full UV-preserving mutation needs wedge-vertex reconstruction, where the reducer operates on split render vertices rather than only Blender mesh vertices.

The first wedge QEM preview path now exists. It:

- splits vertices by source vertex, UV, material, and normal domain
- runs an incremental priority-queue QEM collapse on that split representation
- bounds optimal-vertex placement so collapsed vertices stay near their edge
- rejects collapses that would create non-manifold topology
- holds open mesh borders in place with a boundary-plane cost
- rejects collapses across different material/normal domains
- rejects candidates past the configured UV distance limit
- carries face material IDs through removed degenerate triangles
- carries vertex group weights through collapses using max-weight merging
- safely rejoins only identical position/UV/domain duplicates after reduction
- reduces disconnected mesh islands separately when partitioning is enabled
- rejects collapses that flip or flatten adjacent faces
- reports triangle ratio, vertex ratio, bounding-box delta, surface-area ratio, UV-area ratio, and material preservation
- writes a new preview mesh with UVs and copied material slots

Current behaviour: the reducer is an incremental heap collapse. Quadrics are
built once, edges are scored once, and only the edges touched by a collapse are
re-scored, so it runs in roughly O(E log E) instead of the original quadratic
loop. A ~12,500-triangle part reduces in around 1-2 seconds. It is still
recommended to test problem parts individually before running a whole vehicle.

Performance guard:

- `Max Preview Tris` defaults to 30000.
- `Allow Heavy Preview` is off by default.
- `Safe Test Pack` processes only the heaviest few selected objects that pass the triangle guard.
- `Max QEM Iterations` stops bad topology from running forever.

The safe rejoin rule is strict:

```text
can weld only if:
  rounded position matches
  rounded UV matches
  material/normal domain matches
```

It will leave extra vertices rather than weld across UV seams, material borders, or hard-normal domains.

## First serious test asset

Use a duplicated vehicle file and record:

| Check | Pass condition |
|---|---|
| UVs | no obvious stretching or island border collapse |
| Materials | no material bleeding across panels |
| Normals | hard-surface panels still read cleanly |
| Silhouette | outline remains close at gameplay camera distances |
| Animation | doors/wheels/turrets still pivot correctly |
| Triangle count | LOD0/LOD1 targets are hit without destroying close-view quality |
| Engine import | no invalid vertex/material/selection issues |
