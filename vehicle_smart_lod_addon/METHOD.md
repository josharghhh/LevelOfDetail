# Vehicle-Aware Constraint LOD Method

## Problem

Generic decimation fails on game vehicles because it optimises triangle count without understanding the asset contract.

A vehicle mesh is not just surface geometry. It also carries:

- UV layout
- material assignment
- hard-surface normals
- animation and damage selections
- object origins and pivots
- sockets and attachment points
- shadow/collision requirements
- engine vertex and triangle limits

For LOD0 and LOD1, losing those can be worse than leaving the mesh heavy.

## Core idea

Build a protected constraint net first, then simplify only where the net allows it.

```text
input vehicle
  -> object/island analysis
  -> constraint net
  -> part classifier
  -> per-part reducer selection
  -> error validation
  -> LOD outputs
```

## Constraint net

The constraint net is the set of vertices, edges, and regions that should not be freely collapsed.

Hard constraints:

- UV island borders
- material borders
- mesh boundaries
- named animation selections
- object origins/pivots
- attachment/socket/memory points

Soft constraints:

- hard normal edges
- high-curvature edges
- visible silhouette edges
- high-saliency detail
- player-visible interior regions

Hard constraints block collapse. Soft constraints increase collapse cost.

## Collapse cost

Future reducer core:

```text
cost(edge collapse) =
  w_geo * geometric_qem_error
+ w_uv * uv_distortion
+ w_nrm * normal_deviation
+ w_mat * material_boundary_violation
+ w_hard * hard_edge_deviation
+ w_sil * silhouette_error
+ w_sal * saliency_loss
+ w_anim * animation_selection_violation
+ w_vis * visibility_importance
```

The collapse is rejected if it:

- flips face winding
- creates degenerate triangles
- crosses a locked UV/material/animation boundary
- moves a locked boundary vertex
- exceeds local UV stretch tolerance
- breaks object part identity

## Reducer choices

No single reducer is right for every vehicle part.

| Region | Reducer |
|---|---|
| Flat armour panels | planar constrained dissolve |
| Curved panels | attribute-aware QEM |
| Tyres | radial ring/segment reducer |
| Rims | feature-aware QEM with hub lock |
| Glass | planar-only or skip |
| Tiny bolts/rivets | delete, instance, or bake |
| Grills/vents | alpha/normal replacement where possible |
| Underside | aggressive visibility-weighted QEM |
| Interior | cockpit visibility mask first |
| Collision | convex decomposition / hull fitting |
| Shadow | silhouette-preserving proxy |

## UV preservation

UV preservation is not just keeping the UV layer.

The reducer must preserve:

- UV island borders
- local UV-to-world area ratio
- tangent-space continuity
- material atlas boundaries

For LOD0 and LOD1:

```text
collapse inside a UV island: allowed if stretch stays low
collapse along a UV seam: allowed only if both sides remain separate
collapse across a UV seam: blocked by default
```

For far LODs:

```text
re-atlas and bake may be better than preserving original UVs
```

Current UV-safe QEM planning:

- UV seam and UV island edges are protected.
- A Blender vertex with multiple loop UV values is treated as a wedge vertex and locked.
- Candidate collapses can be rejected by UV endpoint distance.
- Candidate costs include a UV penalty.
- Missing UV data is unsafe in UV-safe mode.

Next required implementation detail:

```text
Blender mesh vertices are not enough for destructive UV-safe simplification.
The reducer needs a render/wedge vertex layer:
  key = position vertex + uv + normal/tangent domain + material domain
```

The reducer can collapse wedge vertices inside the same UV chart, then reconstruct Blender mesh loops and UVs from that wedge representation.

## Validation metrics

Every generated LOD should produce a report:

- triangle and vertex count change
- protected vertex ratio
- UV seam count before/after
- material boundary count before/after
- normal deviation score
- silhouette deviation from sampled cameras
- surface area deviation
- bounding box deviation
- risky part list

## Prototype stages

### Stage 1 - Blender-safe planner

- Analyse selected objects.
- Mark protected vertices.
- Use Blender planar dissolve and collapse decimate conservatively.
- Produce report.

This is implemented in the current prototype.

### Stage 2 - Better object classification

- Mesh island detection.
- Part classifier based on name, material, dimensions, symmetry, and curvature.
- Detect repeated bolt/rivet objects.
- Detect wheels by circular/radial geometry.

### Stage 3 - Custom constrained reducer

- Implement edge-collapse queue.
- Start with geometric QEM.
- Add hard constraints.
- Add UV and normal penalties.
- Add collapse rejection tests.

Current status:

- geometric QEM calculation exists
- optimal/fallback collapse target calculation exists, now bounded so it cannot place vertices far outside the mesh
- protected edge and locked vertex rejection exists
- non-destructive candidate reporting exists
- experimental wedge QEM preview mesh generation exists
- reduction now uses an incremental priority queue (re-scores only touched edges) instead of a quadratic rebuild loop
- non-manifold collapses are rejected via a link-condition test
- open boundaries are held with virtual-plane quadrics instead of only hard locks
- material/normal domain crossing is rejected in wedge preview reduction
- UV distance limits are enforced in wedge preview reduction
- safe rejoin welds only identical position/UV/domain duplicates
- vertex group weights are transferred to preview meshes
- disconnected mesh islands can be partitioned and reduced separately
- face-flip and face-flattening collapse rejection exists
- preview reports include basic UV/material/bounds/surface-area validation
- simple bounding-box collision proxy generation exists
- backend self-test operator and standalone smoke test exist
- safe test defaults and one-click safe test pack exist
- preview generation has triangle-count and max-iteration guards
- validation gate emits warnings for material loss, surface-area drift, UV-area drift, bounding-box drift, and no-op reductions
- source meshes are not modified

### Stage 4 - Distributed QEM planner

Large game vehicles are already naturally partitioned into doors, tyres, hull panels, weapons, suspension parts, interior, glass, and tiny detail objects. A distributed QEM design should exploit that instead of forcing one global simplification pass.

Partition strategy:

- object boundaries first
- mesh islands second
- material/UV islands as optional sub-partitions
- spatial chunks only for oversized single meshes

Each partition gets a local QEM job. Boundary vertices are locked or treated as high-priority soft constraints depending on the LOD target.

```text
vehicle
  -> partitions
  -> local QEM jobs
  -> boundary reconciliation
  -> global validation
  -> LOD output
```

Boundary reconciliation rules:

- locked UV seams must still match original island borders
- neighbouring chunks must not open cracks
- material boundaries must remain assigned to valid faces
- object origins and animation selections must survive unchanged
- silhouette-critical boundary vertices can move only within tolerance

This gives us parallelism and safer failure modes. If one partition fails, the tool can leave that part untouched and continue with the rest of the vehicle.

### Stage 5 - Vehicle-specific reducers

Implemented in `vehicle_reducers.py` (pure Python, covered by TESTS.py):

- Radial tyre reducer - done. Spin-axis estimation with a confidence score, then
  angular snapping that preserves the cross-section profile. Low-confidence parts
  are skipped, not damaged.
- Internal geometry visibility culler - done. Fibonacci-sphere viewpoints, spatial
  grid + Moller-Trumbore occlusion, triangle-count guarded.
- Collision proxy generator - done. Bounding box, single convex hull, or clustered
  multi-hull convex decomposition.
- Post-collapse smooth normal recomputation - done. Replaces the previously stale
  per-domain source face normal in wedge previews.

Still open:

- Shadow mesh generator (silhouette-preserving proxy).
- A volumetric VHACD-grade decomposition (current decomposition is clustered
  hulls, which is tight and cheap but not a perfect concavity match).

### Stage 6 - Engine profiles

- Enfusion/Reforger profile.
- Sollumz/GTA profile.
- LFS-style shadow/collision profile.
- Unreal/Unity generic profile.
