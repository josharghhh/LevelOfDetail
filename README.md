# LevelOfDetail

Vehicle-aware, constraint-aware LOD generation add-on for Blender, aimed at
Arma Reforger / Enfusion vehicles (bringing ~400–500k-triangle source models
down to a sane LOD0 and a coherent LOD chain without destroying UVs, hard-surface
normals, materials, wheels, glass, turrets, or the silhouette).

## Repository layout

- `vehicle_smart_lod_addon/` — the Blender add-on (zip this folder to install).
  - `__init__.py` — Blender operators, panel, and mesh glue.
  - `qem_backend.py` — pure-Python QEM core (quadrics, bounded collapse target).
  - `wedge_backend.py` — incremental wedge (render-vertex) QEM reducer.
  - `vehicle_reducers.py` — smoothing groups, budget allocation, radial tyre
    reducer, visibility culler, convex collision decomposition, smooth normals.
  - `TESTS.py` / `SMOKE_TEST.py` — standalone tests (no Blender required).
  - `README.md` / `METHOD.md` / `CHANGELOG.md` — add-on docs.
  - **`GAPS.md` — AAA readiness gap analysis and roadmap. Start here.**

## Quick start (no Blender)

```
cd vehicle_smart_lod_addon
python3 TESTS.py        # 18 backend tests, including the domain regression
python3 SMOKE_TEST.py
```

## Install in Blender

Zip `vehicle_smart_lod_addon/`, then `Edit > Preferences > Add-ons > Install`,
enable **Vehicle Smart LOD**, and open the `Vehicle LOD` tab in the `N` sidebar.

See `vehicle_smart_lod_addon/README.md` for the suggested safe workflow and
`vehicle_smart_lod_addon/GAPS.md` for what is and isn't production-ready.
