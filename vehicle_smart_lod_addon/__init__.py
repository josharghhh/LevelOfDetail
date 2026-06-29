bl_info = {
    "name": "Vehicle Smart LOD",
    "author": "Josh Farmbrough + GPT-5.5 Thinking",
    "version": (0, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Vehicle LOD",
    "description": "Vehicle-aware LOD analysis, fast UV-safe incremental QEM reduction, radial tyre/visibility/collision reducers, and validation tools.",
    "category": "Object",
}

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime

import bpy
from mathutils import Vector

try:
    from . import qem_backend
    from . import wedge_backend
    from . import vehicle_reducers
except Exception:
    import qem_backend
    import wedge_backend
    import vehicle_reducers


LOD_TEXT_NAME = "Vehicle Smart LOD Report"


def tri_count(mesh):
    return sum(max(0, len(poly.vertices) - 2) for poly in mesh.polygons)


def object_world_dims(obj):
    return tuple(round(v, 5) for v in obj.dimensions)


def edge_polygon_map(mesh):
    usage = defaultdict(list)
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        for i, a in enumerate(verts):
            b = verts[(i + 1) % len(verts)]
            usage[tuple(sorted((a, b)))].append(poly.index)
    return usage


def material_boundary_edges(mesh, edge_usage):
    boundaries = set()
    for key, poly_indices in edge_usage.items():
        mats = {mesh.polygons[i].material_index for i in poly_indices}
        if len(mats) > 1:
            boundaries.add(key)
    return boundaries


def normal_boundary_edges(mesh, edge_usage, angle_limit_rad):
    boundaries = set()
    for key, poly_indices in edge_usage.items():
        if len(poly_indices) != 2:
            continue
        n1 = mesh.polygons[poly_indices[0]].normal
        n2 = mesh.polygons[poly_indices[1]].normal
        if n1.angle(n2, 0.0) >= angle_limit_rad:
            boundaries.add(key)
    return boundaries


def uv_island_edges(mesh, edge_usage):
    if not mesh.uv_layers.active:
        return set()

    uv_layer = mesh.uv_layers.active.data
    by_edge = defaultdict(list)
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        loops = list(poly.loop_indices)
        for i, a in enumerate(verts):
            b = verts[(i + 1) % len(verts)]
            loop_a = loops[i]
            loop_b = loops[(i + 1) % len(loops)]
            key = tuple(sorted((a, b)))
            uv_pair = tuple(sorted((
                tuple(round(x, 6) for x in uv_layer[loop_a].uv),
                tuple(round(x, 6) for x in uv_layer[loop_b].uv),
            )))
            by_edge[key].append(uv_pair)

    seams = set()
    for key, pairs in by_edge.items():
        if len(edge_usage[key]) > 1 and len(set(pairs)) > 1:
            seams.add(key)
    return seams


def explicit_seam_edges(mesh):
    seams = set()
    for edge in mesh.edges:
        if edge.use_seam:
            seams.add(tuple(sorted(edge.vertices)))
    return seams


def sharp_flag_edges(mesh):
    sharp = set()
    for edge in mesh.edges:
        if getattr(edge, "use_edge_sharp", False):
            sharp.add(tuple(sorted(edge.vertices)))
    return sharp


def protection_sets(mesh, angle_limit_rad):
    edge_usage = edge_polygon_map(mesh)
    protected_edges = (
        {key for key, polys in edge_usage.items() if len(polys) == 1}
        | material_boundary_edges(mesh, edge_usage)
        | uv_island_edges(mesh, edge_usage)
        | explicit_seam_edges(mesh)
        | sharp_flag_edges(mesh)
        | normal_boundary_edges(mesh, edge_usage, angle_limit_rad)
    )
    protected_vertices = {v for edge in protected_edges for v in edge}
    return protected_vertices, protected_edges


def mesh_qem_data(mesh):
    vertices = [tuple(vertex.co) for vertex in mesh.vertices]
    faces = [tuple(poly.vertices) for poly in mesh.polygons if len(poly.vertices) >= 3]
    edges = [tuple(edge.vertices) for edge in mesh.edges]
    return vertices, faces, edges


def uv_safety_data(mesh):
    if not mesh.uv_layers.active:
        return {}, set(), {
            "has_uv": False,
            "single_uv_vertices": 0,
            "uv_wedge_locked_vertices": 0,
            "missing_uv_vertices": len(mesh.vertices),
        }

    uv_layer = mesh.uv_layers.active.data
    per_vertex = defaultdict(set)
    for poly in mesh.polygons:
        for vert_index, loop_index in zip(poly.vertices, poly.loop_indices):
            uv = uv_layer[loop_index].uv
            per_vertex[vert_index].add((round(float(uv.x), 6), round(float(uv.y), 6)))

    vertex_uvs = {}
    uv_locked = set()
    for vertex in mesh.vertices:
        values = per_vertex.get(vertex.index, set())
        if len(values) == 1:
            vertex_uvs[vertex.index] = next(iter(values))
        elif len(values) > 1:
            uv_locked.add(vertex.index)

    stats = {
        "has_uv": True,
        "single_uv_vertices": len(vertex_uvs),
        "uv_wedge_locked_vertices": len(uv_locked),
        "missing_uv_vertices": len(mesh.vertices) - len(vertex_uvs) - len(uv_locked),
    }
    return vertex_uvs, uv_locked, stats


def build_wedge_source(obj, angle_limit_rad=None):
    mesh = obj.data
    if angle_limit_rad is None:
        angle_limit_rad = math.radians(35.0)
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    wedge_map = {}
    vertices = []
    uvs = []
    vertex_domains = []
    group_weights = []
    faces = []
    face_materials = []

    # Normal domains are smoothing groups (regions bounded by sharp/hard edges
    # and material borders), NOT raw per-face normals. Keying on the per-face
    # normal shatters any curved surface into one domain per triangle and stops
    # the reducer collapsing anything on curved bodywork; smoothing groups keep
    # smooth regions connected while still protecting genuine hard edges.
    poly_vertex_lists = [tuple(p.vertices) for p in mesh.polygons]
    poly_material_ids = [int(p.material_index) for p in mesh.polygons]
    poly_normals = [tuple(float(v) for v in p.normal) for p in mesh.polygons]
    source_coords = [tuple(float(c) for c in v.co) for v in mesh.vertices]
    sharp_edges = {
        tuple(sorted(edge.vertices))
        for edge in mesh.edges
        if getattr(edge, "use_edge_sharp", False)
    }
    smoothing_groups = vehicle_reducers.face_smoothing_groups(
        source_coords,
        poly_vertex_lists,
        sharp_edges=sharp_edges,
        material_ids=poly_material_ids,
        angle_limit_rad=angle_limit_rad,
        face_normals=poly_normals,
    )

    for poly_index, poly in enumerate(mesh.polygons):
        group_id = smoothing_groups[poly_index]
        wedge_face = []
        for vert_index, loop_index in zip(poly.vertices, poly.loop_indices):
            if uv_layer:
                uv = uv_layer[loop_index].uv
                uv_key = (round(float(uv.x), 6), round(float(uv.y), 6))
            else:
                uv_key = (0.0, 0.0)
            material_key = int(poly.material_index)
            domain_key = (material_key, group_id)
            key = (int(vert_index), uv_key, domain_key)
            if key not in wedge_map:
                wedge_map[key] = len(vertices)
                vertices.append(tuple(float(v) for v in mesh.vertices[vert_index].co))
                uvs.append(uv_key)
                vertex_domains.append(domain_key)
                weights = {}
                for group in mesh.vertices[vert_index].groups:
                    if group.group < len(obj.vertex_groups):
                        weights[obj.vertex_groups[group.group].name] = float(group.weight)
                group_weights.append(weights)
            wedge_face.append(wedge_map[key])

        for tri in wedge_backend.triangulate_face(wedge_face):
            faces.append(tri)
            face_materials.append(int(poly.material_index))

    return {
        "vertices": vertices,
        "uvs": uvs,
        "vertex_domains": vertex_domains,
        "group_weights": group_weights,
        "faces": faces,
        "face_materials": face_materials,
        "source_vertices": len(mesh.vertices),
        "source_faces": len(mesh.polygons),
        "wedge_vertices": len(vertices),
    }


def create_wedge_preview_object(context, source_obj, result, collection, write_custom_normals=False,
                                mesh_suffix="wedge_qem_preview", obj_suffix="WEDGE_QEM_PREVIEW"):
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}.{mesh_suffix}")
    mesh.from_pydata(result["vertices"], [], result["faces"])
    mesh.update()

    for slot in source_obj.material_slots:
        mesh.materials.append(slot.material)

    for poly, mat_index in zip(mesh.polygons, result["face_materials"]):
        poly.material_index = min(mat_index, max(0, len(mesh.materials) - 1))

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index, vert_index in zip(poly.loop_indices, poly.vertices):
            uv_layer.data[loop_index].uv = result["uvs"][vert_index]

    if write_custom_normals and result.get("vertex_domains"):
        try:
            # Recompute smooth, area-weighted normals from the *reduced* geometry
            # instead of writing the stale source face normal. Because the wedge
            # pipeline already split vertices across material/normal domains, a
            # per-vertex smooth normal does not bleed across hard-normal seams.
            smooth_normals = vehicle_reducers.compute_smooth_normals(
                result["vertices"], result["faces"], result.get("vertex_domains")
            )
            loop_normals = []
            for poly in mesh.polygons:
                for vert_index in poly.vertices:
                    if vert_index < len(smooth_normals):
                        loop_normals.append(tuple(float(v) for v in smooth_normals[vert_index]))
                    else:
                        loop_normals.append((0.0, 0.0, 1.0))
            mesh.normals_split_custom_set(loop_normals)
            # Blender 4.0 needs auto-smooth enabled for custom split normals to
            # display; 4.1+ removed the flag and applies them directly.
            if hasattr(mesh, "use_auto_smooth"):
                mesh.use_auto_smooth = True
            for poly in mesh.polygons:
                poly.use_smooth = True
            mesh.update()
        except Exception:
            pass

    new_obj = bpy.data.objects.new(f"{source_obj.name}.{obj_suffix}", mesh)
    collection.objects.link(new_obj)
    new_obj.matrix_world = source_obj.matrix_world.copy()

    group_names = sorted({name for weights in result.get("group_weights", []) for name in weights})
    preview_groups = {name: new_obj.vertex_groups.new(name=name) for name in group_names}
    for vertex_index, weights in enumerate(result.get("group_weights", [])):
        for group_name, weight in weights.items():
            if weight > 0.0 and group_name in preview_groups:
                preview_groups[group_name].add([vertex_index], weight, "REPLACE")

    return new_obj



def apply_wedge_result_to_lod_object(obj, source_obj, result, write_custom_normals=False):
    """Replace a duplicated LOD object's mesh data with a wedge reducer result."""
    old_mesh = obj.data
    mesh = bpy.data.meshes.new(f"{old_mesh.name}.wedge_qem")
    mesh.from_pydata(result["vertices"], [], result["faces"])
    mesh.update()

    for slot in source_obj.material_slots:
        mesh.materials.append(slot.material)

    for poly, mat_index in zip(mesh.polygons, result.get("face_materials", [])):
        poly.material_index = min(int(mat_index), max(0, len(mesh.materials) - 1))

    if result.get("uvs"):
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            for loop_index, vert_index in zip(poly.loop_indices, poly.vertices):
                if vert_index < len(result["uvs"]):
                    uv_layer.data[loop_index].uv = result["uvs"][vert_index]

    if write_custom_normals and result.get("vertex_domains"):
        try:
            smooth_normals = vehicle_reducers.compute_smooth_normals(
                result["vertices"], result["faces"], result.get("vertex_domains")
            )
            loop_normals = []
            for poly in mesh.polygons:
                for vert_index in poly.vertices:
                    if vert_index < len(smooth_normals):
                        loop_normals.append(tuple(float(v) for v in smooth_normals[vert_index]))
                    else:
                        loop_normals.append((0.0, 0.0, 1.0))
            mesh.normals_split_custom_set(loop_normals)
            if hasattr(mesh, "use_auto_smooth"):
                mesh.use_auto_smooth = True
            for poly in mesh.polygons:
                poly.use_smooth = True
            mesh.update()
        except Exception:
            pass

    obj.data = mesh
    # Copy material and vertex-group payloads from the wedge result onto the generated LOD object.
    while obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups[0])
    group_names = sorted({name for weights in result.get("group_weights", []) for name in weights})
    groups = {name: obj.vertex_groups.new(name=name) for name in group_names}
    for vertex_index, weights in enumerate(result.get("group_weights", [])):
        for group_name, weight in weights.items():
            if weight > 0.0 and group_name in groups:
                groups[group_name].add([vertex_index], weight, "REPLACE")

    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    return obj

def mesh_surface_area(mesh):
    return sum(poly.area for poly in mesh.polygons)


def bbox_from_vertices(vertices):
    if not vertices:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "dimensions": [0.0, 0.0, 0.0]}
    mins = [min(vertex[i] for vertex in vertices) for i in range(3)]
    maxs = [max(vertex[i] for vertex in vertices) for i in range(3)]
    return {
        "min": [round(float(value), 6) for value in mins],
        "max": [round(float(value), 6) for value in maxs],
        "dimensions": [round(float(maxs[i] - mins[i]), 6) for i in range(3)],
    }


def tri_area_3d(vertices, face):
    a = Vector(vertices[face[0]])
    b = Vector(vertices[face[1]])
    c = Vector(vertices[face[2]])
    return 0.5 * (b - a).cross(c - a).length


def tri_area_uv(uvs, face):
    a = uvs[face[0]]
    b = uvs[face[1]]
    c = uvs[face[2]]
    return abs(
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    ) * 0.5


def mesh_payload_stats(payload):
    vertices = payload["vertices"]
    faces = payload["faces"]
    uvs = payload["uvs"]
    materials = payload.get("face_materials", [])
    surface_area = sum(tri_area_3d(vertices, face) for face in faces)
    uv_area = sum(tri_area_uv(uvs, face) for face in faces)
    return {
        "vertices": len(vertices),
        "triangles": len(faces),
        "bbox": bbox_from_vertices(vertices),
        "surface_area": round(float(surface_area), 6),
        "uv_area": round(float(uv_area), 6),
        "material_set": sorted(set(int(m) for m in materials)),
    }


def compare_payloads(source, reduced):
    before = mesh_payload_stats(source)
    after = mesh_payload_stats(reduced)
    bbox_delta = [
        round(after["bbox"]["dimensions"][i] - before["bbox"]["dimensions"][i], 6)
        for i in range(3)
    ]
    return {
        "before": before,
        "after": after,
        "triangle_ratio": round(after["triangles"] / max(1, before["triangles"]), 4),
        "vertex_ratio": round(after["vertices"] / max(1, before["vertices"]), 4),
        "surface_area_ratio": round(after["surface_area"] / max(1e-12, before["surface_area"]), 4),
        "uv_area_ratio": round(after["uv_area"] / max(1e-12, before["uv_area"]), 4),
        "bbox_dimension_delta": bbox_delta,
        "materials_preserved": before["material_set"] == after["material_set"],
    }


def validation_gate(validation, settings):
    warnings = []
    surface_ratio = validation["surface_area_ratio"]
    uv_ratio = validation["uv_area_ratio"]
    bbox_delta = validation["bbox_dimension_delta"]
    before_dims = validation["before"]["bbox"]["dimensions"]

    if not validation["materials_preserved"]:
        warnings.append("material set changed")

    surface_tol = settings.validation_surface_tolerance
    if surface_ratio < (1.0 - surface_tol) or surface_ratio > (1.0 + surface_tol):
        warnings.append(f"surface area ratio outside tolerance: {surface_ratio}")

    uv_tol = settings.validation_uv_tolerance
    if uv_ratio < (1.0 - uv_tol) or uv_ratio > (1.0 + uv_tol):
        warnings.append(f"uv area ratio outside tolerance: {uv_ratio}")

    bbox_tol = settings.validation_bbox_tolerance
    for axis, delta in zip(("x", "y", "z"), bbox_delta):
        base = max(abs(before_dims[{"x": 0, "y": 1, "z": 2}[axis]]), 1e-9)
        if abs(delta) / base > bbox_tol:
            warnings.append(f"bbox {axis} dimension changed by {round(abs(delta) / base, 4)}")

    triangle_ratio = validation["triangle_ratio"]
    if triangle_ratio > 0.98:
        warnings.append("little or no reduction achieved")

    return {
        "passed": not warnings,
        "warnings": warnings,
    }


def create_collision_box_proxy(context, source_obj, collection):
    bbox = [source_obj.matrix_world @ Vector(corner) for corner in source_obj.bound_box]
    xs = [v.x for v in bbox]
    ys = [v.y for v in bbox]
    zs = [v.z for v in bbox]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    verts = [
        (min_x, min_y, min_z), (max_x, min_y, min_z), (max_x, max_y, min_z), (min_x, max_y, min_z),
        (min_x, min_y, max_z), (max_x, min_y, max_z), (max_x, max_y, max_z), (min_x, max_y, max_z),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}.collision_box")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{source_obj.name}.COLLISION_BOX_PROXY", mesh)
    collection.objects.link(obj)
    return obj


def create_convex_proxy(context, source_obj, hulls, collection):
    """Build one collision proxy object containing all convex hull islands."""
    verts = []
    faces = []
    for hull in hulls:
        offset = len(verts)
        verts.extend(tuple(float(c) for c in v) for v in hull["vertices"])
        faces.extend(tuple(idx + offset for idx in f) for f in hull["faces"])
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}.collision_hull")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{source_obj.name}.COLLISION_HULL_PROXY", mesh)
    collection.objects.link(obj)
    obj.matrix_world = source_obj.matrix_world.copy()
    obj.display_type = "WIRE" if hasattr(obj, "display_type") else obj.display_type
    return obj


def classify_part(obj, stats):
    name = obj.name.lower()
    mat_names = " ".join(slot.material.name.lower() for slot in obj.material_slots if slot.material)
    dims = obj.dimensions
    max_dim = max(dims) if dims else 0.0
    min_dim = min(dims) if dims else 0.0
    tri = stats["triangles"]

    if any(token in name for token in ("wheel", "tyre", "tire", "rim")):
        return "wheel_or_tyre"
    if any(token in name for token in ("glass", "window", "windscreen")) or "glass" in mat_names:
        return "glass"
    if any(token in name for token in ("turret", "gun", "barrel", "weapon")):
        return "turret_or_weapon"
    if any(token in name for token in ("door", "hatch", "bonnet", "hood")):
        return "animated_panel"
    if any(token in name for token in ("interior", "seat", "dash", "cabin")):
        return "interior"
    if any(token in name for token in ("bolt", "rivet", "nut", "washer", "screw")):
        return "tiny_detail"
    if max_dim > 0 and min_dim / max_dim < 0.035 and tri > 20:
        return "thin_panel_or_trim"
    if tri < 64 and max_dim < 0.25:
        return "tiny_detail"
    return "general_body"


def recommended_method(part_type):
    return {
        "wheel_or_tyre": "radial/manual or protected collapse",
        "glass": "skip or planar only",
        "turret_or_weapon": "protected planar/collapse, preserve pivots",
        "animated_panel": "protected planar, preserve boundaries",
        "interior": "visibility cull then collapse",
        "tiny_detail": "delete/instance/bake to normal",
        "thin_panel_or_trim": "planar dissolve",
        "general_body": "planar first, then attribute-aware QEM",
    }.get(part_type, "inspect")


def analyse_mesh_object(obj, angle_limit_rad):
    mesh = obj.data
    mesh.calc_loop_triangles()
    edge_usage = edge_polygon_map(mesh)
    boundary_edges = {key for key, polys in edge_usage.items() if len(polys) == 1}
    material_edges = material_boundary_edges(mesh, edge_usage)
    uv_edges = explicit_seam_edges(mesh) | uv_island_edges(mesh, edge_usage)
    normal_edges = sharp_flag_edges(mesh) | normal_boundary_edges(mesh, edge_usage, angle_limit_rad)
    protected_edges = boundary_edges | material_edges | uv_edges | normal_edges
    protected_vertices = {v for edge in protected_edges for v in edge}
    area = mesh_surface_area(mesh)

    stats = {
        "name": obj.name,
        "triangles": tri_count(mesh),
        "vertices": len(mesh.vertices),
        "edges": len(mesh.edges),
        "dimensions": object_world_dims(obj),
        "surface_area": round(area, 5),
        "materials": len(obj.material_slots),
        "has_uv": bool(mesh.uv_layers.active),
        "boundary_edges": len(boundary_edges),
        "uv_seam_edges": len(uv_edges),
        "material_boundary_edges": len(material_edges),
        "hard_normal_edges": len(normal_edges),
        "protected_edges": len(protected_edges),
        "protected_vertices": len(protected_vertices),
    }
    stats["protected_vertex_ratio"] = round(len(protected_vertices) / max(1, len(mesh.vertices)), 4)
    stats["part_type"] = classify_part(obj, stats)
    stats["recommended_method"] = recommended_method(stats["part_type"])
    stats["risk"] = classify_risk(stats)
    return stats


def classify_risk(stats):
    if not stats["has_uv"]:
        return "medium - no UV layer found"
    if stats["protected_vertex_ratio"] > 0.72:
        return "high - many seams/hard boundaries"
    if stats["part_type"] in {"glass", "wheel_or_tyre", "turret_or_weapon"}:
        return "medium - important silhouette/pivot part"
    if stats["part_type"] == "tiny_detail":
        return "low - candidate for delete/bake"
    return "low"


def selected_mesh_objects(context):
    return [obj for obj in context.selected_objects if obj.type == "MESH"]


def write_report(report):
    text = bpy.data.texts.get(LOD_TEXT_NAME) or bpy.data.texts.new(LOD_TEXT_NAME)
    text.clear()
    text.write(json.dumps(report, indent=2))

    base_dir = bpy.path.abspath("//") if bpy.data.filepath else tempfile.gettempdir()
    out_path = os.path.join(base_dir, "vehicle_smart_lod_report.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return out_path


def create_protection_vertex_group(obj, angle_limit_rad):
    mesh = obj.data
    protected_vertices, _ = protection_sets(mesh, angle_limit_rad)
    protected_vertices = sorted(protected_vertices)
    group = obj.vertex_groups.get("VLOD_PROTECTED") or obj.vertex_groups.new(name="VLOD_PROTECTED")
    if protected_vertices:
        group.add(protected_vertices, 1.0, "REPLACE")
    return group, len(protected_vertices)


def duplicate_for_lod(obj, lod_name, collection):
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    new_obj.name = f"{obj.name}.{lod_name}"
    new_obj.data.name = f"{obj.data.name}.{lod_name}"
    collection.objects.link(new_obj)
    new_obj.matrix_world = obj.matrix_world.copy()
    return new_obj


def _apply_modifier_safely(obj, mod_name):
    """Apply a modifier in a context-safe way.

    bpy.ops.object.modifier_apply depends on mode + active object + selection,
    which is fragile inside loops. We force OBJECT mode, isolate the selection
    to this object, then apply. Returns True on success.
    """
    try:
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    try:
        for other in bpy.context.selected_objects:
            other.select_set(False)
    except Exception:
        pass
    try:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod_name)
        return True
    except Exception:
        return False


def apply_decimate_modifier(obj, settings, ratio, part_type):
    if part_type in {"glass"} and settings.skip_glass:
        return "skipped glass"
    if part_type in {"tiny_detail"} and settings.delete_tiny_details:
        bpy.data.objects.remove(obj, do_unlink=True)
        return "deleted tiny detail"

    if part_type in {"general_body", "thin_panel_or_trim", "animated_panel", "turret_or_weapon", "glass"}:
        mod = obj.modifiers.new("VLOD_planar_preserve", "DECIMATE")
        mod.decimate_type = "DISSOLVE"
        mod.angle_limit = math.radians(settings.planar_angle_degrees)
        if hasattr(mod, "delimit"):
            mod.delimit = {"NORMAL", "MATERIAL", "SEAM", "UV"}
        if not _apply_modifier_safely(obj, mod.name):
            return "planar modifier left unapplied"

    if ratio < 0.98 and part_type not in {"glass"}:
        mod = obj.modifiers.new("VLOD_collapse", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        if not _apply_modifier_safely(obj, mod.name):
            return "collapse modifier left unapplied"

    return "reduced"


class VLOD_Settings(bpy.types.PropertyGroup):
    lod1_ratio: bpy.props.FloatProperty(
        name="LOD1 Ratio",
        description="Triangle ratio for mid-distance LOD",
        default=0.55,
        min=0.05,
        max=1.0,
    )
    lod2_ratio: bpy.props.FloatProperty(
        name="LOD2 Ratio",
        description="Triangle ratio for shadow/distance LOD",
        default=0.22,
        min=0.02,
        max=1.0,
    )
    lod3_ratio: bpy.props.FloatProperty(
        name="LOD3 Ratio",
        description="Triangle ratio for far/proxy LOD",
        default=0.08,
        min=0.005,
        max=1.0,
    )
    lod_generation_method: bpy.props.EnumProperty(
        name="Generation Method",
        description="Reducer used when generating production LOD output meshes",
        items=[
            ("DECIMATE", "Blender Decimate", "Use the existing Blender Decimate modifier workflow"),
            ("WEDGE_QEM", "Wedge QEM", "Use UV/material/normal-domain-safe wedge QEM reduction"),
        ],
        default="DECIMATE",
    )
    lod1_target_triangles: bpy.props.IntProperty(
        name="LOD1 Target Tris (0=ratio)",
        description="Absolute triangle budget for LOD1 when using Wedge QEM. 0 uses LOD1 Ratio.",
        default=0,
        min=0,
        max=5000000,
    )
    lod2_target_triangles: bpy.props.IntProperty(
        name="LOD2 Target Tris (0=ratio)",
        description="Absolute triangle budget for LOD2 when using Wedge QEM. 0 uses LOD2 Ratio.",
        default=0,
        min=0,
        max=5000000,
    )
    lod3_target_triangles: bpy.props.IntProperty(
        name="LOD3 Target Tris (0=ratio)",
        description="Absolute triangle budget for LOD3 when using Wedge QEM. 0 uses LOD3 Ratio.",
        default=0,
        min=0,
        max=5000000,
    )
    planar_angle_degrees: bpy.props.FloatProperty(
        name="Planar Angle",
        description="Angle threshold for planar dissolve",
        default=5.0,
        min=0.1,
        max=45.0,
    )
    hard_edge_angle_degrees: bpy.props.FloatProperty(
        name="Hard Edge Angle",
        description="Edges above this normal angle are protected in analysis",
        default=35.0,
        min=1.0,
        max=120.0,
    )
    create_lod1: bpy.props.BoolProperty(name="Create LOD1", default=True)
    create_lod2: bpy.props.BoolProperty(name="Create LOD2", default=True)
    create_lod3: bpy.props.BoolProperty(name="Create LOD3", default=True)
    skip_glass: bpy.props.BoolProperty(name="Skip Glass", default=True)
    delete_tiny_details: bpy.props.BoolProperty(
        name="Delete Tiny Detail In Lower LODs",
        default=False,
        description="Removes objects classified as bolts/rivets/tiny details in generated LODs",
    )
    qem_candidate_limit: bpy.props.IntProperty(
        name="QEM Candidates",
        description="Number of best edge-collapse candidates to include per object in the QEM plan",
        default=50,
        min=5,
        max=500,
    )
    qem_uv_safe: bpy.props.BoolProperty(
        name="UV-Safe QEM",
        description="Lock UV wedge vertices and score collapses with UV distance penalties",
        default=True,
    )
    qem_uv_weight: bpy.props.FloatProperty(
        name="UV Weight",
        description="Higher values make QEM collapse planning avoid UV distortion more aggressively",
        default=25.0,
        min=0.0,
        max=10000.0,
    )
    qem_uv_distance_limit: bpy.props.FloatProperty(
        name="UV Distance Limit",
        description="Reject collapse candidates whose endpoint UVs are farther apart than this",
        default=0.08,
        min=0.0,
        max=10.0,
    )
    wedge_preview_ratio: bpy.props.FloatProperty(
        name="Preview Ratio",
        description="Target face ratio for experimental wedge QEM preview meshes",
        default=0.75,
        min=0.05,
        max=1.0,
    )
    wedge_target_triangles: bpy.props.IntProperty(
        name="Target Triangles (0=ratio)",
        description=(
            "Absolute triangle budget for wedge previews (AAA LOD budgeting). "
            "When > 0 this overrides Preview Ratio and the reducer targets this "
            "exact triangle count, allocated across mesh islands by size. 0 uses "
            "Preview Ratio instead."
        ),
        default=0,
        min=0,
        max=5000000,
    )
    wedge_lock_borders: bpy.props.BoolProperty(
        name="Lock Wedge Borders",
        description="Lock UV/material/normal chart borders during wedge QEM preview reduction",
        default=True,
    )
    wedge_partition_islands: bpy.props.BoolProperty(
        name="Partition Mesh Islands",
        description="Reduce disconnected mesh islands separately before safely merging preview output",
        default=True,
    )
    wedge_reject_face_flips: bpy.props.BoolProperty(
        name="Reject Face Flips",
        description="Reject QEM collapses that flip or flatten adjacent triangles",
        default=True,
    )
    wedge_write_custom_normals: bpy.props.BoolProperty(
        name="Write Custom Normals",
        description="Attempt to write preserved split normals to preview meshes",
        default=True,
    )
    max_preview_triangles: bpy.props.IntProperty(
        name="Max Preview Tris",
        description="Skip wedge preview generation above this source triangle count unless heavy previews are allowed",
        default=30000,
        min=1000,
        max=1000000,
    )
    wedge_max_iterations: bpy.props.IntProperty(
        name="Max QEM Iterations",
        description="Hard stop for QEM edge collapses per mesh or partition",
        default=5000,
        min=100,
        max=200000,
    )
    allow_heavy_preview: bpy.props.BoolProperty(
        name="Allow Heavy Preview",
        description="Allow wedge preview on meshes above Max Preview Tris. Use carefully.",
        default=False,
    )
    safe_test_object_limit: bpy.props.IntProperty(
        name="Safe Test Object Limit",
        description="Maximum number of selected objects processed by Safe Test Pack",
        default=3,
        min=1,
        max=20,
    )
    validation_surface_tolerance: bpy.props.FloatProperty(
        name="Surface Tolerance",
        description="Allowed surface-area ratio deviation before validation warns",
        default=0.25,
        min=0.01,
        max=2.0,
    )
    validation_uv_tolerance: bpy.props.FloatProperty(
        name="UV Tolerance",
        description="Allowed UV-area ratio deviation before validation warns",
        default=0.35,
        min=0.01,
        max=2.0,
    )
    validation_bbox_tolerance: bpy.props.FloatProperty(
        name="BBox Tolerance",
        description="Allowed bounding-box dimension change ratio before validation warns",
        default=0.05,
        min=0.001,
        max=1.0,
    )
    create_collision_proxy: bpy.props.BoolProperty(
        name="Create Collision Box Proxy",
        description="Create a simple bounding-box collision proxy for each selected object",
        default=False,
    )
    boundary_weight: bpy.props.FloatProperty(
        name="Boundary Weight",
        description="Cost penalty that holds open mesh borders (panel edges, caps) in place instead of hard-locking them. 0 disables.",
        default=2.0,
        min=0.0,
        max=100.0,
    )
    preserve_manifold: bpy.props.BoolProperty(
        name="Preserve Manifold",
        description="Reject edge collapses that would create non-manifold topology (recommended for engine import)",
        default=True,
    )
    radial_target_segments: bpy.props.IntProperty(
        name="Radial Segments",
        description="Target number of angular segments for the radial tyre reducer",
        default=16,
        min=3,
        max=256,
    )
    radial_confidence_threshold: bpy.props.FloatProperty(
        name="Radial Confidence Min",
        description="Skip radial reduction when the detected spin-axis confidence is below this (protects non-radial parts)",
        default=0.6,
        min=0.0,
        max=1.0,
    )
    cull_samples: bpy.props.IntProperty(
        name="Cull View Samples",
        description="Number of exterior viewpoints used by the visibility culler. Higher is safer but slower.",
        default=64,
        min=8,
        max=512,
    )
    cull_margin: bpy.props.FloatProperty(
        name="Cull View Margin",
        description="Viewpoint sphere radius as a multiple of the mesh size",
        default=1.5,
        min=1.0,
        max=10.0,
    )
    cull_max_faces: bpy.props.IntProperty(
        name="Cull Max Faces",
        description="Skip visibility culling above this triangle count (the culler is O(faces) per ray)",
        default=8000,
        min=100,
        max=200000,
    )
    collision_method: bpy.props.EnumProperty(
        name="Collision Proxy",
        description="How to build the collision proxy object",
        items=[
            ("BOX", "Bounding Box", "Single axis-aligned bounding box (cheapest)"),
            ("HULL", "Convex Hull", "Single convex hull hugging the silhouette"),
            ("DECOMPOSE", "Convex Decomposition", "Multiple convex hulls hugging concavities"),
        ],
        default="HULL",
    )
    collision_max_hulls: bpy.props.IntProperty(
        name="Max Collision Hulls",
        description="Maximum convex hulls produced by convex decomposition",
        default=4,
        min=1,
        max=32,
    )


class VLOD_OT_analyse(bpy.types.Operator):
    bl_idname = "vlod.analyse"
    bl_label = "Analyse Selected Vehicle"
    bl_description = "Analyse selected mesh objects and produce a constraint-aware LOD report"
    bl_options = {"REGISTER"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        angle_limit = math.radians(settings.hard_edge_angle_degrees)
        parts = [analyse_mesh_object(obj, angle_limit) for obj in objects]
        total_tris = sum(part["triangles"] for part in parts)
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "object_count": len(parts),
            "total_triangles": total_tris,
            "total_vertices": sum(part["vertices"] for part in parts),
            "part_type_counts": dict(Counter(part["part_type"] for part in parts)),
            "risk_counts": dict(Counter(part["risk"].split(" - ")[0] for part in parts)),
            "parts": sorted(parts, key=lambda item: item["triangles"], reverse=True),
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Vehicle LOD report written: {out_path}")
        return {"FINISHED"}


class VLOD_OT_mark_constraints(bpy.types.Operator):
    bl_idname = "vlod.mark_constraints"
    bl_label = "Mark Protected Vertices"
    bl_description = "Create VLOD_PROTECTED vertex groups from UV seams, material borders, boundaries, and hard edges"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        angle_limit = math.radians(settings.hard_edge_angle_degrees)
        total = 0
        for obj in objects:
            _, count = create_protection_vertex_group(obj, angle_limit)
            total += count
        self.report({"INFO"}, f"Marked {total} protected vertices.")
        return {"FINISHED"}


class VLOD_OT_qem_plan(bpy.types.Operator):
    bl_idname = "vlod.qem_plan"
    bl_label = "Build QEM Collapse Plan"
    bl_description = "Calculate constraint-aware QEM edge-collapse candidates without modifying meshes"
    bl_options = {"REGISTER"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        angle_limit = math.radians(settings.hard_edge_angle_degrees)
        plans = []

        for obj in objects:
            mesh = obj.data
            protected_vertices, protected_edges = protection_sets(mesh, angle_limit)
            vertices, faces, edges = mesh_qem_data(mesh)
            vertex_uvs, uv_locked, uv_stats = uv_safety_data(mesh)
            if settings.qem_uv_safe:
                protected_vertices = set(protected_vertices) | set(uv_locked)
                qem_vertex_uvs = vertex_uvs
                qem_uv_locked = uv_locked
                qem_uv_weight = settings.qem_uv_weight
                qem_uv_distance_limit = settings.qem_uv_distance_limit
            else:
                qem_vertex_uvs = None
                qem_uv_locked = set()
                qem_uv_weight = 0.0
                qem_uv_distance_limit = None

            qem_plan = qem_backend.analyse_candidates(
                vertices,
                faces,
                edges,
                protected_vertices=protected_vertices,
                protected_edges=protected_edges,
                limit=settings.qem_candidate_limit,
                vertex_uvs=qem_vertex_uvs,
                uv_locked_vertices=qem_uv_locked,
                uv_weight=qem_uv_weight,
                uv_distance_limit=qem_uv_distance_limit,
                boundary_weight=settings.boundary_weight,
            )
            stats = analyse_mesh_object(obj, angle_limit)
            plans.append({
                "name": obj.name,
                "part_type": stats["part_type"],
                "triangles": stats["triangles"],
                "vertices": stats["vertices"],
                "edges": stats["edges"],
                "protected_vertices": len(protected_vertices),
                "protected_edges": len(protected_edges),
                "uv_safe_qem": bool(settings.qem_uv_safe),
                "uv_stats": uv_stats,
                "qem": qem_plan,
            })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "constraint_aware_qem_plan",
            "object_count": len(plans),
            "notes": [
                "This is a non-destructive QEM planning pass.",
                "Best candidates are low-cost edge collapses that do not cross protected boundaries.",
                "Actual mesh mutation should only be enabled after plans are inspected on real vehicle assets.",
            ],
            "objects": sorted(plans, key=lambda item: item["triangles"], reverse=True),
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"QEM plan written: {out_path}")
        return {"FINISHED"}


class VLOD_OT_wedge_preview(bpy.types.Operator):
    bl_idname = "vlod.wedge_preview"
    bl_label = "Generate Wedge QEM Preview"
    bl_description = "Create non-destructive UV-preserving wedge QEM preview meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        collection = bpy.data.collections.new("Vehicle Smart LOD Wedge QEM Preview")
        context.scene.collection.children.link(collection)
        collision_collection = None
        if settings.create_collision_proxy:
            collision_collection = bpy.data.collections.new("Vehicle Smart LOD Collision Proxies")
            context.scene.collection.children.link(collision_collection)
        results = []

        for obj in objects:
            source = build_wedge_source(obj, math.radians(settings.hard_edge_angle_degrees))
            if not source["faces"]:
                results.append({
                    "source": obj.name,
                    "status": "skipped",
                    "reason": "no triangulatable faces",
                })
                continue
            if len(source["faces"]) > settings.max_preview_triangles and not settings.allow_heavy_preview:
                results.append({
                    "source": obj.name,
                    "status": "skipped",
                    "reason": f"source triangle count {len(source['faces'])} exceeds Max Preview Tris {settings.max_preview_triangles}",
                    "source_triangles": len(source["faces"]),
                    "suggestion": "test a smaller part first or enable Allow Heavy Preview",
                })
                continue

            reducer = wedge_backend.simplify_wedge_mesh_partitioned if settings.wedge_partition_islands else wedge_backend.simplify_wedge_mesh
            target_faces = settings.wedge_target_triangles or None
            reduced = reducer(
                    source["vertices"],
                    source["uvs"],
                    source["faces"],
                    face_materials=source["face_materials"],
                    vertex_domains=source["vertex_domains"],
                    group_weights=source["group_weights"],
                    target_ratio=settings.wedge_preview_ratio,
                    uv_weight=settings.qem_uv_weight,
                    uv_distance_limit=settings.qem_uv_distance_limit,
                    lock_border_vertices=settings.wedge_lock_borders,
                    allow_domain_crossing=False,
                    safe_weld=True,
                    reject_face_flips=settings.wedge_reject_face_flips,
                    # auto-raise the iteration ceiling so the target is actually
                    # reachable: each collapse removes ~2 faces, so a low cap
                    # silently under-reduces large parts.
                    max_iterations=max(settings.wedge_max_iterations, len(source["faces"])),
                    boundary_weight=settings.boundary_weight,
                    preserve_manifold=settings.preserve_manifold,
                    target_faces=target_faces,
            )
            preview = create_wedge_preview_object(
                context,
                obj,
                reduced,
                collection,
                write_custom_normals=settings.wedge_write_custom_normals,
            )
            validation = compare_payloads(source, reduced)
            gate = validation_gate(validation, settings)
            collision_name = None
            if collision_collection:
                collision_name = create_collision_box_proxy(context, obj, collision_collection).name
            results.append({
                "source": obj.name,
                "preview": preview.name,
                "collision_proxy": collision_name,
                "status": "created",
                "source_mesh_vertices": source["source_vertices"],
                "wedge_vertices": source["wedge_vertices"],
                "source_polygons": source["source_faces"],
                "source_triangles": len(source["faces"]),
                "preview_triangles": len(reduced["faces"]),
                "preview_vertices": len(reduced["vertices"]),
                "preview_vertex_groups": len({name for weights in reduced.get("group_weights", []) for name in weights}),
                "stats": reduced["stats"],
                "validation": validation,
                "validation_gate": gate,
            })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "wedge_qem_preview",
            "notes": [
                "This creates new preview objects and does not modify the source meshes.",
                "The reducer operates on split wedge vertices so UV/material/normal domains are not merged blindly.",
                "Collapses across different UV/material/normal domains are rejected.",
                "Safe weld only rejoins vertices with identical position, UV, and domain.",
                "Vertex group weights are transferred to preview vertices using conservative max-weight merging.",
                "Validation compares triangle ratio, vertex ratio, surface area, UV area, bounding box, and material set.",
                "Border locking is conservative and may stop reduction early on heavily seamed meshes.",
            ],
            "results": results,
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Wedge QEM preview complete: {out_path}")
        return {"FINISHED"}


class VLOD_OT_radial_tyre(bpy.types.Operator):
    bl_idname = "vlod.radial_tyre"
    bl_label = "Radial Tyre Reduce"
    bl_description = "Decimate tyres/wheels in the angular direction, preserving the round cross-section profile"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more wheel/tyre mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        collection = bpy.data.collections.new("Vehicle Smart LOD Radial Preview")
        context.scene.collection.children.link(collection)
        results = []

        for obj in objects:
            source = build_wedge_source(obj, math.radians(settings.hard_edge_angle_degrees))
            if not source["faces"]:
                results.append({"source": obj.name, "status": "skipped", "reason": "no faces"})
                continue

            center, axis, confidence = vehicle_reducers.estimate_radial_axis(source["vertices"])
            if confidence < settings.radial_confidence_threshold:
                results.append({
                    "source": obj.name,
                    "status": "skipped",
                    "reason": f"radial confidence {confidence:.3f} below threshold {settings.radial_confidence_threshold:.2f}",
                    "detected_axis": [round(float(v), 4) for v in axis],
                    "suggestion": "this part may not be a surface of revolution; use wedge QEM instead",
                })
                continue

            reduced = vehicle_reducers.radial_reduce(
                source["vertices"],
                source["uvs"],
                source["faces"],
                face_materials=source["face_materials"],
                vertex_domains=source["vertex_domains"],
                group_weights=source["group_weights"],
                axis_point=center,
                axis_dir=axis,
                target_segments=settings.radial_target_segments,
            )
            preview = create_wedge_preview_object(
                context, obj, reduced, collection,
                write_custom_normals=settings.wedge_write_custom_normals,
                mesh_suffix="radial_preview", obj_suffix="RADIAL_PREVIEW",
            )
            validation = compare_payloads(source, reduced)
            results.append({
                "source": obj.name,
                "preview": preview.name,
                "status": "created",
                "radial_confidence": round(confidence, 4),
                "detected_axis": [round(float(v), 4) for v in axis],
                "target_segments": settings.radial_target_segments,
                "source_triangles": len(source["faces"]),
                "preview_triangles": len(reduced["faces"]),
                "stats": reduced["stats"],
                "validation": validation,
            })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "radial_tyre_reduce",
            "notes": [
                "Radial reduction snaps vertex angles to evenly spaced segments and welds rings.",
                "The cross-section profile (radius and axial height per vertex) is preserved.",
                "Parts below the radial confidence threshold are skipped, not damaged.",
                "Source meshes are not modified.",
            ],
            "results": results,
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Radial tyre reduce complete: {out_path}")
        return {"FINISHED"}


class VLOD_OT_visibility_cull(bpy.types.Operator):
    bl_idname = "vlod.visibility_cull"
    bl_label = "Cull Hidden Interior Geometry"
    bl_description = "Create a copy with interior faces (never visible from outside) removed"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        collection = bpy.data.collections.new("Vehicle Smart LOD Visibility Cull")
        context.scene.collection.children.link(collection)
        results = []

        for obj in objects:
            source = build_wedge_source(obj, math.radians(settings.hard_edge_angle_degrees))
            if not source["faces"]:
                results.append({"source": obj.name, "status": "skipped", "reason": "no faces"})
                continue
            if len(source["faces"]) > settings.cull_max_faces:
                results.append({
                    "source": obj.name,
                    "status": "skipped",
                    "reason": f"triangle count {len(source['faces'])} exceeds Cull Max Faces {settings.cull_max_faces}",
                    "suggestion": "raise Cull Max Faces or pre-reduce first",
                })
                continue

            kept_faces, flags = vehicle_reducers.visibility_cull(
                source["vertices"],
                source["faces"],
                samples=settings.cull_samples,
                margin=settings.cull_margin,
                return_mask=True,
            )
            culled = len(source["faces"]) - len(kept_faces)
            kept_materials = [m for m, keep in zip(source["face_materials"], flags) if keep]

            payload = {
                "vertices": source["vertices"],
                "uvs": source["uvs"],
                "vertex_domains": source["vertex_domains"],
                "group_weights": source["group_weights"],
                "faces": kept_faces,
                "face_materials": kept_materials,
            }
            # compact away now-unused vertices
            compacted = wedge_backend.compact_mesh(
                payload["vertices"], payload["uvs"], payload["vertex_domains"],
                payload["group_weights"], payload["faces"], payload["face_materials"],
            )
            payload = {
                "vertices": compacted[0], "uvs": compacted[1], "vertex_domains": compacted[2],
                "group_weights": compacted[3], "faces": compacted[4], "face_materials": compacted[5],
            }
            preview = create_wedge_preview_object(
                context, obj, payload, collection,
                write_custom_normals=settings.wedge_write_custom_normals,
                mesh_suffix="visibility_cull", obj_suffix="VISIBILITY_CULL",
            )
            results.append({
                "source": obj.name,
                "preview": preview.name,
                "status": "created",
                "source_triangles": len(source["faces"]),
                "kept_triangles": len(kept_faces),
                "culled_triangles": culled,
                "culled_ratio": round(culled / max(1, len(source["faces"])), 4),
                "view_samples": settings.cull_samples,
            })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "visibility_cull",
            "notes": [
                "Faces not reachable from any exterior viewpoint are removed.",
                "Use more view samples for safety on complex shapes; fewer for speed.",
                "Best run on interior/chassis parts after the exterior shell is finalised.",
                "Source meshes are not modified.",
            ],
            "results": results,
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Visibility cull complete: {out_path}")
        return {"FINISHED"}


class VLOD_OT_collision_hull(bpy.types.Operator):
    bl_idname = "vlod.collision_hull"
    bl_label = "Build Collision Proxy"
    bl_description = "Generate a bounding-box, convex-hull, or convex-decomposition collision proxy"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        collection = bpy.data.collections.new("Vehicle Smart LOD Collision Proxies")
        context.scene.collection.children.link(collection)
        results = []

        for obj in objects:
            if settings.collision_method == "BOX":
                proxy = create_collision_box_proxy(context, obj, collection)
                results.append({
                    "source": obj.name, "proxy": proxy.name,
                    "method": "BOX", "status": "created",
                })
                continue

            source = build_wedge_source(obj, math.radians(settings.hard_edge_angle_degrees))
            if not source["vertices"]:
                results.append({"source": obj.name, "status": "skipped", "reason": "no vertices"})
                continue

            max_hulls = 1 if settings.collision_method == "HULL" else settings.collision_max_hulls
            hulls = vehicle_reducers.convex_decompose(source["vertices"], max_hulls=max_hulls)
            if not hulls:
                results.append({"source": obj.name, "status": "skipped", "reason": "hull generation failed"})
                continue
            proxy = create_convex_proxy(context, obj, hulls, collection)
            total_volume = sum(
                vehicle_reducers.hull_volume(h["vertices"], h["faces"]) for h in hulls
            )
            results.append({
                "source": obj.name,
                "proxy": proxy.name,
                "method": settings.collision_method,
                "status": "created",
                "hull_count": len(hulls),
                "proxy_triangles": sum(len(h["faces"]) for h in hulls),
                "proxy_volume": round(total_volume, 6),
            })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "collision_proxy",
            "notes": [
                "BOX: one axis-aligned bounding box. HULL: one convex hull. DECOMPOSE: clustered multi-hull.",
                "Convex decomposition uses deterministic spatial clustering then a hull per cluster.",
                "Proxies are separate objects; source meshes are not modified.",
            ],
            "results": results,
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Collision proxy complete: {out_path}")
        return {"FINISHED"}


class VLOD_OT_backend_selftest(bpy.types.Operator):
    bl_idname = "vlod.backend_selftest"
    bl_label = "Run Backend Self-Test"
    bl_description = "Run a small wedge QEM backend test without using scene geometry"
    bl_options = {"REGISTER"}

    def execute(self, context):
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        ]
        uvs = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        faces = [(0, 1, 2), (0, 2, 3), (1, 4, 5), (1, 5, 2)]
        domains = [(0, (0.0, 0.0, 1.0)) for _ in vertices]
        weights = [{"selftest": 1.0} for _ in vertices]
        result = wedge_backend.simplify_wedge_mesh_partitioned(
            vertices,
            uvs,
            faces,
            face_materials=[0, 0, 0, 0],
            vertex_domains=domains,
            group_weights=weights,
            target_ratio=0.75,
            uv_weight=25.0,
            uv_distance_limit=0.8,
            lock_border_vertices=False,
            reject_face_flips=True,
        )
        ok = result["stats"]["end_faces"] <= 4 and result["stats"]["end_faces"] >= 1
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "backend_selftest",
            "passed": bool(ok),
            "stats": result["stats"],
        }
        out_path = write_report(report)
        if ok:
            self.report({"INFO"}, f"Backend self-test passed: {out_path}")
            return {"FINISHED"}
        self.report({"ERROR"}, f"Backend self-test failed: {out_path}")
        return {"CANCELLED"}


class VLOD_OT_safe_test_pack(bpy.types.Operator):
    bl_idname = "vlod.safe_test_pack"
    bl_label = "Build Safe Test Pack"
    bl_description = "Run analysis, mark constraints, and generate safe previews for a limited number of selected objects"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        objects = selected_mesh_objects(context)
        if not objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        angle_limit = math.radians(settings.hard_edge_angle_degrees)
        analysed = [(obj, analyse_mesh_object(obj, angle_limit)) for obj in objects]
        candidates = [
            (obj, stats) for obj, stats in analysed
            if stats["triangles"] <= settings.max_preview_triangles or settings.allow_heavy_preview
        ]
        candidates.sort(key=lambda item: item[1]["triangles"], reverse=True)
        candidates = candidates[:settings.safe_test_object_limit]

        preview_collection = bpy.data.collections.new("Vehicle Smart LOD Safe Test Pack")
        context.scene.collection.children.link(preview_collection)
        collision_collection = bpy.data.collections.new("Vehicle Smart LOD Safe Test Collision")
        context.scene.collection.children.link(collision_collection)

        marked_vertices = 0
        for obj, _ in analysed:
            _, count = create_protection_vertex_group(obj, angle_limit)
            marked_vertices += count

        results = []
        skipped = []
        for obj, stats in analysed:
            if obj not in [candidate[0] for candidate in candidates]:
                skipped.append({
                    "source": obj.name,
                    "triangles": stats["triangles"],
                    "reason": "not in safe preview candidate set or above triangle guard",
                })

        for obj, stats in candidates:
            source = build_wedge_source(obj, math.radians(settings.hard_edge_angle_degrees))
            if not source["faces"]:
                results.append({
                    "source": obj.name,
                    "status": "skipped",
                    "reason": "no triangulatable faces",
                })
                continue

            reduced = wedge_backend.simplify_wedge_mesh_partitioned(
                source["vertices"],
                source["uvs"],
                source["faces"],
                face_materials=source["face_materials"],
                vertex_domains=source["vertex_domains"],
                group_weights=source["group_weights"],
                target_ratio=settings.wedge_preview_ratio,
                uv_weight=settings.qem_uv_weight,
                uv_distance_limit=settings.qem_uv_distance_limit,
                lock_border_vertices=settings.wedge_lock_borders,
                allow_domain_crossing=False,
                safe_weld=True,
                reject_face_flips=settings.wedge_reject_face_flips,
                max_iterations=max(settings.wedge_max_iterations, len(source["faces"])),
                boundary_weight=settings.boundary_weight,
                preserve_manifold=settings.preserve_manifold,
                target_faces=settings.wedge_target_triangles or None,
            )
            preview = create_wedge_preview_object(
                context,
                obj,
                reduced,
                preview_collection,
                write_custom_normals=settings.wedge_write_custom_normals,
            )
            collision = create_collision_box_proxy(context, obj, collision_collection)
            validation = compare_payloads(source, reduced)
            gate = validation_gate(validation, settings)
            results.append({
                "source": obj.name,
                "preview": preview.name,
                "collision_proxy": collision.name,
                "status": "created",
                "part_type": stats["part_type"],
                "risk": stats["risk"],
                "source_triangles": len(source["faces"]),
                "preview_triangles": len(reduced["faces"]),
                "preview_vertices": len(reduced["vertices"]),
                "stats": reduced["stats"],
                "validation": validation,
                "validation_gate": gate,
            })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "safe_test_pack",
            "selected_objects": len(objects),
            "processed_objects": len(results),
            "marked_vertices": marked_vertices,
            "settings": {
                "preview_ratio": settings.wedge_preview_ratio,
                "max_preview_triangles": settings.max_preview_triangles,
                "safe_test_object_limit": settings.safe_test_object_limit,
                "uv_weight": settings.qem_uv_weight,
                "uv_distance_limit": settings.qem_uv_distance_limit,
                "max_iterations": settings.wedge_max_iterations,
            },
            "analysis": [stats for _, stats in sorted(analysed, key=lambda item: item[1]["triangles"], reverse=True)],
            "results": results,
            "skipped": skipped,
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Safe Test Pack complete: {out_path}")
        return {"FINISHED"}


class VLOD_OT_safe_defaults(bpy.types.Operator):
    bl_idname = "vlod.safe_defaults"
    bl_label = "Set Safe Test Defaults"
    bl_description = "Set conservative defaults for first vehicle testing"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.vehicle_smart_lod
        settings.wedge_preview_ratio = 0.75
        settings.qem_uv_safe = True
        settings.qem_uv_weight = 50.0
        settings.qem_uv_distance_limit = 0.05
        settings.wedge_lock_borders = True
        settings.wedge_partition_islands = True
        settings.wedge_reject_face_flips = True
        settings.wedge_write_custom_normals = True
        settings.max_preview_triangles = 30000
        settings.wedge_max_iterations = 5000
        settings.allow_heavy_preview = False
        settings.safe_test_object_limit = 3
        settings.validation_surface_tolerance = 0.25
        settings.validation_uv_tolerance = 0.35
        settings.validation_bbox_tolerance = 0.05
        settings.create_collision_proxy = False
        settings.boundary_weight = 2.0
        settings.preserve_manifold = True
        self.report({"INFO"}, "Safe test defaults applied.")
        return {"FINISHED"}


class VLOD_OT_generate(bpy.types.Operator):
    bl_idname = "vlod.generate"
    bl_label = "Generate Conservative LODs"
    bl_description = "Duplicate selected objects and generate conservative LOD meshes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source_objects = selected_mesh_objects(context)
        if not source_objects:
            self.report({"WARNING"}, "Select one or more mesh objects.")
            return {"CANCELLED"}

        settings = context.scene.vehicle_smart_lod
        lods = []
        if settings.create_lod1:
            lods.append(("LOD1", settings.lod1_ratio, settings.lod1_target_triangles))
        if settings.create_lod2:
            lods.append(("LOD2", settings.lod2_ratio, settings.lod2_target_triangles))
        if settings.create_lod3:
            lods.append(("LOD3", settings.lod3_ratio, settings.lod3_target_triangles))
        if not lods:
            self.report({"WARNING"}, "Enable at least one LOD output.")
            return {"CANCELLED"}

        collection = bpy.data.collections.new("Vehicle Smart LOD Output")
        context.scene.collection.children.link(collection)
        angle_limit = math.radians(settings.hard_edge_angle_degrees)
        results = []

        for lod_name, ratio, absolute_target_faces in lods:
            for obj in source_objects:
                stats = analyse_mesh_object(obj, angle_limit)
                source_triangles = stats["triangles"]
                target_faces = absolute_target_faces or max(1, int(round(source_triangles * ratio)))
                new_obj = duplicate_for_lod(obj, lod_name, collection)
                validation = None
                gate = {"passed": True, "warnings": []}

                if settings.lod_generation_method == "WEDGE_QEM":
                    source = build_wedge_source(obj, angle_limit)
                    if not source["faces"]:
                        action = "skipped no triangulatable faces"
                    elif stats["part_type"] in {"glass"} and settings.skip_glass:
                        action = "skipped glass"
                    elif stats["part_type"] in {"tiny_detail"} and settings.delete_tiny_details:
                        bpy.data.objects.remove(new_obj, do_unlink=True)
                        continue
                    else:
                        reducer = wedge_backend.simplify_wedge_mesh_partitioned if settings.wedge_partition_islands else wedge_backend.simplify_wedge_mesh
                        reduced = reducer(
                            source["vertices"],
                            source["uvs"],
                            source["faces"],
                            face_materials=source["face_materials"],
                            vertex_domains=source["vertex_domains"],
                            group_weights=source["group_weights"],
                            target_ratio=ratio,
                            uv_weight=settings.qem_uv_weight,
                            uv_distance_limit=settings.qem_uv_distance_limit,
                            lock_border_vertices=settings.wedge_lock_borders,
                            allow_domain_crossing=False,
                            safe_weld=True,
                            reject_face_flips=settings.wedge_reject_face_flips,
                            max_iterations=max(settings.wedge_max_iterations, len(source["faces"])),
                            boundary_weight=settings.boundary_weight,
                            preserve_manifold=settings.preserve_manifold,
                            target_faces=target_faces if absolute_target_faces else None,
                        )
                        apply_wedge_result_to_lod_object(
                            new_obj,
                            obj,
                            reduced,
                            write_custom_normals=settings.wedge_write_custom_normals,
                        )
                        validation = compare_payloads(source, reduced)
                        gate = validation_gate(validation, settings)
                        action = "wedge qem reduced"
                else:
                    create_protection_vertex_group(new_obj, angle_limit)
                    action = apply_decimate_modifier(new_obj, settings, ratio, stats["part_type"])
                    if action == "deleted tiny detail":
                        continue

                new_stats = analyse_mesh_object(new_obj, angle_limit)
                results.append({
                    "source": obj.name,
                    "lod": lod_name,
                    "part_type": stats["part_type"],
                    "method": settings.lod_generation_method,
                    "action": action,
                    "before_triangles": source_triangles,
                    "target_faces": target_faces,
                    "target_faces_mode": "absolute" if absolute_target_faces else "ratio",
                    "achieved_faces": new_stats["triangles"],
                    "after_triangles": new_stats["triangles"],
                    "ratio": round(new_stats["triangles"] / max(1, source_triangles), 4),
                    "validation": validation,
                    "validation_warnings": gate.get("warnings", []),
                })

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "operation": "generate_conservative_lods",
            "generation_method": settings.lod_generation_method,
            "results": results,
        }
        out_path = write_report(report)
        self.report({"INFO"}, f"Generated LODs. Report written: {out_path}")
        return {"FINISHED"}


class VLOD_PT_panel(bpy.types.Panel):
    bl_label = "Vehicle Smart LOD"
    bl_idname = "VLOD_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Vehicle LOD"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.vehicle_smart_lod

        layout.operator("vlod.analyse", icon="VIEWZOOM")
        layout.operator("vlod.mark_constraints", icon="GROUP_VERTEX")
        layout.operator("vlod.safe_defaults", icon="TOOL_SETTINGS")
        layout.operator("vlod.safe_test_pack", icon="PACKAGE")
        layout.separator()

        col = layout.column(align=True)
        col.prop(settings, "hard_edge_angle_degrees")
        col.prop(settings, "planar_angle_degrees")
        layout.separator()

        box = layout.box()
        box.label(text="LOD Outputs")
        box.prop(settings, "lod_generation_method")
        box.prop(settings, "create_lod1")
        box.prop(settings, "lod1_ratio")
        box.prop(settings, "lod1_target_triangles")
        box.prop(settings, "create_lod2")
        box.prop(settings, "lod2_ratio")
        box.prop(settings, "lod2_target_triangles")
        box.prop(settings, "create_lod3")
        box.prop(settings, "lod3_ratio")
        box.prop(settings, "lod3_target_triangles")

        layout.separator()
        layout.prop(settings, "skip_glass")
        layout.prop(settings, "delete_tiny_details")
        layout.operator("vlod.generate", icon="MOD_DECIM")

        layout.separator()
        layout.label(text="Experimental")
        layout.prop(settings, "qem_candidate_limit")
        layout.prop(settings, "qem_uv_safe")
        if settings.qem_uv_safe:
            layout.prop(settings, "qem_uv_weight")
            layout.prop(settings, "qem_uv_distance_limit")
        layout.operator("vlod.qem_plan", icon="MOD_VERTEX_WEIGHT")
        layout.prop(settings, "wedge_preview_ratio")
        layout.prop(settings, "wedge_target_triangles")
        layout.prop(settings, "wedge_lock_borders")
        layout.prop(settings, "wedge_partition_islands")
        layout.prop(settings, "wedge_reject_face_flips")
        layout.prop(settings, "preserve_manifold")
        layout.prop(settings, "boundary_weight")
        layout.prop(settings, "wedge_write_custom_normals")
        layout.prop(settings, "max_preview_triangles")
        layout.prop(settings, "wedge_max_iterations")
        layout.prop(settings, "allow_heavy_preview")
        layout.prop(settings, "safe_test_object_limit")
        layout.prop(settings, "create_collision_proxy")
        layout.operator("vlod.wedge_preview", icon="MESH_DATA")
        layout.operator("vlod.backend_selftest", icon="CHECKMARK")

        layout.separator()
        box = layout.box()
        box.label(text="Vehicle Reducers", icon="AUTO")
        box.prop(settings, "radial_target_segments")
        box.prop(settings, "radial_confidence_threshold")
        box.operator("vlod.radial_tyre", icon="MESH_CYLINDER")
        box.separator()
        box.prop(settings, "cull_samples")
        box.prop(settings, "cull_margin")
        box.prop(settings, "cull_max_faces")
        box.operator("vlod.visibility_cull", icon="HIDE_OFF")
        box.separator()
        box.prop(settings, "collision_method")
        if settings.collision_method == "DECOMPOSE":
            box.prop(settings, "collision_max_hulls")
        box.operator("vlod.collision_hull", icon="MESH_ICOSPHERE")


classes = (
    VLOD_Settings,
    VLOD_OT_analyse,
    VLOD_OT_mark_constraints,
    VLOD_OT_qem_plan,
    VLOD_OT_wedge_preview,
    VLOD_OT_radial_tyre,
    VLOD_OT_visibility_cull,
    VLOD_OT_collision_hull,
    VLOD_OT_backend_selftest,
    VLOD_OT_safe_test_pack,
    VLOD_OT_safe_defaults,
    VLOD_OT_generate,
    VLOD_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vehicle_smart_lod = bpy.props.PointerProperty(type=VLOD_Settings)


def unregister():
    if hasattr(bpy.types.Scene, "vehicle_smart_lod"):
        del bpy.types.Scene.vehicle_smart_lod
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
