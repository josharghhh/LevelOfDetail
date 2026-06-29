import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wedge_backend


def main():
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
    stats = result["stats"]
    assert 1 <= stats["end_faces"] <= 4, stats
    assert len(result["vertices"]) == len(result["uvs"]) == len(result["group_weights"])
    print("SMOKE TEST PASSED")
    print(stats)


if __name__ == "__main__":
    main()
