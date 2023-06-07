# import pytorch3d
import torch
# from pytorch3d.io import load_objs_as_meshes
# from pytorch3d.structures import Meshes, Pointclouds
import trimesh
import numpy as np
import time
from point_mesh import point_mesh # custom


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()

# Load the bunny mesh.
# bunny_mesh = load_objs_as_meshes(["bunny.obj"]).to(device)
bunny_mesh = trimesh.load_mesh('bunny.obj')
vertices = torch.from_numpy(np.array(bunny_mesh.vertices)).float()
faces = torch.from_numpy(np.array(bunny_mesh.triangles)).float()

# Generate points within the range of the mesh.
min_x, min_y, min_z = vertices.min(0)[0]
max_x, max_y, max_z = vertices.max(0)[0]


torch.manual_seed(42)
points = torch.rand((2000000, 3)).to(device) * \
         torch.tensor((max_x - min_x, max_y - min_y, max_z - min_z)).to(device) + \
         torch.tensor((min_x, min_y, min_z)).to(device)

faces = faces.to(device)


start_time = time.time()
prim_id = point_mesh(
    points,
    faces,
)
print(prim_id.sum())
end_time = time.time()
print(f"Time taken by the function: {end_time - start_time} seconds")
print(points[0:10])
print(prim_id[0:30])

# start_time = time.time()
# sqrdist, prim_id = pytorch3d._C.point_face_dist_forward(
#     points_packed,
#     points_first_idx,
#     faces_packed,
#     faces_first_idx,
#     max_p,
#     1e-4
# )
# print(sqrdist.mean())
# end_time = time.time()
# print(f"Time taken by the function: {end_time - start_time} seconds")