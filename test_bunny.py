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

# torch.manual_seed(42)
# points = torch.rand((30, 3)).to(device)
# faces = torch.tensor([[[0.1, 0.3, 0.6], [0.4, 0.2, 0.1], [0.5, 0.7, 0.3]]]).to(device)


start_time = time.time()
dist, prim_id = point_mesh(
    points,
    faces,
)
print(prim_id.sum())
end_time = time.time()
print(f"Time taken by the function: {end_time - start_time} seconds")
print(points)
print((dist*dist))
print(prim_id)

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