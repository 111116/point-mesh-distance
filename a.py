import torch
from point_mesh import point_mesh
import time

pts = torch.tensor([[1,2,3.0],[4,5,6.0]]).cuda()
trig = torch.tensor([[[0.0,0,0], [0,0,1], [0,1,0]], [[1,6,0],[0,6,1],[0,7,0]]]).cuda()

start_time = time.time()
res = point_mesh(pts, trig)
end_time = time.time()

print(res)

print(f"time: {round(end_time - start_time, 4)}s")
