#include <torch/extension.h>

// CUDA forward declaration
void square_cuda(float *out, float *in, int size);

// CUDA forward declaration
void point_mesh_cuda(torch::Tensor dist, torch::Tensor idx, torch::Tensor points, torch::Tensor mesh);


// C++ interface
// point_mesh takes 2 parameters:
// - points, a n*3 tensor of float representing n 3D points
// - mesh, a m*3*3 tensor of float representing m triangles in 3D space
// returns a tensor of size n, representing the index of triangle
// in the mesh that's closest to each queried point.
std::tuple<torch::Tensor, torch::Tensor> point_mesh(torch::Tensor points, torch::Tensor mesh) {
    // Check inputs
    TORCH_CHECK(points.is_cuda(), "Points tensor must be a CUDA tensor");
    TORCH_CHECK(mesh.is_cuda(), "Mesh tensor must be a CUDA tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "Points tensor must be of size [n, 3]");
    TORCH_CHECK(mesh.dim() == 3 && mesh.size(1) == 3 && mesh.size(2) == 3, "Mesh tensor must be of size [m, 3, 3]");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be a float tensor");
    TORCH_CHECK(mesh.dtype() == torch::kFloat32, "mesh must be a float tensor");
    // create output
    torch::Tensor dist = torch::empty({points.size(0)}, points.options().dtype(torch::kFloat32));
    torch::Tensor idx = torch::empty({points.size(0)}, points.options().dtype(torch::kInt32));
    point_mesh_cuda(dist, idx, points, mesh);
    return std::make_tuple(dist, idx);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("point_mesh", &point_mesh, "point_mesh (CUDA)");
}
