// #include <ATen/ATen.h>
#include <torch/extension.h>

// Kernel definition
extern "C" __global__
void square(float *d_out, float *d_in){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}

// Host function
void square_cuda(float *out, float *in, int size) {
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch kernel
    square<<<numBlocks, blockSize>>>(out, in);

    // Synchronize
    cudaDeviceSynchronize();
}


__device__ float3 operator*(float scalar, float3 vec) {
    return make_float3(scalar * vec.x, scalar * vec.y, scalar * vec.z);
}

__device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length(float3 a) {
    return sqrt(dot(a, a));
}

__device__ float point_edge_distance(float3 point, float3 v0, float3 v1) {
    float3 edge = v1 - v0;
    float3 diff = point - v0;

    float t = dot(diff, edge) / dot(edge, edge);

    if (t < 0) return length(diff);
    if (t > 1) {
        diff = point - v1;
        return length(diff);
    }

    float3 proj = v0 + t*edge;
    diff = point - proj;
    return length(diff);
}


__device__ bool point_inside_triangle(float3 point, float3 v0, float3 v1, float3 v2) {
    float3 edge0 = v1 - v0;
    float3 edge1 = v2 - v1;
    float3 edge2 = v0 - v2;

    float3 C0 = point - v0;
    float3 C1 = point - v1;
    float3 C2 = point - v2;

    float3 cross0 = cross(edge0, C0);
    float3 cross1 = cross(edge1, C1);
    float3 cross2 = cross(edge2, C2);

    return dot(cross0, cross1) >= 0 && dot(cross0, cross2) >= 0;
}

__device__ float point_triangle_distance(float3 point, float3 v0, float3 v1, float3 v2) {
    // Compute triangle edges and normal
    float3 edge0 = v1 - v0;
    float3 edge1 = v2 - v0;
    float3 n = cross(edge0, edge1);

    // Project point onto the plane defined by the triangle
    float t = dot(point - v0, n) / dot(n, n);
    float3 proj = point - t*n;

    // Check if the projected point lies inside the triangle
    bool inside = point_inside_triangle(proj, v0, v1, v2);

    // If the projected point lies inside the triangle, return the distance to the plane
    if (inside) {
        return fabs(t) * length(n);
    }

    // If the projected point lies outside the triangle, return the minimum distance to the edges
    float dist_edge0 = point_edge_distance(point, v0, v1);
    float dist_edge1 = point_edge_distance(point, v1, v2);
    float dist_edge2 = point_edge_distance(point, v2, v0);

    return fminf(dist_edge0, fminf(dist_edge1, dist_edge2));
}


__global__ void point_mesh_kernel(float* out_dist, int* out_idx, float3* points, int num_points,
                                  float3* mesh, int num_triangles) {
    // check thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) {
        return;
    }
    // get point
    float3 point = points[tid];
    // printf("!! %f %f %f\n",  point.x, point.y, point.z);
    float min_distance = FLT_MAX;
    int min_idx = -1;
    // iterate through all triangles
    for (int i = 0; i < num_triangles; ++i) {
        float3 a = mesh[i*3+0];
        float3 b = mesh[i*3+1];
        float3 c = mesh[i*3+2];
        // Compute distance from point to triangle
        float distance = point_triangle_distance(point, a, b, c);
        // compare
        if (distance < min_distance) {
            min_distance = distance;
            min_idx = i;
        }
    }
    // return result
    out_dist[tid] = min_distance;
    out_idx[tid] = min_idx;
}


void point_mesh_cuda(torch::Tensor dist, torch::Tensor out, torch::Tensor points, torch::Tensor mesh) {
    int num_points = points.size(0);
    int num_triangles = mesh.size(0);

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;

    // test data
    std::cerr << "num_points: " << num_points << "\n";
    std::cerr << "num_triangles: " << num_triangles << "\n";
    std::cerr << "sizeof(float3): " << sizeof(float3) << "\n";
    
    point_mesh_kernel<<<numBlocks, blockSize>>>(
        dist.data_ptr<float>(),
        out.data_ptr<int>(),
        reinterpret_cast<float3*>(points.data_ptr<float>()),
        num_points,
        reinterpret_cast<float3*>(mesh.data_ptr<float>()),
        num_triangles);

    cudaDeviceSynchronize();
}
