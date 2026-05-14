/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include "legacy_dawn.cuh"

namespace DAWN {
namespace SSSP_GPU {

float kernel(Graph::Graph_t& graph,
             int source,
             cudaStream_t streams,
             thrust::device_vector<int> d_row_ptr,
             thrust::device_vector<int> d_col,
             thrust::device_vector<float> d_val,
             std::string& output_path);

float run(Graph::Graph_t& graph, std::string& output_path);

}  // namespace SSSP_GPU
}  // namespace DAWN

__device__ static float atomicMin(float* address, float value);

__global__ void GOVM(const int* row_ptr,
                     const int* col,
                     const float* val,
                     bool* alpha,
                     bool* beta,
                     float* distance,
                     int rows,
                     int source,
                     bool* ptr);
