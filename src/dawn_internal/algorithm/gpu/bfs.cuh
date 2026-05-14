/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <dawn_internal/dawn.cuh>

namespace DAWN {
namespace BFS_GPU {

float run(Graph::Graph_t& graph, std::string& output_path);

float kernel(Graph::Graph_t& graph,
             int source,
             cudaStream_t streams,
             thrust::device_vector<int> d_row_ptr,
             thrust::device_vector<int> d_col,
             std::string& output_path);

}  // namespace BFS_GPU
}  // namespace DAWN

__global__ void SOVM(const int* row_ptr,
                     const int* col,
                     bool* alpha,
                     bool* beta,
                     int* distance,
                     int rows,
                     int step,
                     bool* ptr);
