/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <dawn_internal/algorithm/gpu/sssp.cuh>
#include <dawn_internal/algorithm/gpu/bfs.cuh>

namespace DAWN {
namespace CC_GPU {

// Centrality Algorithm
float kernel(Graph::Graph_t& graph,
             int source,
             cudaStream_t streams,
             thrust::device_vector<int> d_row_ptr,
             thrust::device_vector<int> d_col);

float kernel_Weighted(Graph::Graph_t& graph,
                      int source,
                      cudaStream_t streams,
                      thrust::device_vector<int> d_row_ptr,
                      thrust::device_vector<int> d_col,
                      thrust::device_vector<float> d_val);

float run(Graph::Graph_t& graph, int source);

float run_Weighted(Graph::Graph_t& graph, int source);

}  // namespace CC_GPU
}  // namespace DAWN
