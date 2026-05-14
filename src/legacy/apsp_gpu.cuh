/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include "sssp_gpu.cuh"
#include "bfs_gpu.cuh"

namespace DAWN {
namespace APSP_GPU {

float run(Graph::Graph_t& graph, std::string& output_path);

float run_Weighted(Graph::Graph_t& graph, std::string& output_path);

}  // namespace APSP_GPU
}  // namespace DAWN
