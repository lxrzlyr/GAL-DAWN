/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <dawn_internal/algorithm/cpu/sssp.hxx>
#include <dawn_internal/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace MSSP_CPU {

// Shortest Path Algorithm
DAWN::RunResult run(Graph::Graph_t& graph, std::string& output_path);

}  // namespace MSSP_CPU
}  // namespace DAWN
