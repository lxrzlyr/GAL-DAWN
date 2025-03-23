/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/sssp.hxx>
#include <dawn/algorithm/cpu/bfs.hxx>

namespace DAWN {
namespace BC_CPU {

float Betweenness_Centrality(Graph::Graph_t& graph, std::string& output_path);

float Betweenness_Centrality_run(int* row_ptr,
                                 int* col,
                                 int node,
                                 int n,
                                 float*& bc);

bool kernel(int* row_ptr,
            int* col,
            int row,
            std::vector<bool>& alpha,
            std::vector<bool>& beta,
            std::vector<int>& distance,
            std::vector<std::vector<int>>& Pretree,
            std::vector<float>& visit,
            std::vector<int>& prequeue,
            int& index,
            int step);
}  // namespace BC_CPU
}  // namespace DAWN