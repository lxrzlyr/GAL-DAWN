/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn_internal/algorithm/cpu/mssp.hxx>

DAWN::RunResult DAWN::MSSP_CPU::run(Graph::Graph_t& graph,
                                    std::string& output_path) {
  float elapsed_time = 0.0f;
  auto row = graph.rows;

#pragma omp parallel for reduction(+ : elapsed_time)
  for (int i = 0; i < static_cast<int>(graph.msource.size()); i++) {
    int source = graph.msource[static_cast<size_t>(i)] % row;
    if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
      continue;
    }
    float time = 0.0f;
    if (graph.weighted) {
      time = DAWN::SSSP_CPU::SSSP_kernel(graph.csr.row_ptr, graph.csr.col,
                                         graph.csr.val, row, source,
                                         graph.print, output_path);
    } else {
      time = DAWN::BFS_CPU::BFS_kernel(graph.csr.row_ptr, graph.csr.col, row,
                                       source, graph.print, output_path);
    }
    elapsed_time += time;
  }
  elapsed_time = elapsed_time / (graph.thread * 1000);
  return DAWN::RunResult::ok(elapsed_time);
}
