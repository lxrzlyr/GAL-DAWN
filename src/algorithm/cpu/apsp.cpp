/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn_internal/algorithm/cpu/apsp.hxx>
#include <dawn_internal/tool.hxx>

#include <iostream>

DAWN::RunResult DAWN::APSP_CPU::run(Graph::Graph_t& graph,
                                    std::string& output_path) {
  float elapsed_time = 0.0f;
  int proEntry = 0;
  auto row = graph.rows;

  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;

#pragma omp parallel for reduction(+ : elapsed_time)
  for (int i = 0; i < row; i++) {
    if (graph.csr.row_ptr[i] == graph.csr.row_ptr[i + 1]) {
      ++proEntry;
      DAWN::Tool::infoprint(proEntry, row, graph.interval, graph.thread,
                            elapsed_time);
      continue;
    }
    float time = 0.0f;
    if (graph.weighted) {
      time = DAWN::SSSP_CPU::SSSP_kernel(graph.csr.row_ptr, graph.csr.col,
                                         graph.csr.val, row, i, graph.print,
                                         output_path);
    } else {
      time = DAWN::BFS_CPU::BFS_kernel(graph.csr.row_ptr, graph.csr.col, row, i,
                                       graph.print, output_path);
    }
    elapsed_time += time;
    ++proEntry;
    DAWN::Tool::infoprint(proEntry, row, graph.interval, graph.thread,
                          elapsed_time);
  }

  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;

  elapsed_time = elapsed_time / (graph.thread * 1000);
  return DAWN::RunResult::ok(elapsed_time);
}
