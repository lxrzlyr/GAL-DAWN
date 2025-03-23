/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bc.hxx>

int main(int argc, char* argv[]) {
  DAWN::IO::parameters_t params = DAWN::IO::parameters(argc, argv);
  DAWN::Graph::Graph_t graph;

  graph.thread = omp_get_num_threads();
  graph.print = true;
  graph.interval = 100;

  graph.weighted = false;
  DAWN::Graph::createGraph(params.input_path, graph);
  printf("%-21s%3.5d\n", "Nodes:", graph.rows);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  float elapsed_time =
      DAWN::BC_CPU::Betweenness_Centrality(graph, params.output_path);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return 0;
}
