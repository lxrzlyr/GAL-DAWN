#include <dawn/dawn.cuh>

int main(int argc, char* argv[])
{
  DAWN::Graph graph;
  DAWN::GPU   gpurun;

  std::string algo        = argv[1];
  std::string input_path  = argv[2];
  std::string output_path = argv[3];
  graph.thread            = 1;
  graph.stream            = 1;
  graph.block_size        = atoi(argv[4]);
  std::string prinft      = argv[5];
  graph.source            = atoi(argv[6]);
  std::string weighted    = argv[7];

  if (prinft == "true") {
    graph.prinft = true;
    std::cout << "Prinft source " << graph.source << std::endl;
  } else {
    graph.prinft = false;
  }

  if (weighted == "weighted") {
    graph.weighted = true;
    std::cout << "Weighted Graph" << std::endl;
  } else {
    graph.weighted = false;
  }
  graph.createGraph(input_path, graph);
  gpurun.runSsspGpu(graph, output_path);

  return 0;
}
//./dawn_gpu_sssp Default $GRAPH_DIR/XXX.mtx ../output.txt 256
// false 0 unweighted
//./dawn_gpu_sssp Test $GRAPH_DIR/XXX.mtx ../output.txt
// false 0 unweighted