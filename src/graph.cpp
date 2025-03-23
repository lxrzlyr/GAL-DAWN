/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/graph.hxx>
#include <dawn/io.hxx>
void DAWN::Graph::createGraph(std::string& input_path, Graph::Graph_t& graph) {
  std::ifstream file(input_path);
  if (!file.is_open()) {
    std::cerr << "Error opening file " << input_path << std::endl;
    return;
  }

  std::string line;
  int rows, cols, nnz;

  std::getline(file, line);
  std::stringstream ss(line);
  bool tmp_weight = false;
  std::string format, object, matrixtype, datatype, direct;
  ss >> format >> object >> matrixtype >> datatype >> direct;
  if ((format == "%%MatrixMarket") && (matrixtype == "coordinate")) {
    if (datatype == "pattern") {
      tmp_weight = false;
      if (graph.weighted != tmp_weight) {
        std::cout << "This is an unweighted graph, but it has been instructed "
                     "to use weighted functions for computation."
                  << std::endl;
      }
    } else {
      tmp_weight = true;
      if (graph.weighted != tmp_weight) {
        std::cout << "This is a weighted graph, and it has been instructed "
                     "to use unweighted functions for computation. "
                  << std::endl;
      }
    }
    if (direct == "symmetric") {
      graph.directed = false;
    } else {
      graph.directed = true;
    }
  } else {
    std::cout << "invalid file" << std::endl;
    return;
  }

  while (std::getline(file, line)) {
    if (line[0] == '%')
      continue;
    std::stringstream ss(line);
    ss >> rows >> cols >> nnz;
    break;
  }
  file.close();

  graph.rows = rows;
  graph.cols = cols;
  graph.nnz = nnz;
  graph.source = graph.source % rows;

  std::cout << "Read Input Graph" << std::endl;

  if (graph.directed) {
    if (graph.weighted) {
      DAWN::IO::readGraph_Directed_Weighted(input_path, graph);
      // std::cout << "readGraph_Directed_Weighted" << std::endl;
    } else {
      // std::cout << "readGraph_Directed" << std::endl;
      DAWN::IO::readGraph_Directed(input_path, graph);
    }
  } else {
    if (graph.weighted) {
      DAWN::IO::readGraph_Weighted(input_path, graph);
      // std::cout << "readGraph_Weighted" << std::endl;
    } else {
      DAWN::IO::readGraph(input_path, graph);
      // std::cout << "readGraph" << std::endl;
    }
  }
}
