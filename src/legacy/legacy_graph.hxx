/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include "legacy_matrix.hxx"

#include <cstdint>
#include <vector>

namespace DAWN {
namespace Graph {

class Graph_t {
 public:
  int rows;
  int cols;
  uint64_t nnz;
  DAWN::Matrix::Csr_t csr;
  DAWN::Matrix::Coo_t coo;
  int thread;
  int interval;
  int stream;
  int block_size;
  int source;
  bool print;
  bool weighted;
  bool directed;
  std::vector<int> msource;
};

}  // namespace Graph
}  // namespace DAWN
