// Smoke test for the legacy CUDA BFS kernel: runs it on a 5x5 grid and
// compares every distance with the Manhattan distance from the source.
// The grid fits in one warp on purpose: the kernel's stop flag is written by
// all threads without synchronisation, so a multi-block launch can stop early.
#include "legacy/bfs_gpu.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
    std::fprintf(stderr, "FAIL: no CUDA device found\n");
    return 1;
  }

  const int side = 5;
  const int n = side * side;
  std::vector<int> row_ptr(n + 1, 0);
  std::vector<int> col;
  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      const int v = y * side + x;
      if (x > 0) col.push_back(v - 1);
      if (x < side - 1) col.push_back(v + 1);
      if (y > 0) col.push_back(v - side);
      if (y < side - 1) col.push_back(v + side);
      row_ptr[v + 1] = static_cast<int>(col.size());
    }
  }

  DAWN::Graph::Graph_t graph;
  graph.rows = n;
  graph.cols = n;
  graph.nnz = col.size();
  graph.csr.row_ptr = row_ptr.data();
  graph.csr.col = col.data();
  graph.csr.val = nullptr;
  graph.thread = 1;
  graph.interval = 1;
  graph.stream = 1;
  graph.block_size = 32;
  graph.source = 0;
  graph.print = true;
  graph.weighted = false;
  graph.directed = false;

  std::string out_path = "cuda_bfs_smoke.out";
  std::remove(out_path.c_str());
  DAWN::BFS_GPU::run(graph, out_path);

  std::vector<int> dist(n, -1);
  std::ifstream in(out_path);
  int src = 0, v = 0, d = 0;
  while (in >> src >> v >> d) {
    if (src == 0 && v >= 0 && v < n) dist[v] = d;
  }
  int bad = 0;
  for (int i = 1; i < n; ++i) {
    const int expect = (i / side) + (i % side);
    if (dist[i] != expect) {
      std::fprintf(stderr, "node %d: got %d, expected %d\n", i, dist[i],
                   expect);
      ++bad;
    }
  }
  std::remove(out_path.c_str());
  if (bad) {
    std::fprintf(stderr, "FAIL: %d wrong distances\n", bad);
    return 1;
  }
  std::printf("PASS: CUDA BFS matched on %d nodes (%d device(s))\n", n,
              device_count);
  return 0;
}
