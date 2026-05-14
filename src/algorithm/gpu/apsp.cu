/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn_internal/algorithm/gpu/apsp.cuh>

#include <iostream>

float DAWN::APSP_GPU::run_Weighted(DAWN::Graph::Graph_t& graph,
                                   std::string& output_path) {
  float elapsed_time = 0.0;
  int proEntry = 0;

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::device_vector<float> d_val(graph.nnz, 0);
  thrust::copy_n(graph.csr.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csr.col, graph.nnz, d_col.begin());
  thrust::copy_n(graph.csr.val, graph.nnz, d_val.begin());

  // Create streams
  cudaStream_t streams[graph.stream];
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamCreate(&streams[i]);
  }

  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  for (int i = 0; i < graph.rows; i++) {
    int source = i;
    if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
      ++proEntry;
      // printf("Source [%d] is isolated node\n", source);
      DAWN::Tool::infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                            elapsed_time);
      continue;
    }
    int cuda_stream = source % graph.stream;

    elapsed_time +=
        DAWN::SSSP_GPU::kernel(graph, source, streams[cuda_stream], d_row_ptr,
                               d_col, d_val, output_path);

    ++proEntry;
    DAWN::Tool::infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                          elapsed_time);
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;

  elapsed_time = elapsed_time / (graph.thread * 1000);
  // Synchronize streams
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  return elapsed_time;
}

float DAWN::APSP_GPU::run(DAWN::Graph::Graph_t& graph,
                          std::string& output_path) {
  float elapsed_time = 0.0;
  int proEntry = 0;

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::copy_n(graph.csr.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csr.col, graph.nnz, d_col.begin());

  // Create streams
  cudaStream_t streams[graph.stream];
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamCreate(&streams[i]);
  }

  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP start <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;
  for (int i = 0; i < graph.rows; i++) {
    int source = i;
    if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
      ++proEntry;
      // printf("Source [%d] is isolated node\n", source);
      DAWN::Tool::infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                            elapsed_time);
      continue;
    }
    int cuda_stream = source % graph.stream;

    elapsed_time += DAWN::BFS_GPU::kernel(graph, source, streams[cuda_stream],
                                          d_row_ptr, d_col, output_path);

    ++proEntry;
    DAWN::Tool::infoprint(proEntry, graph.rows, graph.interval, graph.thread,
                          elapsed_time);
  }
  std::cout
      << ">>>>>>>>>>>>>>>>>>>>>>>>>>> APSP end <<<<<<<<<<<<<<<<<<<<<<<<<<<"
      << std::endl;

  elapsed_time = elapsed_time / (graph.thread * 1000);

  // Synchronize streams
  for (int i = 0; i < graph.stream; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  return elapsed_time;
}
