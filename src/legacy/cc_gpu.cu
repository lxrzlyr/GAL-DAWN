/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include "cc_gpu.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>

float DAWN::CC_GPU::run(DAWN::Graph::Graph_t& graph, int source) {
  auto row = graph.rows;
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    printf("Source [%d] is isolated node\n", source);
    exit(0);
  }

  thrust::device_vector<int> d_row_ptr(row + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::copy_n(graph.csr.row_ptr, row + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csr.col, graph.nnz, d_col.begin());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float closeness_centrality =
      DAWN::CC_GPU::kernel(graph, source, stream, d_row_ptr, d_col);

  cudaStreamDestroy(stream);

  return closeness_centrality;
}

float DAWN::CC_GPU::kernel(DAWN::Graph::Graph_t& graph,
                           int source,
                           cudaStream_t streams,
                           thrust::device_vector<int> d_row_ptr,
                           thrust::device_vector<int> d_col) {
  int step = 1;
  auto row = graph.rows;
  thrust::host_vector<bool> h_input(row, false);
  thrust::host_vector<int> h_distance(row, 0);
  thrust::host_vector<bool> h_ptr(1, false);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    h_input[graph.csr.col[i]] = true;
    h_distance[graph.csr.col[i]] = 1;
  }
  thrust::device_vector<bool> d_alpha(row, false);
  thrust::device_vector<bool> d_beta(row, false);
  thrust::device_vector<int> d_distance(row, 0);
  thrust::device_vector<bool> d_ptr(1, false);

  thrust::copy(h_input.begin(), h_input.end(), d_alpha.begin());
  thrust::copy(h_input.begin(), h_input.end(), d_beta.begin());
  thrust::copy(h_distance.begin(), h_distance.end(), d_distance.begin());

  // Launch kernel
  int block_size = graph.block_size;
  int num_blocks = (graph.cols + block_size - 1) / block_size;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;

    if (!(step % 2)) {
      SOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_alpha.data().get(),
          d_beta.data().get(), d_distance.data().get(), row, step,
          d_ptr.data().get());
    } else {
      SOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_beta.data().get(),
          d_alpha.data().get(), d_distance.data().get(), row, step,
          d_ptr.data().get());
    }

    if (!(step % 3)) {
      bool ptr = d_ptr[0];
      if (!ptr) {
        break;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  elapsed_time += elapsed.count();

  thrust::fill(d_distance.begin() + source, d_distance.begin() + source + 1, 0);
  float closeness_centrality =
      (row - 1) / thrust::reduce(d_distance.begin(), d_distance.end(), 0,
                                 thrust::plus<int>());

  elapsed_time = elapsed_time / (graph.thread * 1000);

  printf("%-21s%3.5d\n", "Node:", row);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return closeness_centrality;
}

float DAWN::CC_GPU::run_Weighted(DAWN::Graph::Graph_t& graph, int source) {
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    printf("Source [%d] is isolated node\n", source);
    exit(0);
  }

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::device_vector<float> d_val(graph.nnz, 0);
  thrust::copy_n(graph.csr.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csr.col, graph.nnz, d_col.begin());
  thrust::copy_n(graph.csr.val, graph.nnz, d_val.begin());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float closeness_centrality = DAWN::CC_GPU::kernel_Weighted(
      graph, source, stream, d_row_ptr, d_col, d_val);

  cudaStreamDestroy(stream);
  return closeness_centrality;
}

float DAWN::CC_GPU::kernel_Weighted(DAWN::Graph::Graph_t& graph,
                                    int source,
                                    cudaStream_t streams,
                                    thrust::device_vector<int> d_row_ptr,
                                    thrust::device_vector<int> d_col,
                                    thrust::device_vector<float> d_val) {
  int step = 1;
  auto row = graph.rows;
  float INF = std::numeric_limits<float>::max();

  thrust::host_vector<bool> h_alpha(row, 0);
  thrust::host_vector<bool> h_beta(row, 0);
  thrust::host_vector<float> h_distance(row, INF);
  thrust::host_vector<bool> h_ptr(1, false);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    h_alpha[graph.csr.col[i]] = true;
    h_distance[graph.csr.col[i]] = graph.csr.val[i];
  }
  thrust::device_vector<bool> d_alpha(row, 0);
  thrust::device_vector<bool> d_beta(row, 0);
  thrust::device_vector<float> d_distance(row, INF);
  thrust::device_vector<bool> d_ptr(1, false);

  thrust::copy(h_alpha.begin(), h_alpha.end(), d_alpha.begin());
  thrust::copy(h_distance.begin(), h_distance.end(), d_distance.begin());

  // Launch kernel
  int block_size = graph.block_size;
  int num_blocks = (graph.cols + block_size - 1) / block_size;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2)) {
      GOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_val.data().get(),
          d_alpha.data().get(), d_beta.data().get(), d_distance.data().get(),
          row, source, d_ptr.data().get());
      thrust::fill_n(d_alpha.begin(), row, false);
    } else {
      GOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_val.data().get(),
          d_beta.data().get(), d_alpha.data().get(), d_distance.data().get(),
          row, source, d_ptr.data().get());
      thrust::fill_n(d_beta.begin(), row, false);
    }
    if (!(step % 5)) {
      bool ptr = d_ptr[0];
      if (!ptr) {
        break;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  elapsed_time += elapsed.count();

  thrust::fill(d_distance.begin() + source, d_distance.begin() + source + 1,
               0.0f);
  float closeness_centrality =
      (1.0 * row - 1.0f) / thrust::reduce(d_distance.begin(), d_distance.end(),
                                          0.0f, thrust::plus<float>());

  elapsed_time = elapsed_time / (graph.thread * 1000);
  printf("%-21s%3.5d\n", "Node:", row);
  printf("%-21s%3.5ld\n", "Edges:", graph.nnz);
  printf("%-21s%3.5lf\n", "Time:", elapsed_time);

  return closeness_centrality;
}
