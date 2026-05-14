/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include "bfs_gpu.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ void SOVM(const int* row_ptr,
                     const int* col,
                     bool* alpha,
                     bool* beta,
                     int* distance,
                     int rows,
                     int step,
                     bool* ptr) {
  ptr[0] = false;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((j < rows) && (alpha[j])) {
    int start = row_ptr[j];
    int end = row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int index = col[k];
        if (!distance[index]) {
          distance[index] = step;
          beta[index] = true;
          if (!ptr[0])
            ptr[0] = true;
        }
      }
    }
    alpha[j] = false;
  }
}

float DAWN::BFS_GPU::kernel(DAWN::Graph::Graph_t& graph,
                            int source,
                            cudaStream_t streams,
                            thrust::device_vector<int> d_row_ptr,
                            thrust::device_vector<int> d_col,
                            std::string& output_path) {
  int step = 1;

  thrust::host_vector<bool> h_input(graph.rows, false);
  thrust::host_vector<int> h_distance(graph.rows, 0);
  thrust::host_vector<bool> h_ptr(1, false);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    h_input[graph.csr.col[i]] = true;
    h_distance[graph.csr.col[i]] = 1;
  }
  thrust::device_vector<bool> d_alpha(graph.rows, false);
  thrust::device_vector<bool> d_beta(graph.rows, false);
  thrust::device_vector<int> d_distance(graph.rows, 0);
  thrust::device_vector<bool> d_ptr(1, false);

  thrust::copy(h_input.begin(), h_input.end(), d_alpha.begin());
  thrust::copy(h_input.begin(), h_input.end(), d_beta.begin());
  thrust::copy(h_distance.begin(), h_distance.end(), d_distance.begin());

  // Launch kernel
  int block_size = graph.block_size;
  int num_blocks = (graph.cols + block_size - 1) / block_size;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < graph.rows) {
    step++;

    if (!(step % 2)) {
      SOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_alpha.data().get(),
          d_beta.data().get(), d_distance.data().get(), graph.rows, step,
          d_ptr.data().get());
    } else {
      SOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_beta.data().get(),
          d_alpha.data().get(), d_distance.data().get(), graph.rows, step,
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

  if ((graph.print) && (source == graph.source)) {
    thrust::copy(d_distance.begin(), d_distance.end(), h_distance.begin());
    printf("Start print\n");

    DAWN::IO::outfile(graph.rows, h_distance.data(), source, output_path);
  }

  return elapsed_time;
}

float DAWN::BFS_GPU::run(DAWN::Graph::Graph_t& graph,
                         std::string& output_path) {
  int source = graph.source;
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    printf("Source [%d] is isolated node\n", source);
    exit(0);
  }

  thrust::device_vector<int> d_row_ptr(graph.rows + 1, 0);
  thrust::device_vector<int> d_col(graph.nnz, 0);
  thrust::copy_n(graph.csr.row_ptr, graph.rows + 1, d_row_ptr.begin());
  thrust::copy_n(graph.csr.col, graph.nnz, d_col.begin());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float elapsed_time = 0.0f;
  elapsed_time +=
      kernel(graph, source, stream, d_row_ptr, d_col, output_path) / 1000;

  cudaStreamDestroy(stream);
  return elapsed_time;
}
