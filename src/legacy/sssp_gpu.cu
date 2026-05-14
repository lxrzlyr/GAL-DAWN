/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include "sssp_gpu.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>

/**
 * @brief CUDA natively doesn't support atomicMin on float based addresses and
 * values. This is a workaround (as of CUDA 11.1, there've been no support).
 * (This function has been copy from Gunrock)
 *
 * @param address
 * @param value
 * @return float
 */
__device__ static float atomicMin(float* address, float value) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = ::atomicCAS(addr_as_int, expected,
                      __float_as_int(::fminf(value, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

__global__ void GOVM(const int* row_ptr,
                     const int* col,
                     const float* val,
                     bool* alpha,
                     bool* beta,
                     float* distance,
                     int rows,
                     int source,
                     bool* ptr) {
  ptr[0] = false;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if ((j < rows) && (alpha[j])) {
    int start = row_ptr[j];
    int end = row_ptr[j + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int index = col[k];
        float tmp = distance[j] + val[k];
        float old_distance = atomicMin(&distance[index], tmp);
        if (old_distance > tmp) {
          beta[index] = true;
          if ((!ptr[0]) && (index != source))
            ptr[0] = true;
        }
      }
    }
  }
}

float DAWN::SSSP_GPU::kernel(DAWN::Graph::Graph_t& graph,
                             int source,
                             cudaStream_t streams,
                             thrust::device_vector<int> d_row_ptr,
                             thrust::device_vector<int> d_col,
                             thrust::device_vector<float> d_val,
                             std::string& output_path) {
  int step = 1;
  float INF = std::numeric_limits<float>::max();

  thrust::host_vector<bool> h_alpha(graph.rows, 0);
  thrust::host_vector<bool> h_beta(graph.rows, 0);
  thrust::host_vector<float> h_distance(graph.rows, INF);
  thrust::host_vector<bool> h_ptr(1, false);

  float elapsed_time = 0.0f;
  omp_set_dynamic(1);
#pragma omp parallel for
  for (int i = graph.csr.row_ptr[source]; i < graph.csr.row_ptr[source + 1];
       i++) {
    h_alpha[graph.csr.col[i]] = true;
    h_distance[graph.csr.col[i]] = graph.csr.val[i];
  }
  thrust::device_vector<bool> d_alpha(graph.rows, 0);
  thrust::device_vector<bool> d_beta(graph.rows, 0);
  thrust::device_vector<float> d_distance(graph.rows, INF);
  thrust::device_vector<bool> d_ptr(1, false);

  thrust::copy(h_alpha.begin(), h_alpha.end(), d_alpha.begin());
  thrust::copy(h_distance.begin(), h_distance.end(), d_distance.begin());

  // Launch kernel
  int block_size = graph.block_size;
  int num_blocks = (graph.cols + block_size - 1) / block_size;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < graph.rows) {
    step++;
    if (!(step % 2)) {
      GOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_val.data().get(),
          d_alpha.data().get(), d_beta.data().get(), d_distance.data().get(),
          graph.rows, source, d_ptr.data().get());
      thrust::fill_n(d_alpha.begin(), graph.rows, false);
    } else {
      GOVM<<<num_blocks, block_size, 0, streams>>>(
          d_row_ptr.data().get(), d_col.data().get(), d_val.data().get(),
          d_beta.data().get(), d_alpha.data().get(), d_distance.data().get(),
          graph.rows, source, d_ptr.data().get());
      thrust::fill_n(d_beta.begin(), graph.rows, false);
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

  // Output
  if ((graph.print) && (source == graph.source)) {
    thrust::copy(d_distance.begin(), d_distance.end(), h_distance.begin());
    printf("Start print\n");

    DAWN::IO::outfile(graph.rows, h_distance.data(), source, output_path);
  }
  return elapsed_time;
}

float DAWN::SSSP_GPU::run(DAWN::Graph::Graph_t& graph,
                          std::string& output_path) {
  int source = graph.source;
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

  float elapsed_time = 0.0f;
  elapsed_time +=
      kernel(graph, source, stream, d_row_ptr, d_col, d_val, output_path) /
      1000;

  cudaStreamDestroy(stream);

  return elapsed_time;
}
