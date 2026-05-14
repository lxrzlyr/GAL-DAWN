/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn_internal/algorithm/cpu/sssp.hxx>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>

DAWN::RunResult DAWN::SSSP_CPU::run(Graph::Graph_t& graph,
                                    std::string& output_path) {
  int source = graph.source;
  auto row = graph.rows;
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    return DAWN::RunResult::err("Source is isolated node, please check");
  }
  float elapsed_time =
      DAWN::SSSP_CPU::SSSP(graph.csr.row_ptr, graph.csr.col, graph.csr.val, row,
                           source, graph.print, output_path) /
      1000;
  return DAWN::RunResult::ok(elapsed_time);
}

float DAWN::SSSP_CPU::SSSP(int* row_ptr,
                           int* col,
                           float* val,
                           int row,
                           int source,
                           bool print,
                           std::string& output_path) {
  int step = 1;
  bool is_converged = false;
  bool* alpha = new bool[row];
  bool* beta = new bool[row];
  float* distance = new float[row];
  float elapsed_time = 0.0f;
  float INF = std::numeric_limits<float>::max();

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, INF);

#pragma omp parallel for
  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    distance[col[i]] = val[i];
    alpha[col[i]] = true;
  }
  distance[source] = 0.0f;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      is_converged = DAWN::SSSP_CPU::GOVMP(row_ptr, col, val, row, alpha, beta,
                                           distance, source);
    else
      is_converged = DAWN::SSSP_CPU::GOVMP(row_ptr, col, val, row, beta, alpha,
                                           distance, source);
    if (is_converged) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_time_tmp = end - start;
  elapsed_time += elapsed_time_tmp.count();

  // Output
  if (print) {
    printf("Start print\n");
    DAWN::IO::outfile(row, distance, source, output_path);
  }

  delete[] beta;
  beta = nullptr;
  delete[] alpha;
  alpha = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed_time;
}

float DAWN::SSSP_CPU::SSSP_kernel(int* row_ptr,
                                  int* col,
                                  float* val,
                                  int row,
                                  int source,
                                  bool print,
                                  std::string& output_path) {
  int step = 1;
  int entry = row_ptr[source + 1] - row_ptr[source];
  int* alpha = new int[row];
  int* beta = new int[row];
  float* distance = new float[row];
  float elapsed_time = 0.0f;
  float INF = std::numeric_limits<float>::max();

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, INF);

  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    distance[col[i]] = val[i];
    alpha[i - row_ptr[source]] = col[i];
  }

  distance[source] = 0.0f;

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      entry = DAWN::SSSP_CPU::GOVM(row_ptr, col, val, row, alpha, beta,
                                   distance, source, entry);
    else
      entry = DAWN::SSSP_CPU::GOVM(row_ptr, col, val, row, beta, alpha,
                                   distance, source, entry);
    if (!entry) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_time_tmp = end - start;
  elapsed_time += elapsed_time_tmp.count();

  // Output
  if (print) {
    printf("Start print\n");
    DAWN::IO::outfile(row, distance, source, output_path);
  }

  delete[] alpha;
  alpha = nullptr;
  delete[] beta;
  beta = nullptr;
  delete[] distance;
  distance = nullptr;

  return elapsed_time;
}

int DAWN::SSSP_CPU::GOVM(int* row_ptr,
                         int* col,
                         float* val,
                         int row,
                         int*& alpha,
                         int*& beta,
                         float*& distance,
                         int source,
                         int entry) {
  (void)row;
  int tmpEntry = 0;
  for (int j = 0; j < entry; j++) {
    int current = alpha[j];
    int start = row_ptr[current];
    int end = row_ptr[current + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        int index = col[k];
        float tmp = distance[current] + val[k];
        if ((distance[index] > tmp) && (index != source)) {
          distance[index] = std::min(distance[index], tmp);
          beta[tmpEntry] = index;
          ++tmpEntry;
        }
      }
    }
  }
  return tmpEntry;
}

bool DAWN::SSSP_CPU::GOVMP(int* row_ptr,
                           int* col,
                           float* val,
                           int row,
                           bool*& alpha,
                           bool*& beta,
                           float*& distance,
                           int source) {
  bool converged = true;
#pragma omp parallel for
  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = row_ptr[j];
      int end = row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          int index = col[k];
          float tmp = distance[j] + val[k];
          if ((distance[index] > tmp) && (index != source)) {
            distance[index] = std::min(distance[index], tmp);
            beta[index] = true;
            if (converged)
              converged = false;
          }
        }
      }
      alpha[j] = false;
    }
  }
  return converged;
}
