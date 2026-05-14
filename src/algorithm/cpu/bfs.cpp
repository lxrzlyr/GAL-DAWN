/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn_internal/algorithm/cpu/bfs.hxx>

#include <algorithm>
#include <chrono>
#include <cstdio>

DAWN::RunResult DAWN::BFS_CPU::run(Graph::Graph_t& graph,
                                   std::string& output_path) {
  int source = graph.source;
  auto row = graph.rows;
  if (graph.csr.row_ptr[source] == graph.csr.row_ptr[source + 1]) {
    return DAWN::RunResult::err("Source is isolated node, please check");
  }
  float elapsed_time = DAWN::BFS_CPU::BFS(graph.csr.row_ptr, graph.csr.col, row,
                                          source, graph.print, output_path) /
                       1000;
  return DAWN::RunResult::ok(elapsed_time);
}

// kernel
float DAWN::BFS_CPU::BFS(int* row_ptr,
                         int* col,
                         int row,
                         int source,
                         bool print,
                         std::string& output_path) {
  int step = 1;
  bool is_converged = false;
  bool* alpha = new bool[row];
  bool* beta = new bool[row];
  int* distance = new int[row];
  float elapsed_time = 0.0f;

  std::fill_n(alpha, row, false);
  std::fill_n(beta, row, false);
  std::fill_n(distance, row, 0);

#pragma omp parallel for
  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    alpha[col[i]] = true;
    distance[col[i]] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      is_converged =
          DAWN::BFS_CPU::SOVMP(row_ptr, col, row, alpha, beta, distance, step);
    else
      is_converged =
          DAWN::BFS_CPU::SOVMP(row_ptr, col, row, beta, alpha, distance, step);
    if (is_converged) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_time_tmp = end - start;
  elapsed_time += elapsed_time_tmp.count();
  distance[source] = 0;

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

float DAWN::BFS_CPU::BFS_kernel(int* row_ptr,
                                int* col,
                                int row,
                                int source,
                                bool print,
                                std::string& output_path) {
  int step = 1;
  int entry = row_ptr[source + 1] - row_ptr[source];
  int* alpha = new int[row];
  int* beta = new int[row];
  int* distance = new int[row];
  float elapsed_time = 0.0f;

  std::fill_n(distance, row, 0);
  std::fill_n(alpha, row, 0);
  std::fill_n(beta, row, 0);

  for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) {
    distance[col[i]] = 1;
    alpha[i - row_ptr[source]] = col[i];
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (step < row) {
    step++;
    if (!(step % 2))
      entry = DAWN::BFS_CPU::SOVM(row_ptr, col, row, alpha, beta, distance,
                                  step, entry);
    else
      entry = DAWN::BFS_CPU::SOVM(row_ptr, col, row, beta, alpha, distance,
                                  step, entry);
    if (!entry) {
      break;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_time_tmp = end - start;
  elapsed_time += elapsed_time_tmp.count();

  distance[source] = 0;

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

int DAWN::BFS_CPU::SOVM(int* row_ptr,
                        int* col,
                        int row,
                        int*& alpha,
                        int*& beta,
                        int*& distance,
                        int step,
                        int entry) {
  (void)row;
  int tmpEntry = 0;
  for (int j = 0; j < entry; j++) {
    int start = row_ptr[alpha[j]];
    int end = row_ptr[alpha[j] + 1];
    if (start != end) {
      for (int k = start; k < end; k++) {
        if (!distance[col[k]]) {
          distance[col[k]] = step;
          beta[tmpEntry] = col[k];
          ++tmpEntry;
        }
      }
    }
  }
  return tmpEntry;
}

bool DAWN::BFS_CPU::SOVMP(int* row_ptr,
                          int* col,
                          int row,
                          bool*& alpha,
                          bool*& beta,
                          int*& distance,
                          int step) {
  bool is_converged = true;
#pragma omp parallel for
  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      int start = row_ptr[j];
      int end = row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (!distance[col[k]]) {
            distance[col[k]] = step;
            beta[col[k]] = true;
            is_converged = false;
          }
        }
      }
      alpha[j] = false;
    }
  }
  return is_converged;
}
