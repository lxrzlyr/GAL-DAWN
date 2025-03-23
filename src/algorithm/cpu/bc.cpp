/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-04-21
 *
 * @copyright Copyright (c) 2024
 */
#include <dawn/algorithm/cpu/bc.hxx>

float DAWN::BC_CPU::Betweenness_Centrality(DAWN::Graph::Graph_t& graph,
                                           std::string& output_path) {
  auto row = graph.rows;
  float* bc = new float[row]();
  float elapsed_time = 0.0f;
  int num_threads;
#pragma omp parallel
  { num_threads = omp_get_num_threads(); }
// std::cout << "num_threads =" << num_threads << std::endl;
#pragma omp parallel for reduction(+ : bc[:row]) reduction(+ : elapsed_time)
  for (int i = 0; i < row; i++) {
    float* bc_temp = new float[row]();
    float elapsed_time_tmp = DAWN::BC_CPU::Betweenness_Centrality_run(
        graph.csr.row_ptr, graph.csr.col, i, row, bc_temp);

    for (int j = 0; j < row; j++) {
      bc[j] += bc_temp[j];
    }

    elapsed_time += elapsed_time_tmp;

    delete[] bc_temp;
    bc_temp = nullptr;
  }

  for (int i = 0; i < row; i++) {
    if (!graph.directed) {
      bc[i] = bc[i] / 2.0;
    }
    bc[i] = bc[i] / ((row - 1) * (row - 2));
  }

  DAWN::IO::outfile(row, bc, output_path);

  delete[] bc;
  bc = nullptr;

  return (elapsed_time / (1000 * num_threads));
}

float DAWN::BC_CPU::Betweenness_Centrality_run(int* row_ptr,
                                               int* col,
                                               int node,
                                               int n,
                                               float*& bc) {
  int step = 1;
  bool is_converged = false;
  std::vector<std::vector<int>> Pretree(n);
  std::vector<bool> alpha(n, false);
  std::vector<bool> beta(n, false);
  std::vector<int> distance(n, -1);
  std::vector<float> visit(n, 0.0f);
  std::vector<float> bc_temp(n, 0.0f);
  std::vector<int> prequeue(n, 0);

  distance[node] = 0;
  visit[node] = 1.0f;
  int index = 0;
  prequeue[index] = node;
  for (int i = row_ptr[node]; i < row_ptr[node + 1]; i++) {
    alpha[col[i]] = true;
    distance[col[i]] = 1;
    visit[col[i]] += visit[node];
  }
  auto start = std::chrono::high_resolution_clock::now();
  while (step < n) {
    step++;
    if (!(step % 2)) {
      is_converged =
          DAWN::BC_CPU::kernel(row_ptr, col, n, alpha, beta, distance, Pretree,
                               visit, prequeue, index, step);
    } else {
      is_converged =
          DAWN::BC_CPU::kernel(row_ptr, col, n, beta, alpha, distance, Pretree,
                               visit, prequeue, index, step);
    }
    if (is_converged) {
      break;
    }
  }
  for (int i = index; i >= 0; i--) {
    int w = prequeue[i];
    for (int v : Pretree[w]) {
      bc_temp[v] += (visit[v] / visit[w]) * (1.0 + bc_temp[w]);
    }
    if (w != node) {
      bc[w] += bc_temp[w];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_time_tmp = end - start;

  return elapsed_time_tmp.count();
}

bool DAWN::BC_CPU::kernel(int* row_ptr,
                          int* col,
                          int row,
                          std::vector<bool>& alpha,
                          std::vector<bool>& beta,
                          std::vector<int>& distance,
                          std::vector<std::vector<int>>& Pretree,
                          std::vector<float>& visit,
                          std::vector<int>& prequeue,
                          int& index,
                          int step) {
  bool is_converged = true;

  for (int j = 0; j < row; j++) {
    if (alpha[j]) {
      alpha[j] = false;
      index++;
      prequeue[index] = j;
      int start = row_ptr[j];
      int end = row_ptr[j + 1];
      if (start != end) {
        for (int k = start; k < end; k++) {
          if (distance[col[k]] < 0) {
            beta[col[k]] = true;
            distance[col[k]] = step;
            is_converged = false;
          }
          if (distance[col[k]] == distance[j] + 1) {
            Pretree[col[k]].push_back(j);
            visit[col[k]] += visit[j];
          }
        }
      }
    }
  }
  return is_converged;
}
