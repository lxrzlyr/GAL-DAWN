/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#pragma once

#include <dawn/result.hxx>
#include <dawn_internal/graph.hxx>
#include <dawn_internal/io.hxx>

#include <string>

namespace DAWN {
namespace BFS_CPU {

DAWN::RunResult run(Graph::Graph_t& graph, std::string& output_path);

float BFS(int* row_ptr,
          int* col,
          int row,
          int source,
          bool print,
          std::string& output_path);

float BFS_kernel(int* row_ptr,
                 int* col,
                 int row,
                 int source,
                 bool print,
                 std::string& output_path);

int SOVM(int* row_ptr,
         int* col,
         int row,
         int*& alpha,
         int*& beta,
         int*& distance,
         int step,
         int entry);

bool SOVMP(int* row_ptr,
           int* col,
           int row,
           bool*& alpha,
           bool*& beta,
           int*& distance,
           int step);

}  // namespace BFS_CPU
}  // namespace DAWN
